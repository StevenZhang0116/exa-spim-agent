"""Conservative data-flow checks for repeated work in generated feature code.

This is a bounded abstract interpreter, not a Python complexity proof. It tracks
graph-array axes through aliases, slices, local calls and instance attributes.
Confirmed global work in row processing is rejected. Variable-size local work
and unresolved calls are reported separately, never certified as constant cost.
No generated code or real data is executed by this module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


GLOBAL = frozenset({"nodes", "edges", "candidates"})
VARIABLE = GLOBAL | {"component", "density", "unknown"}
REDUCTIONS = frozenset({
    "min", "max", "amin", "amax", "argmin", "argmax", "sum", "mean",
    "median", "percentile", "quantile", "std", "var", "any", "all",
})
SCANS = frozenset({"sort", "argsort", "unique", "where", "flatnonzero",
                   "nonzero", "bincount", "cumsum", "cumprod"})
INDEXES = frozenset({"KDTree", "cKDTree", "BallTree", "NearestNeighbors",
                     "KMeans", "MiniBatchKMeans", "GaussianMixture"})
ELEMENTWISE = frozenset({"abs", "absolute", "clip", "sqrt", "square", "log",
                        "log1p", "exp", "isfinite", "isnan", "isinf",
                        "minimum", "maximum", "sin", "cos"})
ARRAY_WRAPPERS = frozenset({"asarray", "array", "copy", "astype", "reshape",
                           "ravel", "flatten", "squeeze", "transpose"})
UNKNOWN_LITERAL = object()


@dataclass
class Value:
    axes: tuple[str, ...] = ()
    source: str = ""
    deps: frozenset[str] = frozenset()
    count: str | None = None
    fields: dict[str, "Value"] = field(default_factory=dict)
    items: tuple["Value", ...] = ()
    cls: str = ""
    function: ast.AST | None = None
    closure: dict[str, "Value"] | None = None
    cache: dict[str, "Value"] | None = None
    created_in_row: bool = False
    literal: object = UNKNOWN_LITERAL


def combined(values, *, axes=None):
    values = list(values)
    return Value(
        axes=tuple(axes) if axes is not None else next(
            (v.axes for v in values if set(v.axes) & GLOBAL),
            next((v.axes for v in values if v.axes), ())),
        source="; ".join(dict.fromkeys(v.source for v in values if v.source)),
        deps=frozenset().union(*(v.deps for v in values)),
    )


@dataclass
class CostReport:
    violations: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    @property
    def status(self):
        return "rejected" if self.violations else "partial" if self.unverified else "checked"


class Analyzer:
    def __init__(self, tree, max_steps=30000):
        self.report = CostReport()
        self.classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        # These modules are also supplied by the reviewed runtime template to
        # fragments that do not repeat its imports.
        self.env = {name: Value(cls="@module", source=module) for name, module in
                    (("np", "numpy"), ("nx", "networkx"), ("math", "math"))}
        self.row = False
        self.cached = False
        self.conditional = False
        self.chain = []
        self.active = set()
        self.steps = max_steps
        self.stopped = False
        self.timed_functions = {"start_analysis"}
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        # Detect the row path even when samples are supplied by an opaque helper.
        for _ in range(len(functions) + 1):
            before = len(self.timed_functions)
            for fn in functions:
                if any(isinstance(n, ast.Call) and self.name(n.func) in self.timed_functions
                       for n in ast.walk(fn)):
                    self.timed_functions.add(fn.name)
            if len(self.timed_functions) == before:
                break
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.env[n.name] = Value(function=n, closure=self.env)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                self.imports(n, self.env)
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
                for target in n.targets:
                    self.bind(target, self.expr(n.value, self.env), self.env)

    @staticmethod
    def imports(node, env):
        for alias in node.names:
            if isinstance(node, ast.Import):
                env[alias.asname or alias.name.split('.')[0]] = Value(
                    cls="@module", source=alias.name if alias.asname else alias.name.split('.')[0])
            else:
                env[alias.asname or alias.name] = Value(
                    cls="@external", source=f"{node.module}.{alias.name}")

    @staticmethod
    def name(node):
        return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""

    def note(self, node, message, rejected=False):
        target = self.report.violations if rejected else self.report.unverified
        text = f"line {getattr(node, 'lineno', '?')}: {message}; path: {' -> '.join(self.chain)}"
        if text not in target:
            target.append(text)

    def work(self, node, operation, value):
        if not self.row:
            return
        scales = set(value.axes)
        if scales & GLOBAL:
            self.note(node, f"repeated global {operation} on {value.source or 'graph-derived data'} "
                      f"(axes={value.axes}); compute the derived result before row processing", True)
        elif scales & VARIABLE and not self.cached:
            self.note(node, f"{operation} has {sorted(scales & VARIABLE)} cost; "
                      "a bound or persistent cache has not been verified")

    def bind(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, part in enumerate(target.elts):
                self.bind(part, value.items[i] if i < len(value.items) else value, env)
        elif isinstance(target, ast.Attribute):
            if self.conditional:
                self.note(target, "conditional object mutation; field-state merging is unverified")
            self.expr(target.value, env).fields[target.attr] = value

    def invoke(self, function, args, kwargs, node):
        fn = function.function
        if fn is None:
            return Value()
        if id(fn) in self.active:
            if self.row:
                self.note(node, f"recursive call {getattr(fn, 'name', 'lambda')} is unverified")
            return Value(axes=("unknown",))
        env = dict(function.closure or self.env)
        params = [*fn.args.posonlyargs, *fn.args.args]
        defaults = [None] * (len(params) - len(fn.args.defaults)) + list(fn.args.defaults)
        for i, (param, default) in enumerate(zip(params, defaults)):
            env[param.arg] = args[i] if i < len(args) else kwargs.get(
                param.arg, self.expr(default, env) if default is not None else Value())
        for param, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            env[param.arg] = kwargs.get(param.arg, self.expr(default, env))
        self.active.add(id(fn))
        self.chain.append(getattr(fn, "name", "lambda"))
        try:
            if isinstance(fn, ast.Lambda):
                return self.expr(fn.body, env)
            return self.block(fn.body, env)
        finally:
            self.chain.pop()
            self.active.remove(id(fn))

    def slice_value(self, node, base, env):
        index = node.slice
        if base.items and isinstance(index, ast.Constant) and isinstance(index.value, int):
            return base.items[index.value] if -len(base.items) <= index.value < len(base.items) else Value()
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            if index.value == "fragments_graph":
                return Value(cls="@graph", source="payload.fragments_graph")
            if index.value in base.fields:
                return base.fields[index.value]
            deps = frozenset({f"{d}[{index.value}]" for d in base.deps})
            return Value(source=f"{base.source}[{index.value}]", deps=deps)
        parts = list(index.elts) if isinstance(index, ast.Tuple) else [index]
        axes = list(base.axes)
        result = []
        for part in parts:
            if isinstance(part, ast.Constant) and part.value is Ellipsis:
                take = max(0, len(axes) - (len(parts) - len(result) - 1))
                result.extend(axes[:take])
                axes = axes[take:]
            elif isinstance(part, ast.Constant) and part.value is None:
                result.append("fixed")
            elif isinstance(part, ast.Slice):
                axis = axes.pop(0) if axes else "unknown"
                # Only a statically bounded slice is independent of graph size.
                fixed = (isinstance(part.upper, ast.Constant) and isinstance(part.upper.value, int)
                         and part.upper.value >= 0
                         and (part.lower is None or isinstance(part.lower, ast.Constant)
                              and isinstance(part.lower.value, int) and part.lower.value >= 0)
                         and (part.step is None or isinstance(part.step, ast.Constant)
                              and isinstance(part.step.value, int) and part.step.value > 0))
                result.append("fixed" if fixed else axis)
            else:
                ix = self.expr(part, env)
                if axes:
                    axes.pop(0)
                result.extend(ix.axes)
        result.extend(axes)
        ix = self.expr(index, env)
        # Masks and fancy indexing still copy/process variable-size selections.
        if ix.axes:
            self.work(node, "index/filter", ix)
        return Value(axes=tuple(result), source=base.source, deps=base.deps | ix.deps)

    def expr(self, node, env):
        if node is None or self.stopped:
            return Value()
        self.steps -= 1
        if self.steps < 0:
            self.note(node, "analysis budget exhausted; remaining paths are unverified")
            self.stopped = True
            return Value()
        if isinstance(node, ast.Name):
            return env.get(node.id, Value(axes=("unknown",), source=node.id))
        if isinstance(node, ast.Constant):
            return Value(source=repr(node.value), literal=node.value)
        if isinstance(node, ast.NamedExpr):
            value = self.expr(node.value, env)
            self.bind(node.target, value, env)
            return value
        if isinstance(node, ast.Lambda):
            return Value(function=node, closure=dict(env))
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            vals = tuple(self.expr(n, env) for n in node.elts)
            v = combined(vals, axes=("fixed",))
            v.items = vals
            if isinstance(node, ast.Tuple) and all(x.literal is not UNKNOWN_LITERAL for x in vals):
                v.literal = tuple(x.literal for x in vals)
            return v
        if isinstance(node, ast.Dict):
            vals = [self.expr(n, env) for n in node.values]
            v = combined(vals)
            v.cache = {}
            v.created_in_row = self.row
            for key, val in zip(node.keys, vals):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    v.fields[key.value] = val
            return v
        if isinstance(node, ast.Attribute):
            base = self.expr(node.value, env)
            if base.cls in {"@module", "@external"}:
                return Value(cls="@external", source=f"{base.source}.{node.attr}")
            if node.attr in base.fields:
                return base.fields[node.attr]
            if base.cls == "@graph":
                axes = {"node_xyz": ("nodes", "fixed"), "node_radius": ("nodes",),
                        "node_component_id": ("nodes",)}.get(node.attr)
                if axes:
                    return Value(axes=axes, source=f"{base.source}.{node.attr}")
            if base.cls in self.classes:
                method = next((n for n in self.classes[base.cls].body
                               if isinstance(n, ast.FunctionDef) and n.name == node.attr), None)
                if method:
                    return Value(function=method, closure=self.env, items=(base,))
            if node.attr == "shape":
                return Value(items=tuple(Value(count=a, source=base.source) for a in base.axes))
            if node.attr == "T":
                return Value(axes=tuple(reversed(base.axes)), source=base.source, deps=base.deps)
            return Value(axes=("unknown",), source=f"{base.source}.{node.attr}", deps=base.deps)
        if isinstance(node, ast.Subscript):
            return self.slice_value(node, self.expr(node.value, env), env)
        if isinstance(node, ast.Call):
            return self.call(node, env)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            local = dict(env)
            axes = []
            for gen in node.generators:
                val = self.expr(gen.iter, local)
                self.work(node, "comprehension", val)
                axes.extend(val.axes[:1])
                self.bind(gen.target, Value(axes=val.axes[1:], source=val.source, deps=val.deps), local)
                for cond in gen.ifs:
                    self.expr(cond, local)
            result = self.expr(node.value if isinstance(node, ast.DictComp) else node.elt, local)
            if isinstance(node, ast.DictComp):
                result = combined([result, self.expr(node.key, local)])
            return combined([result], axes=tuple(axes) + result.axes)
        values = [self.expr(n, env) for n in ast.iter_child_nodes(node) if isinstance(n, ast.expr)]
        result = combined(values)
        if isinstance(node, (ast.BinOp, ast.Compare, ast.UnaryOp, ast.BoolOp)):
            self.work(node, "array expression", result)
        return result

    def call(self, node, env):
        if any(isinstance(n, ast.Starred) for n in node.args) or any(k.arg is None for k in node.keywords):
            self.note(node, "dynamic argument unpacking is unverified")
        name = self.name(node.func)
        fn = self.expr(node.func, env)
        if fn.cls == "@external":
            name = fn.source.rsplit('.', 1)[-1]
        args = [self.expr(n, env) for n in node.args]
        kwargs = {k.arg: self.expr(k.value, env) for k in node.keywords}
        base = self.expr(node.func.value, env) if isinstance(node.func, ast.Attribute) else Value()
        if fn.function is not None:
            return self.invoke(fn, [*fn.items, *args], kwargs, node)
        if name in self.classes:
            instance = Value(cls=name, source=name)
            init = next((n for n in self.classes[name].body
                         if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            if init:
                self.invoke(Value(function=init, closure=self.env), [instance, *args], kwargs, node)
            return instance
        if name == "FeatureAccumulator":
            return Value(cls="@feature_accumulator", source=name)
        if name == "compute" and base.cls == "@feature_accumulator":
            callback = args[2] if len(args) >= 3 else kwargs.get("evaluate", Value())
            previous = self.row
            self.row = True
            # all_compatible can pass a variable-length tuple. Without the
            # inventory here, direct iteration cannot be certified fixed-size.
            context = Value(axes=("unknown",), source="split feature context",
                            deps=frozenset({"row:split_context"}))
            if callback.function is None:
                self.note(node, "split evaluator cannot be resolved; formula cost is unverified")
            result = self.invoke(callback, [*callback.items, context], {}, node)
            self.row = previous
            return result
        if name == "build_sample_universe":
            return Value(items=(Value(axes=("candidates",), source="candidate universe"),
                                Value(axes=("candidates",), source="labels")))
        if name == "_memoized" and len(args) >= 3:
            cache, key, callback = args[:3]
            signature = repr(key.literal) if key.literal is not UNKNOWN_LITERAL else None
            if cache.cache is not None and signature is not None and signature in cache.cache:
                return cache.cache[signature]
            previous = self.cached
            persistent = cache.cache is not None and not cache.created_in_row
            self.cached = persistent
            result = self.invoke(callback, [], {}, node)
            self.cached = previous
            if self.row:
                # A row-local cache is legitimate for image patches/local work.
                # It does not discharge variable-size cost checks across rows:
                # self.cached remained False while evaluating its callback.
                omitted = result.deps - key.deps
                if omitted:
                    self.note(node, f"memoization key omits row-dependent inputs: {sorted(omitted)}", True)
            elif cache.cache is not None and not self.conditional and signature is not None:
                cache.cache[signature] = result
            return result
        if name == "_bounded_thread_map" and len(args) >= 2:
            previous = self.row
            self.row = True
            item = Value(source="worker item", deps=frozenset({"row:worker"}))
            result = self.invoke(args[0], [item], {}, node)
            self.row = previous
            return result
        if name in {"number_of_nodes", "number_of_edges"}:
            return Value(count="nodes" if name.endswith("nodes") else "edges", source=base.source)
        if name == "len" and args:
            return Value(count=args[0].axes[0] if args[0].axes else None, source=args[0].source)
        if name in {"int", "float", "bool", "str"}:
            return args[0] if args else Value()
        if name == "range":
            return Value(axes=(next((v.count for v in args if v.count), "fixed"),),
                         source=combined(args).source)
        if name in {"list", "tuple", "set", "enumerate", "reversed"} and args:
            if name in {"list", "tuple", "set"}:
                self.work(node, name + " materialization", args[0])
            return args[0]
        if name == "dict":
            return Value(cache={}, created_in_row=self.row)
        if base.cache is not None and name in {"clear", "pop", "popitem", "update"}:
            # Mutation invalidates any proof that a global cache is prewarmed.
            base.cache.clear()
            return Value(axes=("unknown",) if name in {"pop", "popitem"} else ())
        if name in {"query_ball_point", "query_ball_tree", "neighbors"}:
            return Value(axes=("density",), source=name, deps=combined(args).deps)
        if name in {"connected_components", "node_connected_component"}:
            return Value(axes=("component",), source=name, deps=combined(args).deps)
        values = [*args, *kwargs.values()]
        if base.axes and base.cls not in {"@module", "@external"}:
            values.insert(0, base)
        value = combined(values)
        if name in REDUCTIONS | SCANS | INDEXES | ELEMENTWISE | ARRAY_WRAPPERS or name == "sorted":
            # asarray may be zero-copy; other array transforms can traverse data.
            if name != "asarray":
                self.work(node, name, value)
            if name in REDUCTIONS:
                # A pre-pass reduction over coordinates can still leave a
                # whole-node axis (e.g. mean(xyz, axis=1)). Keep that provenance.
                method = bool(base.axes) and base.cls not in {"@module", "@external"}
                target = base if method else args[0] if args else kwargs.get("a", Value())
                axis_node = next((k.value for k in node.keywords if k.arg == "axis"), None)
                axis_pos = 0 if method else 2 if name in {"percentile", "quantile"} else 1
                if axis_node is None and len(node.args) > axis_pos:
                    axis_node = node.args[axis_pos]
                axes = ()
                if axis_node is not None:
                    axis = self.expr(axis_node, env).literal
                    if axis is UNKNOWN_LITERAL:
                        axes = ("unknown",)
                    elif axis is not None:
                        selected = axis if isinstance(axis, tuple) else (axis,)
                        if target.axes and all(isinstance(x, int) for x in selected):
                            indices = {x % len(target.axes) for x in selected}
                            axes = tuple(a for i, a in enumerate(target.axes) if i not in indices)
                return combined(values, axes=axes)
            if name in INDEXES:
                return Value(source=name, deps=value.deps)
            return value
        if name in {"dot", "norm", "cross"}:
            self.work(node, name, value)
            return combined(values, axes=())
        benign = {"start_analysis", "stop_analysis", "start_phase", "stop_phase", "set",
                  "append", "add", "print", "sample_display", "compatible_occurrences"}
        if self.row:
            # Timing, recording and logging calls do not inspect feature arrays.
            if name not in benign:
                self.note(node, f"unresolved call {name or '<dynamic>'}; argument cost is unverified")
        return Value(axes=() if name in benign else ("unknown",),
                     source=f"return of {name}", deps=value.deps)

    def block(self, statements, env):
        returns = []
        for stmt in statements:
            if self.stopped:
                break
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                env[stmt.name] = Value(function=stmt, closure=env)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self.imports(stmt, env)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = self.expr(stmt.value, env)
                for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
                    self.bind(target, value, env)
            elif isinstance(stmt, ast.AugAssign):
                value = combined([self.expr(stmt.target, env), self.expr(stmt.value, env)])
                self.work(stmt, "augmented array expression", value)
                self.bind(stmt.target, value, env)
            elif isinstance(stmt, ast.Return):
                returns.append(self.expr(stmt.value, env))
                break
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                seq = self.expr(stmt.iter, env)
                previous = self.row
                timed = any(isinstance(n, ast.Call) and self.name(n.func) in self.timed_functions
                            for s in stmt.body for n in ast.walk(s))
                self.row = previous or "candidates" in seq.axes or timed
                if previous:
                    self.work(stmt, "iteration", seq)
                local = dict(env)
                deps = frozenset({f"row:{ast.unparse(stmt.target)}"}) if self.row else seq.deps
                self.bind(stmt.target, Value(axes=seq.axes[1:], source=seq.source, deps=deps), local)
                self.block(stmt.body, local)
                self.row = previous
                self.block(stmt.orelse, env)
            elif isinstance(stmt, ast.If):
                condition = self.expr(stmt.test, env)
                old = self.conditional
                self.conditional = True
                left, right = dict(env), dict(env)
                a, b = self.block(stmt.body, left), self.block(stmt.orelse, right)
                result = combined([a, b])
                result.deps |= condition.deps
                returns.append(result)
                for key in left.keys() | right.keys():
                    lv, rv = left.get(key), right.get(key)
                    if lv is rv:
                        env[key] = lv
                    elif lv is not None and rv is not None:
                        env[key] = combined([lv, rv])
                self.conditional = old
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    self.expr(item.context_expr, env)
                returns.append(self.block(stmt.body, env))
            elif isinstance(stmt, ast.Try):
                old = self.conditional
                self.conditional = True
                for body in [stmt.body, *(h.body for h in stmt.handlers), stmt.orelse, stmt.finalbody]:
                    returns.append(self.block(body, dict(env)))
                self.conditional = old
            elif isinstance(stmt, ast.While):
                self.expr(stmt.test, env)
                if self.row:
                    self.note(stmt, "while-loop bound is unverified")
                returns.append(self.block(stmt.body, dict(env)))
            elif isinstance(stmt, ast.Expr):
                self.expr(stmt.value, env)
            elif isinstance(stmt, ast.Assert):
                self.expr(stmt.test, env)
                self.expr(stmt.msg, env)
            elif isinstance(stmt, ast.Delete):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        env.pop(target.id, None)
                    elif isinstance(target, ast.Subscript):
                        owner = self.expr(target.value, env)
                        if owner.cache is not None:
                            owner.cache.clear()
            elif isinstance(stmt, ast.Raise):
                self.expr(stmt.exc, env)
            elif isinstance(stmt, (ast.Pass, ast.Break, ast.Continue, ast.Global, ast.Nonlocal)):
                pass
            elif self.row:
                self.note(stmt, f"statement {type(stmt).__name__} is unverified")
        if len(returns) == 1:
            return returns[0]
        return combined(returns)


def analyze_row_cost(tree: ast.Module, max_steps: int = 30000) -> CostReport:
    analyzer = Analyzer(tree, max_steps)
    extract = analyzer.env.get("extract_features")
    if extract is not None:
        analyzer.invoke(extract, [Value(source="payload")], {}, extract.function)
    return analyzer.report
