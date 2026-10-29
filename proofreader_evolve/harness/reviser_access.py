"""Deny-by-default file-tool boundary. Not a Python execution sandbox.

Reads are limited to the bound generation artifacts, explicitly listed reports,
current/earlier generations' TRAIN artefact directories (attempt snapshots, classifier
fits, descriptor runs/probes; regular files only) and the SDK's parked tool-result files for this
run; writes to the editable artifacts. Validation descriptor logs, including the
legacy descriptor_runs/inference subtree, are always private to the host.
"""

from pathlib import Path
import json
import os
import re


def sdk_project_dir(cwd):
    """Claude Code keeps per-session files for a working directory under
    <config dir>/projects/<cwd with every non-alphanumeric character as "-">/.
    Oversized MCP results are parked in <session>/tool-results/ there."""
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    return (config / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(Path(cwd).resolve()))).resolve()


def canonical_file(path, cwd):
    """Normalize mount/parent aliases, but do NOT trust a symlink at the file."""
    path = Path(path)
    path = path if path.is_absolute() else Path(cwd) / path
    return path.parent.resolve() / path.name


def _private_descriptor_log(path, run_dir):
    """Reserve both current and legacy inference log trees in every generation."""
    try:
        parts = path.relative_to(run_dir).parts
    except ValueError:
        return False
    return (len(parts) >= 2 and re.fullmatch(r"gen[0-9]+", parts[0]) is not None
            and (parts[1] == "validation_descriptor_runs"
                 or parts[1:3] == ("descriptor_runs", "inference")))


def make_guard(cwd, editable_dir, readable_paths, audit_path, state=None, *,
               policy_filename="scorer.py", allowed_tools=(), allow_proposal=False,
               sdk_tool_results=True, readable_dirs=()):
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    cwd = Path(cwd).resolve()
    editable_dir = Path(editable_dir)
    editable_dir = (editable_dir if editable_dir.is_absolute() else cwd / editable_dir).resolve()
    if editable_dir != cwd and cwd not in editable_dir.parents:
        raise ValueError("Editable directory must be inside this run")
    # Pin canonical parent paths so /tmp and shared-mount aliases work. Keep
    # leaves unresolved and recheck every target at EVERY call: a later symlink
    # must not widen this fixed allowlist.
    if policy_filename != "scorer.py":
        raise ValueError("Unsupported policy filename")
    writable = {editable_dir / policy_filename, editable_dir / "rules.md"}
    if allow_proposal:
        writable.update({editable_dir / "proposal.json", editable_dir / "training.py",
                         editable_dir / "analysis.py", editable_dir / "analysis_request.json",
                         editable_dir / "descriptor.py"})
    readable = writable | {canonical_file(p, cwd) for p in readable_paths}
    # Parked MCP tool results are this run's own TRAIN tool output, so regular files
    # directly inside <project>/<session>/tool-results/ are readable (never writable).
    parked_root = sdk_project_dir(cwd) if sdk_tool_results else None
    # Whole directories of this run's own TRAIN artefacts (attempt snapshots, classifier fits,
    # descriptor runs/probes) are readable recursively, except the private inference trees;
    # regular files only, with the resolved path checked to prevent symlink escapes.
    directories = tuple((d if d.is_absolute() else cwd / d).resolve() for d in readable_dirs)
    if any(cwd not in d.parents for d in directories):
        raise ValueError("Readable directories must be inside this run")
    state = state if state is not None else {"violations": []}
    trusted_tools = frozenset(allowed_tools)

    async def guard(tool_name, tool_input, context):
        if tool_name in trusted_tools:
            return PermissionResultAllow(behavior="allow")
        value = (tool_input or {}).get("file_path")
        allowed = False
        if tool_name in {"Read", "Write", "Edit"} and isinstance(value, str) and value:
            try:
                raw = Path(value)
                path = (raw if raw.is_absolute() else cwd / raw).absolute()
                resolved = path.resolve()
                permitted = readable if tool_name == "Read" else writable
                # Require an allowed lexical entry with an unchanged target.
                allowed = any(resolved == p and p.resolve() == p for p in permitted)
                if (not allowed and tool_name == "Read" and directories
                        and any(d in resolved.parents for d in directories)
                        and resolved.is_file() and not path.is_symlink()):
                    allowed = True
                if (not allowed and tool_name == "Read" and parked_root is not None
                        and resolved.parent.name == "tool-results"
                        and resolved.parent.parent.parent == parked_root
                        and resolved.is_file() and not path.is_symlink()):
                    allowed = True
                # A recursive or explicit allowlist must never expose frozen-inference
                # stdout/stderr. Check the target too, so aliases cannot reopen old logs.
                if _private_descriptor_log(path, cwd) or _private_descriptor_log(resolved, cwd):
                    allowed = False
            except (OSError, RuntimeError, ValueError):
                allowed = False
        if allowed:
            return PermissionResultAllow(behavior="allow")
        record = {"tool": tool_name, "paths": [value], "scope": str(editable_dir)}
        state["violations"].append(record)
        try:
            with open(audit_path, "a") as stream:
                stream.write(json.dumps({"DENIED": record}) + "\n")
        except OSError:
            pass
        return PermissionResultDeny(behavior="deny", interrupt=False,
                                    message="Only explicitly allowed current-generation artifacts are writable; "
                                            "reads require an explicitly allowed source or report.")

    return guard, state


def pre_tool_hook(guard):
    async def hook(input_data, tool_use_id, context):
        result = await guard(input_data.get("tool_name"), input_data.get("tool_input"), context)
        output = {"hookEventName": "PreToolUse", "permissionDecision": result.behavior}
        if result.behavior == "deny":
            output["permissionDecisionReason"] = result.message
        return {"hookSpecificOutput": output}
    return hook
