"""Deny-by-default file-tool boundary. Not a Python execution sandbox."""

from pathlib import Path
import json


def canonical_file(path, cwd):
    """Normalize mount/parent aliases, but do NOT trust a symlink at the file."""
    path = Path(path)
    path = path if path.is_absolute() else Path(cwd) / path
    return path.parent.resolve() / path.name


def make_guard(cwd, editable_dir, readable_paths, audit_path, state=None, *,
               policy_filename="scorer.py"):
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
    readable = writable | {canonical_file(p, cwd) for p in readable_paths}
    state = state if state is not None else {"violations": []}

    async def guard(tool_name, tool_input, context):
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
                                    message="Only the current policy/rules are writable; "
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
