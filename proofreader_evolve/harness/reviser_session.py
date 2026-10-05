"""Scorer reviser SDK configuration.

SDK imports and credentials are needed only when building a live session.
File-tool guards are not an OS sandbox for executing generated Python.
"""

from dataclasses import replace
import os
from pathlib import Path

from .reviser_access import canonical_file, make_guard, pre_tool_hook


DEFAULT_MODEL = "claude-opus-5"


def anthropic_api_env():
    """Use the configured Anthropic API key, not inherited Bedrock/Vertex routes."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; export it before running evolution")
    return {
        "ANTHROPIC_API_KEY": api_key,
        "CLAUDE_CODE_USE_BEDROCK": "0",
        "CLAUDE_CODE_USE_VERTEX": "0",
        "CLAUDE_EFFORT": "high",
    }


def build_options(*, run_dir, system_prompt, model=DEFAULT_MODEL,
                  readable_paths=(), audit_path=None, max_turns=None):
    """Build a deny-by-default session; bind generation artifacts before use."""
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    if run_dir is None:
        raise ValueError("A run directory is required for reviser isolation")
    run_dir = Path(run_dir).resolve()
    audit_path = audit_path or run_dir / "tool_audit.jsonl"
    guard, state = make_guard(run_dir, run_dir / "artifacts", readable_paths, audit_path)
    options = ClaudeAgentOptions(
        cwd=str(run_dir), setting_sources=[], skills=[], strict_mcp_config=True,
        tools=["Read", "Write", "Edit"], allowed_tools=["Read", "Write", "Edit"],
        model=model, env=anthropic_api_env(), system_prompt=system_prompt,
        max_turns=max_turns, permission_mode="default", can_use_tool=guard,
        hooks={"PreToolUse": [HookMatcher(hooks=[pre_tool_hook(guard)])]},
    )
    options._isolation_state = state
    return options, state


def bind_session_options(options, policy_path, rules_path, report_path, *, readable_paths=(), training_server=None):
    """Bind generation files and TRAIN tools, including analysis and descriptor code; explicit reads only."""
    from claude_agent_sdk import HookMatcher

    run_dir = Path(options.cwd).resolve()
    policy = canonical_file(policy_path, run_dir)
    rules = canonical_file(rules_path, run_dir)
    if (policy.name != "scorer.py"
            or rules != policy.parent / "rules.md" or run_dir not in policy.parents):
        raise ValueError("Reviser artifacts must be policy/rules in this run")
    readable = list(readable_paths)
    if report_path is not None:
        readable.append(report_path)
    training_tools = (["mcp__training__evaluate_train", "mcp__training__search_parameters",
                       "mcp__training__train_classifier", "mcp__training__inspect_candidate",
                       "mcp__training__inspect_failure_cases", "mcp__training__evaluate_feature_ablation",
                       "mcp__training__inspect_candidate_image", "mcp__training__inspect_failure_images",
                       "mcp__training__run_volume_analysis", "mcp__training__plan_image_scoring",
                       "mcp__training__plan_descriptor_run", "mcp__training__compute_descriptors",
                       "mcp__training__restore_candidate", "mcp__training__search_memory"]
                      if training_server is not None else [])
    guard, state = make_guard(
        run_dir, policy.parent, readable, run_dir / "tool_audit.jsonl",
        state=getattr(options, "_isolation_state", None), policy_filename=policy.name,
        allowed_tools=training_tools, allow_proposal=training_server is not None,
    )
    bound = replace(options, can_use_tool=guard, permission_mode="default",
                    hooks={"PreToolUse": [HookMatcher(hooks=[pre_tool_hook(guard)])]},
                    allowed_tools=["Read", "Write", "Edit", *training_tools],
                    mcp_servers={"training": training_server} if training_server is not None else {})
    bound._isolation_state = state
    return bound
