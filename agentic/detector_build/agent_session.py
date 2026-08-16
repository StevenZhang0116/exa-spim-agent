"""Lazy Claude SDK loading for the parts of the workflow that need an agent."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import AsyncIterator, Callable


@dataclass(frozen=True)
class ClaudeSDKBindings:
    assistant_message: type
    agent_options: type
    client: type
    result_message: type
    text_block: type
    tool_use_block: type | tuple


@lru_cache(maxsize=1)
def load_claude_sdk() -> ClaudeSDKBindings:
    """Import the optional SDK only when an agent session is about to start."""
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
        )
    except ImportError as exc:
        raise SystemExit(
            "claude_agent_sdk is required to run detector generation. Install it "
            "in the workflow environment; pure input validation and --help do not "
            "require it."
        ) from exc
    try:
        from claude_agent_sdk import ToolUseBlock
    except ImportError:  # pragma: no cover - depends on installed SDK version
        ToolUseBlock = ()
    return ClaudeSDKBindings(
        assistant_message=AssistantMessage,
        agent_options=ClaudeAgentOptions,
        client=ClaudeSDKClient,
        result_message=ResultMessage,
        text_block=TextBlock,
        tool_use_block=ToolUseBlock,
    )


@asynccontextmanager
async def open_agent_session(
    client_factory: Callable[..., object],
    *,
    options: object,
    connect_timeout_s: int,
    close_timeout_s: int = 30,
) -> AsyncIterator[object]:
    """Open and close an SDK client with bounded lifecycle waits.

    A per-turn timeout cannot protect the workflow while ``__aenter__`` is
    starting the Claude subprocess or establishing its transport.  Keep that
    infrastructure concern here so every workflow stage sees either a usable
    client or a prompt, explicit failure.
    """
    manager = client_factory(options=options)
    try:
        client = await asyncio.wait_for(
            manager.__aenter__(), timeout=connect_timeout_s
        )
    except asyncio.TimeoutError as exc:
        raise SystemExit(
            "Claude SDK session did not open within "
            f"{connect_timeout_s}s. Check network/API access and retry; no "
            "workflow stage was started."
        ) from exc

    try:
        yield client
    finally:
        try:
            await asyncio.wait_for(
                manager.__aexit__(None, None, None), timeout=close_timeout_s
            )
        except asyncio.TimeoutError:
            # The build outputs have already been validated at this point. A
            # stuck transport shutdown must not turn them into a false failure.
            pass
