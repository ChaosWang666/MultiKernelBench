"""Adapter for invoking Claude Code via the claude-agent-sdk-python.

Replaces the previous `subprocess.run(['claude', '-p', ...])` pattern in
the generation scripts. Key differences:

* Output is delivered through an in-process MCP tool `submit_kernel` with a
  strict 6-field schema, eliminating the brittle regex extraction of
  `project_json_src` ... `model_src` from stdout.
* Per-task cost / num_turns / duration / limit signals are surfaced as
  structured fields on `KernelResult`.
* Rate-limit / budget events are detected from `RateLimitEvent`,
  `AssistantMessage.error == 'rate_limit'`, and `ResultMessage.subtype`,
  not via stdout heuristics.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeAgentOptions,
    ProcessError,
    RateLimitEvent,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)


EXPECTED_FIELDS = (
    "project_json_src",
    "host_tiling_src",
    "host_operator_src",
    "kernel_src",
    "python_bind_src",
    "model_src",
)


SUBMIT_TOOL_QUALIFIED = "mcp__kernel__submit_kernel"


SUBMIT_TOOL_DIRECTIVE = (
    "你必须且只能通过 `submit_kernel` 工具一次性提交完整结果，"
    "工具的 6 个字段（project_json_src、host_tiling_src、host_operator_src、"
    "kernel_src、python_bind_src、model_src）必须全部填齐。"
    "禁止在工具调用之外输出任何代码片段或解释。"
    "禁止多次调用 `submit_kernel`，只允许一次最终提交。"
)


@dataclass
class KernelResult:
    op: str
    fields: dict[str, str] | None = None
    cost_usd: float = 0.0
    num_turns: int = 0
    duration_s: float = 0.0
    limit_hit: bool = False
    error: str | None = None
    raw_text: str = ""


_LIMIT_SUBTYPES = {
    "error_max_budget_usd",
    "error_max_turns",
    "error_rate_limit",
}


def _is_limit_signal(msg: Any) -> bool:
    if isinstance(msg, RateLimitEvent):
        # SDK emits these as routine quota-status updates with status='allowed';
        # only 'rejected' on the primary status means the request was blocked.
        # overage_status='rejected' just means the account has overage billing
        # disabled — it is NOT a limit hit when primary status is 'allowed'.
        return msg.rate_limit_info.status == "rejected"
    if isinstance(msg, AssistantMessage) and msg.error == "rate_limit":
        return True
    if isinstance(msg, ResultMessage) and msg.subtype in _LIMIT_SUBTYPES:
        return True
    return False


def _make_kernel_server(capture: dict[str, Any]):
    """Build an SDK MCP server whose `submit_kernel` tool writes into the
    supplied capture dict. A fresh server per `run_generation` call keeps
    concurrent generations isolated (no module-level shared state)."""
    schema = {name: str for name in EXPECTED_FIELDS}

    @tool(
        "submit_kernel",
        "Submit the 6 AscendC source files for this operator. Call exactly "
        "once with all 6 fields filled in.",
        schema,
    )
    async def submit_kernel(args: dict[str, Any]) -> dict[str, Any]:
        capture["fields"] = {k: args[k] for k in EXPECTED_FIELDS}
        return {"content": [{"type": "text", "text": "ok"}]}

    return create_sdk_mcp_server(
        name="kernel",
        version="1.0.0",
        tools=[submit_kernel],
    )


async def run_generation(
    prompt: str,
    op: str,
    *,
    timeout: float,
    with_wiki: bool = False,
    disable_skills: bool = False,
    cwd: str | None = None,
    extra_allowed_tools: list[str] | None = None,
) -> KernelResult:
    """Run one Claude Code generation via the SDK.

    Returns a `KernelResult` whose `fields` is None when the model failed
    to submit via the `submit_kernel` tool (timeout, error, or limit hit).

    `extra_allowed_tools` is appended verbatim to the allowed_tools list, so
    callers can opt into skills (e.g. ``Skill(cann-ask)``) or other built-in
    tools without forking the adapter.
    """
    capture: dict[str, Any] = {}
    server = _make_kernel_server(capture)

    if with_wiki:
        allowed_tools = ["Read", "Glob", "Grep", SUBMIT_TOOL_QUALIFIED]
    else:
        allowed_tools = [SUBMIT_TOOL_QUALIFIED]
    if extra_allowed_tools:
        allowed_tools = allowed_tools + list(extra_allowed_tools)

    options = ClaudeAgentOptions(
        cwd=cwd,
        allowed_tools=allowed_tools,
        mcp_servers={"kernel": server},
        permission_mode="bypassPermissions",
        setting_sources=[] if disable_skills else None,
    )

    result = KernelResult(op=op)
    text_chunks: list[str] = []
    start = time.time()

    async def _drive() -> None:
        async for msg in query(prompt=prompt, options=options):
            if _is_limit_signal(msg):
                result.limit_hit = True
            if isinstance(msg, AssistantMessage):
                for blk in msg.content:
                    if isinstance(blk, TextBlock):
                        text_chunks.append(blk.text)
            elif isinstance(msg, ResultMessage):
                result.num_turns = msg.num_turns
                result.cost_usd = msg.total_cost_usd or 0.0
                result.duration_s = msg.duration_ms / 1000.0
                if msg.is_error and msg.errors:
                    result.error = "; ".join(msg.errors)

    try:
        await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        result.error = f"timeout after {timeout}s"
    except CLINotFoundError as e:
        result.error = f"CLINotFoundError: {e}"
    except (CLIConnectionError, CLIJSONDecodeError, ProcessError) as e:
        result.error = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 - surfaced via result.error for log
        result.error = f"{type(e).__name__}: {e}"

    if result.duration_s == 0.0:
        result.duration_s = time.time() - start

    result.raw_text = "".join(text_chunks)
    if "fields" in capture:
        result.fields = capture["fields"]
    return result


def render_kernel_text(fields: dict[str, str]) -> str:
    """Render the 6 fields as a Python source file with triple-quoted string
    assignments, matching the format `eval_single_runner.py` expects when
    it `exec()`s the file."""
    parts: list[str] = []
    for name in EXPECTED_FIELDS:
        body = fields[name]
        if "'''" not in body:
            quote = "'''"
        elif '"""' not in body:
            quote = '"""'
        else:
            body = body.replace("'''", "\\'\\'\\'")
            quote = "'''"
        parts.append(f"{name} = {quote}{body}{quote}\n\n")
    return "".join(parts).rstrip() + "\n"


def append_metric(out_dir: str | Path, result: KernelResult) -> None:
    """Append one JSON line to `{out_dir}/metrics.jsonl` for this op."""
    payload = {
        "op": result.op,
        "cost_usd": round(result.cost_usd, 6),
        "num_turns": result.num_turns,
        "duration_s": round(result.duration_s, 2),
        "limit_hit": result.limit_hit,
        "error": result.error,
        "fields_present": result.fields is not None,
    }
    path = Path(out_dir) / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
