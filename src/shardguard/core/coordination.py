"""
coordination.py

Zero-trust coordinator that:
1) Detects & redacts PII from user input and tool outputs into opaque placeholders like $[VAR_1]
2) Loads *strict* tool schemas from MCP servers registered with the application
3) Uses the OpenAI Responses API function-calling loop for chaining prompts. We use a new instance in the executor to keep least permissions.
4) Resolves placeholders back to real values only at tool execution time
 Responses API tool calling loop:
  https://platform.openai.com/docs/guides/function-calling
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI  # type: ignore

from shardguard.core.execution import McpToolExecutor
from shardguard.core.planning import PlanningLLM
from shardguard.core.prompts import COORDINATION_PROMPT, REDACTION_PROMPT
from shardguard.core.redactor import LlmOpaqueRedactor
from shardguard.core.sanitization import (
    _DOLLAR_KEY_RE,
    _PLACEHOLDER_RE,
    _PLACEHOLDER_TOKEN_RE,
    _as_steps_list,
    _as_str_list,
    _build_tool_def,
    _canonicalize_scalar_placeholders,
    _coerce_types,
    _extract_placeholder_keys,
    _is_allowed_tool,
    _validate_formats,
)
from shardguard.core.tool_args_llm import (
    GeminiToolArgsLLM,
    OllamaToolArgsLLM,
    OpenAIResponsesToolArgsLLM,
    ToolArgsLLM,
)
from shardguard.mcp_servers import registry

logging.basicConfig(
    filename="shardguard_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",  # overwrite log file on each run
    force=True,
)

# Log levels, keeps log file smaller
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    parameters: dict[str, Any]
    server: str


def _augment_schema_for_placeholders(schema: dict[str, Any]) -> dict[str, Any]:
    r"""
    OpenAI strict schemas don't enforce formats like email/date-time strictly,
    and we also need to allow placeholder tokens like $[EMAIL_1] as values

    For scalar fields, wrap the property schema as:
      anyOf: [ <original scalar schema>, {"type":"string","pattern": r"^\$\[[A-Za-z0-9_]+\]$"} ]

    This prevents invalid concatenations from being considered "valid" by the model
    while still allowing placeholders for zero-trust redaction
    """
    if not isinstance(schema, dict):
        return schema

    def wrap_scalar(prop: dict[str, Any]) -> dict[str, Any]:
        if isinstance(prop.get("anyOf"), list):
            for opt in prop["anyOf"]:
                if (
                    isinstance(opt, dict)
                    and opt.get("type") == "string"
                    and opt.get("pattern") == _PLACEHOLDER_TOKEN_RE.pattern
                ):
                    return prop

            prop = dict(prop)
            prop["anyOf"] = [
                augment(o) for o in prop["anyOf"] if isinstance(o, dict)
            ] + [o for o in prop["anyOf"] if not isinstance(o, dict)]
            return prop

        t = prop.get("type")
        fmt = prop.get("format")
        is_scalar = (t in ("string", "integer", "number", "boolean")) or (
            fmt in ("email", "date-time")
        )
        if not is_scalar:
            return augment(prop)

        original = dict(prop)
        placeholder_opt = {"type": "string", "pattern": _PLACEHOLDER_TOKEN_RE.pattern}
        return {"anyOf": [original, placeholder_opt]}

    def augment(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out = dict(node)
        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            out["properties"] = {
                k: wrap_scalar(v) if isinstance(v, dict) else v
                for k, v in out["properties"].items()
            }
            return out
        if out.get("type") == "array" and isinstance(out.get("items"), dict):
            out["items"] = augment(out["items"])
            return out
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(out.get(key), list):
                out[key] = [augment(x) for x in out[key]]
        return out

    return augment(schema)


def _normalize_json_schema(schema: Any) -> dict[str, Any]:
    """
    Make MCP tool schemas safe for OpenAI function calling.
    - Ensure object schema
    - Ensure properties/required exist
    - Ensure additionalProperties is boolean (default False for strictness)
    """
    if not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    out = dict(schema)
    if out.get("type") != "object":
        # MCP tools should be object inputs; wrap if not.
        out = {
            "type": "object",
            "properties": {"input": out},
            "required": ["input"],
            "additionalProperties": False,
        }

    out.setdefault("properties", {})
    out.setdefault("required", [])
    out.setdefault("additionalProperties", False)
    out = _augment_schema_for_placeholders(out)

    return out


class McpToolCatalog:
    """
    Loads tools
    """

    def __init__(
        self, registry_path: str, *, init: bool = True, timeout: float = 15.0
    ) -> None:
        self.registry_path = registry_path
        self.init = init
        self.timeout = timeout

        self._tools: dict[str, McpTool] = {}

    def refresh(self) -> None:
        # Getting the tools from the registry; the registry has instances of MCPClient
        tools_by_server = registry.fetch_all_tools(
            self.registry_path, init=self.init, timeout=self.timeout
        )
        flat: dict[str, McpTool] = {}

        for server_name, tools in tools_by_server.items():
            for t in tools or []:
                name = t.get("name") or t.get("title")
                if not name or not isinstance(name, str):
                    continue
                desc = t.get("description") or t.get("title") or ""
                schema = t.get("inputSchema") or t.get("parameters") or {}
                params = _normalize_json_schema(schema)

                flat[name] = McpTool(
                    name=name,
                    description=str(desc),
                    parameters=params,
                    server=server_name,
                )

        self._tools = flat

    def get(self, tool_name: str) -> McpTool | None:
        return self._tools.get(tool_name)


def _resolve_placeholders(value: Any, secrets: dict[str, str]) -> Any:
    """
    Replace placeholders with their secret values.
    Supported formats:
      - preferred: $[KEY]
      - tolerated: $KEY or $KEY$ (only replaced if KEY exists)
    Works recursively on dict/list/str.
    """
    if isinstance(value, str):
        # Replace preferred $[KEY]
        def repl_bracket(m: re.Match[str]) -> str:
            key = m.group(1)
            return secrets.get(key, m.group(0))

        s = _PLACEHOLDER_RE.sub(repl_bracket, value)

        # Tolerate $KEY or $KEY$ if KEY exists
        def repl_dollar(m: re.Match[str]) -> str:
            key = m.group(1)
            if key in secrets:
                return secrets[key]
            return m.group(0)

        s = _DOLLAR_KEY_RE.sub(repl_dollar, s)
        return s

    if isinstance(value, list):
        return [_resolve_placeholders(v, secrets) for v in value]

    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, secrets) for k, v in value.items()}

    return value


def _collect_unresolved_placeholders(value: Any, secrets: dict[str, str]) -> list[str]:
    """
    Return placeholder keys referenced that are not resolvable.
    Flags:
      - $[KEY] where KEY not in secrets
      - $KEY / $KEY$ where KEY not in secrets AND looks like an identifier
    """
    unresolved: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, str):
            for m in _PLACEHOLDER_RE.finditer(x):
                key = m.group(1)
                if key not in secrets:
                    unresolved.add(key)
            for m in _DOLLAR_KEY_RE.finditer(x):
                key = m.group(1)
                if key not in secrets:
                    unresolved.add(key)
            return
        if isinstance(x, list):
            for i in x:
                walk(i)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return

    walk(value)
    return sorted(unresolved)


def _iter_strings(value: Any) -> Iterable[str]:
    """Yield all string leaves inside a JSON-like object."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _iter_strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)


def _contains_concat_placeholders(s: str) -> bool:
    """
    Detect obvious placeholder concatenation like "$[VAR_1]$[VAR_2]" or "$VAR_1$$[VAR_2]".
    """
    return bool(re.search(r"(\$\[[A-Za-z0-9_]+\]|\$[A-Za-z][A-Za-z0-9_]*\$?){2,}", s))


class CoordinationService:
    """
    Coordination service for planning
    Redact -> plan tools and steps needed -> call tool loop -> return
    """

    def __init__(
        self,
        *,
        registry_path: str,
        openai_model: str = "gpt-4o-mini",
        step_executor: str | None = None,
        planning_model: str | None = None,
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
        gemini_model: str = "gemini-2.0-flash-exp",
        gemini_api_key: str | None = None,
        system_prompt: str = COORDINATION_PROMPT,
        max_tool_output_len: int = 4000,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai package not installed; cannot create OpenAI client."
            )

        self.registry_path = registry_path
        self.openai_model = openai_model
        self.planning_model = planning_model or openai_model
        self.system_prompt = system_prompt
        self.max_tool_output_len = max_tool_output_len
        self._openai = AsyncOpenAI()
        self._exec_llm_step: ToolArgsLLM | None = None

        # Default executor LLM provider
        self._exec_llm_default: ToolArgsLLM = OpenAIResponsesToolArgsLLM(
            openai_client=self._openai,
            model=self.openai_model,
        )

        # Optional alternate executor
        if step_executor == "ollama":
            self._exec_llm_step = OllamaToolArgsLLM(
                base_url=ollama_url,
                model=ollama_model,
            )
        elif step_executor == "gemini":
            if not gemini_api_key:
                raise ValueError(
                    "Gemini API key required for first_step_executor='gemini'"
                )
            self._exec_llm_step = GeminiToolArgsLLM(
                api_key=gemini_api_key,
                model=gemini_model,
            )

        # Redact for responses in subprompts steps
        self._redactor = LlmOpaqueRedactor(
            self._openai,
            model=self.openai_model,
            prompt_template=REDACTION_PROMPT,
        )
        self._planner = PlanningLLM(self._openai, model=self.planning_model)
        self._catalog = McpToolCatalog(registry_path)
        self._executor = McpToolExecutor(registry_path)
        self._catalog.refresh()

    def _tool_summaries(self) -> list[tuple[str, str]]:
        return [
            (t.name, t.description)
            for t in sorted(self._catalog._tools.values(), key=lambda x: x.name)
        ]

    def _truncate_for_llm(self, redacted_obj: Any) -> str:
        """
        Convert a sanitized (redacted) object into a string payload for the model.
        If too large, wrap into a truncated envelope (still valid JSON).
        """
        s = json.dumps(redacted_obj, ensure_ascii=False)
        if len(s) <= self.max_tool_output_len:
            return s
        preview = s[: self.max_tool_output_len]
        envelope = {"truncated": True, "preview": preview, "total_chars": len(s)}
        return json.dumps(envelope, ensure_ascii=False)

    def _log_step_context(
        self, secrets: dict[str, str], previous_results: Any, step_desc: str
    ) -> None:
        """Helper to log opaque vars and results in consistent format."""
        opaque_vars_log = json.dumps(sorted(secrets.keys()), indent=2)
        prev_results_log = json.dumps(previous_results or {}, indent=2)

        logger.debug(
            f"\n### Available Opaque Variables (Use these as $VAR_NAME) ###\n"
            f"{opaque_vars_log}\n"
            f"(If this list is empty, DO NOT use placeholders like $API_KEY.)\n\n"
            f"### Previous Step Results ###\n"
            f"{prev_results_log}\n\n"
            f"### Task ###\n"
            f"{step_desc}"
        )

    def _record_step_error(
        self,
        trace: list[dict[str, Any]],
        previous_results: dict[str, Any],
        *,
        step_id: str,
        tool_name: str,
        error: Any,
    ) -> None:
        trace.append({"step": step_id, "tool": tool_name, "error": error})
        previous_results[step_id] = error

    def _select_arg_llm(self, idx: int) -> Any:
        if idx == 0 and self._exec_llm_step is not None:
            return self._exec_llm_step
        return self._exec_llm_default

    def _prev_for_step(
        self, step: dict[str, Any], previous_results: dict[str, Any]
    ) -> dict[str, Any]:
        reads = _as_str_list(step.get("reads", []) or step.get("depends_on", []))
        return {sid: previous_results[sid] for sid in reads if sid in previous_results}

    def _visible_for_step(
        self,
        *,
        task: str,
        step: dict[str, Any],
        prev_for_step: dict[str, Any],
        secrets: dict[str, str],
    ) -> tuple[dict[str, str], set[str]]:
        allowed_ph = _as_str_list(step.get("allowed_placeholders", []))

        allowed_key_set = set(allowed_ph)
        for k in _extract_placeholder_keys(task):
            allowed_key_set.add(k)
        for s in _iter_strings(prev_for_step):
            for k in _extract_placeholder_keys(s):
                allowed_key_set.add(k)

        secrets_for_step = {k: v for k, v in secrets.items() if k in allowed_key_set}
        return secrets_for_step, set(secrets_for_step.keys())

    async def _get_tool_args_from_llm(
        self,
        *,
        idx: int,
        task: str,
        tool_def: dict[str, Any],
        tool_name: str,
        visible_for_step: set[str],
        prev_for_step: dict[str, Any],
    ) -> dict[str, Any]:
        arg_llm = self._select_arg_llm(idx)
        return await McpToolExecutor._tool_executor_llm(
            llm=arg_llm,
            tool_def=tool_def,
            tool_name=tool_name,
            step_text=task,
            visible_placeholders=visible_for_step,
            previous_results=prev_for_step,
        )

    async def _validate_and_resolve_args(
        self,
        *,
        parsed_args: dict[str, Any],
        tool_parameters: dict[str, Any],
        secrets_for_step: dict[str, str],
        visible_for_step: set[str],
        secrets_all: dict[str, str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """
        Returns (resolved_args, error_payload). Exactly one is not None.
        """
        canon = _canonicalize_scalar_placeholders(
            parsed_args, tool_parameters, visible_for_step
        )

        unresolved = _collect_unresolved_placeholders(canon, secrets_for_step)
        concat = any(_contains_concat_placeholders(s) for s in _iter_strings(canon))
        if unresolved or concat:
            received = (
                await self._redactor.redact_obj(canon, prior_secrets=secrets_all)
            ).redacted
            err = {
                "error": "Unresolved placeholders or invalid placeholder formatting in tool arguments.",
                "unresolved": unresolved,
                "concat_placeholders": concat,
                "received": received,
            }
            return None, err

        resolved = _resolve_placeholders(canon, secrets_for_step)
        resolved = _coerce_types(resolved, tool_parameters)

        format_errs = _validate_formats(resolved, tool_parameters)
        if format_errs:
            err = {
                "error": "Tool argument format validation failed.",
                "details": format_errs,
                "received": parsed_args,
            }
            return None, err

        return resolved, None

    async def _call_tool(
        self,
        *,
        server: str,
        tool_name: str,
        resolved_args: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                self._executor.call, server, tool_name, resolved_args
            )
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}

    async def _finalize_answer(
        self,
        *,
        redacted_prompt: str,
        final_payload: dict[str, Any],
        secrets: dict[str, str],
        usage_stats: dict[str, int],
    ) -> str:
        subprompt_result = (
            f"User request:\n{redacted_prompt}\n\n"
            f"Step results (redacted JSON):\n{json.dumps(final_payload, ensure_ascii=False, indent=2)}\n\n"
            "Write the final answer to the user. Do not reveal any hidden values behind placeholders."
        )

        resp = await self._openai.responses.create(
            model=self.openai_model,
            instructions=self.system_prompt,
            input=[{"role": "user", "content": subprompt_result}],
        )

        u = getattr(resp, "usage", None)
        if u:
            usage_stats["prompt_tokens"] += (
                getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0) or 0
            )
            usage_stats["completion_tokens"] += (
                getattr(u, "completion_tokens", None)
                or getattr(u, "output_tokens", 0)
                or 0
            )
            usage_stats["total_tokens"] += getattr(u, "total_tokens", 0) or 0

        final_text = (getattr(resp, "output_text", "") or "").strip()
        sf = await self._redactor.redact_text(final_text, secrets)
        return sf.redacted

    async def _run_planned_steps(
        self,
        *,
        redacted_prompt: str,
        planning: dict[str, Any],
        allowed_tools: Iterable[str],
        secrets: dict[str, str],
        usage_stats: dict[str, int],
        llm_call_count: int,
    ) -> tuple[str, dict[str, Any], int]:
        trace: list[dict[str, Any]] = []
        previous_results: dict[str, Any] = {}

        steps = _as_steps_list(planning)
        allowed_set = set(allowed_tools or [])

        for idx, step in enumerate(steps):
            step_id = step.get("id")
            tool_name = step.get("tool") or step.get("tool_hint")
            task = step.get("task")

            if not (
                isinstance(step_id, str)
                and isinstance(tool_name, str)
                and isinstance(task, str)
            ):
                continue

            if not _is_allowed_tool(tool_name, allowed_set):
                self._record_step_error(
                    trace,
                    previous_results,
                    step_id=step_id,
                    tool_name=tool_name,
                    error="Tool not in allowed_tools",
                )
                continue

            tool_info = self._catalog.get(tool_name)
            if not tool_info:
                self._record_step_error(
                    trace,
                    previous_results,
                    step_id=step_id,
                    tool_name=tool_name,
                    error="Tool not found in catalog",
                )
                continue

            tool_def = _build_tool_def(tool_info)

            llm_call_count += 1
            logger.debug(
                f"LLM Call #{llm_call_count}: Execute planned step {step_id} via {tool_name}"
            )

            prev_for_step = self._prev_for_step(step, previous_results)
            secrets_for_step, visible_for_step = self._visible_for_step(
                task=task, step=step, prev_for_step=prev_for_step, secrets=secrets
            )

            parsed_args = await self._get_tool_args_from_llm(
                idx=idx,
                task=task,
                tool_def=tool_def,
                tool_name=tool_name,
                visible_for_step=visible_for_step,
                prev_for_step=prev_for_step,
            )

            resolved, err = await self._validate_and_resolve_args(
                parsed_args=parsed_args,
                tool_parameters=tool_info.parameters,
                secrets_for_step=secrets_for_step,
                visible_for_step=visible_for_step,
                secrets_all=secrets,
            )
            if err is not None:
                self._record_step_error(
                    trace,
                    previous_results,
                    step_id=step_id,
                    tool_name=tool_name,
                    error=err,
                )
                continue

            tool_out = await self._call_tool(
                server=tool_info.server,
                tool_name=tool_name,
                resolved_args=resolved,
            )

            so = await self._redactor.redact_obj(tool_out, secrets)
            secrets.update(so.secrets)

            trace.append(
                {
                    "step": step_id,
                    "tool": tool_name,
                    "server": tool_info.server,
                    "arguments": parsed_args,
                    "result": so.redacted,
                }
            )
            previous_results[step_id] = so.redacted

        llm_call_count += 1
        last_step_id = steps[-1].get("id") if steps else None
        final_payload = previous_results.get(last_step_id, previous_results)

        final_text = await self._finalize_answer(
            redacted_prompt=redacted_prompt,
            final_payload=final_payload,
            secrets=secrets,
            usage_stats=usage_stats,
        )

        return final_text, trace, llm_call_count

    async def runJob(
        self,
        user_prompt: str,
        execute: bool,
        allowed_tools: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        llm_call_count = 0

        s0 = await self._redactor.redact_text(user_prompt)
        redacted_prompt = s0.redacted
        secrets: dict[str, str] = dict(s0.secrets)

        planning: dict[str, Any] | None = None

        # Planning: choose minimal tools
        if allowed_tools is None:
            llm_call_count += 1
            logger.debug(f"LLM Call #{llm_call_count}: Planning")
            planning = await self._planner.plan(
                redacted_prompt, self._tool_summaries()
            )  # Sending redacted prompt to planning
            allowed_tools = planning.get("allowed_tools") or []

        # Stats for OpenAI api key monitoring
        usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        self._log_step_context(secrets, {}, f"User Request: {redacted_prompt}")

        if execute is False:
            return planning

        # Planned subprompts
        if (
            planning is not None
            and isinstance(planning.get("steps"), list)
            and len(planning["steps"]) > 0
            and isinstance(planning["steps"][0], dict)
        ):
            final_text, step_trace, llm_call_count = await self._run_planned_steps(
                redacted_prompt=redacted_prompt,
                planning=planning,
                allowed_tools=allowed_tools or [],
                secrets=secrets,
                usage_stats=usage_stats,
                llm_call_count=llm_call_count,
            )

            return {
                "final_text": final_text,
                "trace": step_trace,
                "secrets_keys": sorted(secrets.keys()),
                "usage": usage_stats,
                "planning": planning,
            }
