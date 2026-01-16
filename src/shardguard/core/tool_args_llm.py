"""
tool_args_llm.py

Abstraction for the "execution LLM" that produces JSON arguments for one tool call
Executor can be swapped between OpenAI / Gemini / Ollama returning a dict of tool arguments

Executor returns a JSON object matching the tool schema
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from shardguard.core.prompts import TOOL_PROMPT

logger = logging.getLogger(__name__)


# Tools arguments - depending on LLM
@runtime_checkable
class ToolArgsLLM(Protocol):
    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _build_executor_prompt(
    *,
    tool_name: str,
    step_text: str,
    visible_placeholders: set[str],
    previous_results: dict[str, Any] | None,
) -> tuple[str, str]:
    """Returns (system, user) prompts."""

    sys = TOOL_PROMPT

    ph_line = ""
    if visible_placeholders:
        ph_line = "Allowed placeholders:\n" + "\n".join(sorted(visible_placeholders))

    prev_block = ""
    if previous_results:
        prev_block = "Previous step results (redacted):\n" + json.dumps(
            previous_results, indent=2, ensure_ascii=False
        )

    user = "\n\n".join(
        x
        for x in [
            f"Step:\n{step_text}".strip(),
            ph_line.strip(),
            prev_block.strip(),
            f"Return ONLY the JSON arguments object for `{tool_name}`.",
        ]
        if x
    )

    return sys, user


@dataclass
class OpenAIResponsesToolArgsLLM:
    openai_client: Any
    model: str

    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sys, user = _build_executor_prompt(
            tool_name=tool_name,
            step_text=step_text,
            visible_placeholders=visible_placeholders,
            previous_results=previous_results,
        )

        tool_def = {
            "type": "function",
            "name": tool_name,
            "description": f"Tool `{tool_name}`",
            "parameters": tool_schema,
            "strict": True,
        }

        # uses OpenAI Responses API
        resp = await self.openai_client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            tools=[tool_def],
            tool_choice={"type": "function", "name": tool_name},
            parallel_tool_calls=False,
        )

        fc = next(
            (
                it
                for it in (getattr(resp, "output", None) or [])
                if getattr(it, "type", None) == "function_call"
            ),
            None,
        )
        if not fc:
            raise RuntimeError(
                f"Executor did not return a function_call for {tool_name}"
            )

        logger.debug(
            "Executor(OpenAI) function_call arguments: %s",
            getattr(fc, "arguments", None),
        )
        return json.loads(getattr(fc, "arguments", "") or "{}")


@dataclass
class GeminiToolArgsLLM:
    """
    Uses Google GenAI SDK (google-genai) and JSON schema output.
    Requires google-genai

    Uses:
      response_mime_type='application/json'
      response_json_schema=<tool_schema>
    so the model returns JSON
    """

    gemini_async_client: Any
    model: str

    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sys, user = _build_executor_prompt(
            tool_name=tool_name,
            step_text=step_text,
            visible_placeholders=visible_placeholders,
            previous_results=previous_results,
        )
        prompt = sys + "\n\n" + user

        # response_json_schema (JSON schema) to constrain output
        resp = await self.gemini_async_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": tool_schema,
                "temperature": 0,
            },
        )

        # google-genai provides resp.parsed when JSON schema is used
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, dict):
            return parsed

        # Fallback: try parsing
        txt = getattr(resp, "text", "") or ""
        return json.loads(txt)


@dataclass
class OllamaToolArgsLLM:
    """
    Uses Ollama's /api/chat to ask for a JSON object.
    """

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout_s: float = 60.0

    async def generate_tool_args(
        self,
        *,
        tool_name: str,
        tool_schema: dict[str, Any],
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sys, user = _build_executor_prompt(
            tool_name=tool_name,
            step_text=step_text,
            visible_placeholders=visible_placeholders,
            previous_results=previous_results,
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            "format": tool_schema,
            "options": {"temperature": 0},
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()

        msg = (
            (data.get("message") or {}).get("content")
            if isinstance(data, dict)
            else None
        )
        if not isinstance(msg, str):
            raise RuntimeError(f"Ollama response missing message.content: {data}")

        return json.loads(msg)
