from __future__ import annotations

import json
import logging
from typing import Any

from shardguard.core.tool_args_llm import ToolArgsLLM
from shardguard.mcp_servers import registry

logger = logging.getLogger(__name__)


class McpToolExecutor:
    def __init__(self, registry_path: str, *, timeout: float = 60.0) -> None:
        self.registry_path = registry_path
        self.timeout = timeout

    def call(
        self, server_name: str, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        client = registry.get_or_create_client(self.registry_path, server_name)
        client.initialize(timeout=15.0)
        content = client.tools_call(tool_name, args, timeout=self.timeout)

        text_parts: list[str] = []
        for item in content or []:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)

        return {"content": content, "text": "\n".join(text_parts).strip()}

    @staticmethod
    async def _tool_executor_llm(
        *,
        llm: ToolArgsLLM,
        tool_def: dict[str, Any],
        tool_name: str,
        step_text: str,
        visible_placeholders: set[str],
        previous_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opaque_vars_log = json.dumps(sorted(visible_placeholders), indent=2)
        prev_results_log = json.dumps(previous_results or {}, indent=2)
        logger.debug(
            f"\n### Available Opaque Variables (Use these as $[KEY]) ###\n"
            f"{opaque_vars_log}\n\n"
            f"### Previous Step Results ###\n"
            f"{prev_results_log}\n\n"
            f"### Task ###\n"
            f"Generate the JSON arguments for '{tool_name}'."
        )

        tool_schema = tool_def.get("parameters") or {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        return await llm.generate_tool_args(
            tool_name=tool_name,
            tool_schema=tool_schema,
            tool_description=tool_def.get("description"),
            step_text=step_text,
            visible_placeholders=visible_placeholders,
            previous_results=previous_results,
        )
