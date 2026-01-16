import json
from typing import Any

from openai import AsyncOpenAI  # type: ignore

from shardguard.core.prompts import PLANNING_PROMPT_FULL

from .llm_providers import LLMProviderFactory


class PlanningLLM:
    """
    Least-privilege planner:
    - No access to tools
    - No access to the opaque-store keys list (only sees the redacted prompt)
    - Outputs a tool list
    """

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini") -> None:
        self._client = client
        provider_type: str = "ollama"
        api_key: str | None = None
        model: str = "gpt-4o-mini"
        base_url: str = "http://localhost:11434"
        self.model = model
        provider_kwargs = {}
        if provider_type.lower() == "ollama":
            provider_kwargs["base_url"] = base_url
        elif provider_type.lower() == "gemini":
            provider_kwargs["api_key"] = api_key

        self.llm_provider = LLMProviderFactory.create_provider(
            provider_type=provider_type, model=model, **provider_kwargs
        )

    @staticmethod
    def _extract_json_object(s: str) -> str | None:
        s = s.strip()
        if not s:
            return None
        if s.startswith("{") and s.endswith("}"):
            return s
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]
        return None

    async def plan(
        self, redacted_prompt: str, tool_summaries: list[tuple[str, str]]
    ) -> dict[str, Any]:
        tools_txt = "\n".join(f"- {name}: {desc}" for name, desc in tool_summaries)

        instructions = PLANNING_PROMPT_FULL
        # This is using OpenAI *only* for planning - can be updated to use other LLM provider
        resp = await self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": f"User request:\n{redacted_prompt}\n\nAvailable tools:\n{tools_txt}",
                }
            ],
        )

        txt = (getattr(resp, "output_text", "") or "").strip()
        blob = self._extract_json_object(txt) or "{}"

        try:
            obj = json.loads(blob)
        except Exception:
            obj = {}

        allowed = obj.get("allowed_tools")
        steps = obj.get("steps")

        if not isinstance(allowed, list) or not all(
            isinstance(x, str) for x in allowed
        ):
            allowed = []

        steps2: list[dict[str, Any]] = []
        if isinstance(steps, list):
            for s in steps:
                if not isinstance(s, dict):
                    continue
                sid = s.get("id")
                task = s.get("task")
                tool_hint = s.get("tool_hint")

                depends_on = s.get("depends_on", [])

                if not isinstance(sid, str) or not isinstance(task, str):
                    continue
                if tool_hint is not None and not isinstance(tool_hint, str):
                    tool_hint = None
                if not isinstance(depends_on, list) or not all(
                    isinstance(x, str) for x in depends_on
                ):
                    depends_on = []

                steps2.append(
                    {
                        "id": sid,
                        "task": task,
                        "tool_hint": tool_hint,
                        "depends_on": depends_on,
                    }
                )
        else:
            steps2 = []

        # de-dupe
        seen: set[str] = set()
        allowed2: list[str] = []
        for name in allowed:
            if name not in seen:
                seen.add(name)
                allowed2.append(name)

        return {"allowed_tools": allowed2, "steps": steps2}
