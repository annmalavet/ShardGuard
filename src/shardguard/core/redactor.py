from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from shardguard.core.models import OpaqueValues

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^https?://", re.I)
_PHONE_RE = re.compile(r"^\+?\d[\d\-\s\(\)]{7,}\d$")
_PLACEHOLDER_TOKEN = re.compile(r"^\$\[[A-Za-z0-9_]+\]$")
_P_TOKEN = re.compile(r"^\[\[P\d+\]\]$")


def _classify_kind(value: str) -> str:
    v = (value or "").strip()
    if _EMAIL_RE.match(v):
        return "EMAIL"
    if _URL_RE.match(v):
        return "URL"
    if _PHONE_RE.match(v):
        return "PHONE"
    return "SECRET"


def _extract_json_object(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return None


def llm_redaction_to_sanitization_result(
    llm_obj: dict[str, Any],
    *,
    prior_secrets: dict[str, str] | None = None,
) -> OpaqueValues:
    """
    Convert LLM redactor output:
      { "redacted_prompt": "...[[P1]]...", "opaque_values": {"[[P1]]": "secret"} }
    """
    prior_secrets = dict(prior_secrets or {})

    redacted_prompt = llm_obj.get("redacted_prompt")
    opaque_values = llm_obj.get("opaque_values")

    if not isinstance(redacted_prompt, str) or not isinstance(opaque_values, dict):
        return OpaqueValues(redacted=redacted_prompt or "", secrets=prior_secrets)

    counters: dict[str, int] = {}
    for k in prior_secrets.keys():
        m = re.match(r"^([A-Z]+)_(\d+)$", k)
        if m:
            kind = m.group(1)
            idx = int(m.group(2))
            counters[kind] = max(counters.get(kind, 0), idx)

    def new_key(kind: str) -> str:
        n = counters.get(kind, 0) + 1
        counters[kind] = n
        return f"{kind}_{n}"

    def token_sort(t: str) -> tuple[int, str]:
        m = re.match(r"^\[\[P(\d+)\]\]$", t.strip())
        return (int(m.group(1)) if m else 10**9, t)

    secrets_out = dict(prior_secrets)
    token_to_placeholder: dict[str, str] = {}

    for token in sorted(
        [t for t in opaque_values.keys() if isinstance(t, str)],
        key=token_sort,
    ):
        value = opaque_values.get(token)
        if not isinstance(value, str):
            continue
        if _PLACEHOLDER_TOKEN.match(value.strip()) or _P_TOKEN.match(value.strip()):
            continue
        existing_key = next((k for k, v in secrets_out.items() if v == value), None)
        if existing_key is None:
            kind = _classify_kind(value)
            k = new_key(kind)
            secrets_out[k] = value
            existing_key = k

        token_to_placeholder[token] = f"$[{existing_key}]"

    for token, ph in token_to_placeholder.items():
        redacted_prompt = redacted_prompt.replace(token, ph)

    return OpaqueValues(redacted=redacted_prompt, secrets=secrets_out)


class LlmOpaqueRedactor:
    """
    LLM redaction of personal information
    """

    def __init__(
        self, client: AsyncOpenAI, *, model: str, prompt_template: str
    ) -> None:
        self._client = client
        self.model = model
        self.prompt_template = prompt_template

    def _format_prompt(self, user_input: str) -> str:
        return self.prompt_template.replace("{user_prompt}", user_input)

    async def redact_text(
        self, text: str, prior_secrets: dict[str, str] | None = None
    ) -> OpaqueValues:
        prompt = self._format_prompt(text)

        # Only OpenAI used as a provider here for now
        resp = await self._client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=900,
        )

        raw = (getattr(resp, "output_text", "") or "").strip()
        blob = _extract_json_object(raw) or "{}"

        try:
            llm_obj = json.loads(blob)
        except Exception as exc:
            logger.warning("Redactor JSON parse failure: %s | raw=%r", exc, raw[:200])
            llm_obj = {}

        return llm_redaction_to_sanitization_result(
            llm_obj, prior_secrets=prior_secrets
        )

    async def redact_obj(
        self, obj: Any, prior_secrets: dict[str, str] | None = None
    ) -> OpaqueValues:
        secrets = dict(prior_secrets or {})

        async def walk(x: Any) -> Any:
            nonlocal secrets
            if isinstance(x, str):
                res = await self.redact_text(x, prior_secrets=secrets)
                secrets = res.secrets
                return res.redacted
            if isinstance(x, list):
                return [await walk(i) for i in x]
            if isinstance(x, dict):
                out: dict[str, Any] = {}
                for k, v in x.items():
                    out[k] = await walk(v)
                return out
            return x

        redacted = await walk(obj)
        return OpaqueValues(redacted=redacted, secrets=secrets)
