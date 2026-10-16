"""OpenAI Chat Completions, and anything that speaks it (vLLM, Ollama, OpenRouter, Azure-style proxies)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from .base import ChatResult, StreamChunk, UpstreamError, raise_for_status, read_error, sse_events

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Fields we forward. Anything else stays at the gateway (e.g. our own metadata).
FORWARDED = {
    "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "presence_penalty",
    "frequency_penalty", "seed", "response_format", "tools", "tool_choice", "user", "n", "logit_bias",
    "reasoning_effort",
}  # fmt: skip


class OpenAICompatProvider:
    def __init__(self, name: str, client: httpx.AsyncClient, api_key: str | None, base_url: str | None, timeout: float):
        self.name = name
        self._client = client
        self._base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout

    def _body(self, request: dict, model: str, stream: bool) -> dict:
        body = {k: v for k, v in request.items() if k in FORWARDED}
        body["model"] = model
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, request: dict, model: str) -> ChatResult:
        try:
            r = await self._client.post(
                f"{self._base}/chat/completions",
                json=self._body(request, model, False),
                headers=self._headers,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise UpstreamError(self.name, None, exc.__class__.__name__) from exc
        raise_for_status(r, self.name)
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            text=message.get("content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            tool_calls=message.get("tool_calls"),
        )

    async def stream(self, request: dict, model: str) -> AsyncIterator[StreamChunk]:
        try:
            async with self._client.stream(
                "POST",
                f"{self._base}/chat/completions",
                json=self._body(request, model, True),
                headers=self._headers,
                timeout=self._timeout,
            ) as r:
                if r.status_code >= 400:
                    raise_for_status(r, self.name, await read_error(r))
                async for _, data in sse_events(r):
                    for choice in data.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content") or choice.get("finish_reason"):
                            yield StreamChunk(
                                text=delta.get("content") or "", finish_reason=choice.get("finish_reason")
                            )
                    if usage := data.get("usage"):
                        yield StreamChunk(
                            input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0)
                        )
        except httpx.HTTPError as exc:
            raise UpstreamError(self.name, None, exc.__class__.__name__) from exc
