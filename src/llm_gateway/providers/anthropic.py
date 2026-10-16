"""Anthropic Messages API adapter (text and images; tools are not translated in v1)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from .base import (
    BadRequest,
    ChatResult,
    StreamChunk,
    UpstreamError,
    max_tokens_of,
    parse_data_url,
    raise_for_status,
    read_error,
    sse_events,
    stop_list,
    text_of,
)

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def _parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for part in content or []:
        if part.get("type") == "text":
            out.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if parsed := parse_data_url(url):
                out.append({"type": "image", "source": {"type": "base64", "media_type": parsed[0], "data": parsed[1]}})
            else:
                out.append({"type": "image", "source": {"type": "url", "url": url}})
    return out


def translate(request: dict, model: str) -> dict:
    if request.get("tools"):
        raise BadRequest("Tool calling is only supported on OpenAI-compatible targets in this version.")
    system, messages = [], []
    for m in request.get("messages", []):
        role = m.get("role")
        if role in ("system", "developer"):
            system.append(text_of(m.get("content")))
            continue
        role = "assistant" if role == "assistant" else "user"
        parts = _parts(m.get("content"))
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(parts)  # Anthropic requires alternating roles
        else:
            messages.append({"role": role, "content": parts})

    body: dict = {"model": model, "max_tokens": max_tokens_of(request), "messages": messages}
    if system:
        body["system"] = "\n\n".join(system)
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if request.get(src) is not None:
            body[dst] = request[src]
    if stops := stop_list(request):
        body["stop_sequences"] = stops
    return body


class AnthropicProvider:
    def __init__(self, name: str, client: httpx.AsyncClient, api_key: str | None, base_url: str | None, timeout: float):
        self.name = name
        self._client = client
        self._base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"x-api-key": api_key or "", "anthropic-version": API_VERSION}
        self._timeout = timeout

    async def complete(self, request: dict, model: str) -> ChatResult:
        try:
            r = await self._client.post(
                f"{self._base}/messages", json=translate(request, model), headers=self._headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            raise UpstreamError(self.name, None, exc.__class__.__name__) from exc
        raise_for_status(r, self.name)
        data = r.json()
        usage = data.get("usage") or {}
        return ChatResult(
            text="".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"),
            finish_reason=FINISH.get(data.get("stop_reason"), "stop"),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    async def stream(self, request: dict, model: str) -> AsyncIterator[StreamChunk]:
        body = {**translate(request, model), "stream": True}
        input_tokens = 0
        try:
            async with self._client.stream(
                "POST", f"{self._base}/messages", json=body, headers=self._headers, timeout=self._timeout
            ) as r:
                if r.status_code >= 400:
                    raise_for_status(r, self.name, await read_error(r))
                async for event, data in sse_events(r):
                    kind = event or data.get("type")
                    if kind == "message_start":
                        input_tokens = ((data.get("message") or {}).get("usage") or {}).get("input_tokens", 0)
                    elif kind == "content_block_delta":
                        delta = data.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            yield StreamChunk(text=delta.get("text", ""))
                    elif kind == "message_delta":
                        stop = (data.get("delta") or {}).get("stop_reason")
                        yield StreamChunk(
                            finish_reason=FINISH.get(stop, "stop"),
                            input_tokens=input_tokens,
                            output_tokens=(data.get("usage") or {}).get("output_tokens", 0),
                        )
                    elif kind == "error":
                        raise UpstreamError(self.name, None, str((data.get("error") or {}).get("message")))
        except httpx.HTTPError as exc:
            raise UpstreamError(self.name, None, exc.__class__.__name__) from exc
