"""Google Gemini generateContent adapter (text and inline images; tools not translated in v1)."""

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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter"}


def _parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    out = []
    for part in content or []:
        if part.get("type") == "text":
            out.append({"text": part.get("text", "")})
        elif part.get("type") == "image_url":
            parsed = parse_data_url((part.get("image_url") or {}).get("url", ""))
            if not parsed:
                raise BadRequest("Gemini targets accept images only as base64 data URLs.")
            out.append({"inline_data": {"mime_type": parsed[0], "data": parsed[1]}})
    return out


def translate(request: dict) -> dict:
    if request.get("tools"):
        raise BadRequest("Tool calling is only supported on OpenAI-compatible targets in this version.")
    system, contents = [], []
    for m in request.get("messages", []):
        role = m.get("role")
        if role in ("system", "developer"):
            system.append(text_of(m.get("content")))
            continue
        role = "model" if role == "assistant" else "user"
        parts = _parts(m.get("content"))
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    config: dict = {"maxOutputTokens": max_tokens_of(request)}
    if request.get("temperature") is not None:
        config["temperature"] = request["temperature"]
    if request.get("top_p") is not None:
        config["topP"] = request["top_p"]
    if stops := stop_list(request):
        config["stopSequences"] = stops
    if (request.get("response_format") or {}).get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"

    body: dict = {"contents": contents, "generationConfig": config}
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
    return body


def _usage(data: dict) -> tuple[int, int]:
    u = data.get("usageMetadata") or {}
    return u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


class GeminiProvider:
    def __init__(self, name: str, client: httpx.AsyncClient, api_key: str | None, base_url: str | None, timeout: float):
        self.name = name
        self._client = client
        self._base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"x-goog-api-key": api_key or ""}
        self._timeout = timeout

    async def complete(self, request: dict, model: str) -> ChatResult:
        try:
            r = await self._client.post(
                f"{self._base}/models/{model}:generateContent",
                json=translate(request),
                headers=self._headers,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise UpstreamError(self.name, None, exc.__class__.__name__) from exc
        raise_for_status(r, self.name)
        data = r.json()
        candidate = (data.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []))
        tokens_in, tokens_out = _usage(data)
        return ChatResult(
            text=text,
            finish_reason=FINISH.get(candidate.get("finishReason"), "stop"),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )

    async def stream(self, request: dict, model: str) -> AsyncIterator[StreamChunk]:
        url = f"{self._base}/models/{model}:streamGenerateContent"
        tokens_in = tokens_out = 0
        finish = None
        try:
            async with self._client.stream(
                "POST",
                url,
                params={"alt": "sse"},
                json=translate(request),
                headers=self._headers,
                timeout=self._timeout,
            ) as r:
                if r.status_code >= 400:
                    raise_for_status(r, self.name, await read_error(r))
                async for _, data in sse_events(r):
                    candidate = (data.get("candidates") or [{}])[0]
                    text = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []))
                    if text:
                        yield StreamChunk(text=text)
                    if candidate.get("finishReason"):
                        finish = FINISH.get(candidate["finishReason"], "stop")
                    if data.get("usageMetadata"):
                        tokens_in, tokens_out = _usage(data)  # cumulative, keep the last
        except httpx.HTTPError as exc:
            raise UpstreamError(self.name, None, exc.__class__.__name__) from exc
        yield StreamChunk(finish_reason=finish or "stop", input_tokens=tokens_in, output_tokens=tokens_out)
