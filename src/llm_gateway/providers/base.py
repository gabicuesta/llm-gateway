"""Provider contract. The lingua franca is the OpenAI Chat Completions request format:
clients speak it, and each adapter translates it to its vendor and back."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ChatResult:
    text: str
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[dict] | None = None


@dataclass
class StreamChunk:
    """Either a text delta or, at the end, the finish reason and usage."""

    text: str = ""
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class UpstreamError(Exception):
    provider: str
    status: int | None
    message: str
    retryable: bool = field(default=True)

    def __str__(self) -> str:
        code = f"HTTP {self.status}" if self.status else "network error"
        return f"{self.provider}: {code}: {self.message}"


class Provider(Protocol):
    name: str

    async def complete(self, request: dict, model: str) -> ChatResult: ...

    def stream(self, request: dict, model: str) -> AsyncIterator[StreamChunk]: ...


class BadRequest(ValueError):
    """The client request can't be served by this provider (not retryable on another target)."""


# ----------------------------------------------------------------- helpers


def raise_for_status(response: httpx.Response, provider: str, body: str | None = None) -> None:
    if response.status_code < 400:
        return
    text = (body if body is not None else response.text)[:300]
    # 400/413/422 mean the request itself is invalid: another target won't help.
    # Everything else (auth/config problems on one provider, 404 model, 429, 5xx) is worth a fallback.
    retryable = response.status_code not in (400, 413, 422)
    raise UpstreamError(provider, response.status_code, text, retryable)


async def read_error(response: httpx.Response) -> str:
    try:
        return (await response.aread()).decode("utf-8", "replace")
    except httpx.HTTPError:
        return ""


async def sse_events(response: httpx.Response) -> AsyncIterator[tuple[str | None, Any]]:
    """Yield (event, parsed data) from a Server-Sent Events response."""
    event: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                data = "\n".join(data_lines)
                if data != "[DONE]":
                    yield event, json.loads(data)
            event, data_lines = None, []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines and (data := "\n".join(data_lines)) != "[DONE]":
        yield event, json.loads(data)


def text_of(content: Any) -> str:
    """Flatten OpenAI message content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


_DATA_URL = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)


def parse_data_url(url: str) -> tuple[str, str] | None:
    m = _DATA_URL.match(url)
    if not m:
        return None
    base64.b64decode(m.group("data")[:64] + "==", validate=False)  # cheap sanity check
    return m.group("mime"), m.group("data")


def max_tokens_of(request: dict, default: int = 4096) -> int:
    return int(request.get("max_completion_tokens") or request.get("max_tokens") or default)


def stop_list(request: dict) -> list[str] | None:
    stop = request.get("stop")
    if stop is None:
        return None
    return [stop] if isinstance(stop, str) else list(stop)
