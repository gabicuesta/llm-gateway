"""Test harness: fake upstream providers behind httpx.MockTransport, and a TestClient for the gateway."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.config import GatewayConfig
from llm_gateway.teams import hash_key

PRODUCT_KEY = "gw-product-key"
DATA_KEY = "gw-data-key"
ADMIN_KEY = "gw-admin-key"


def openai_json(text="Hola", prompt=10, completion=5):
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def anthropic_json(text="Hola", tokens_in=12, tokens_out=6):
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }


def gemini_json(text="Hola"):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4},
    }


def sse(events: list[tuple[str | None, dict | str]]) -> bytes:
    out = []
    for event, data in events:
        if event:
            out.append(f"event: {event}")
        out.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
        out.append("")
    return ("\n".join(out) + "\n").encode()


class Upstream:
    """Routes requests by host to per-provider handlers; records what each provider received."""

    def __init__(self) -> None:
        self.handlers: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.requests: list[httpx.Request] = []

    def on(self, host: str, handler) -> Upstream:
        if isinstance(handler, dict):
            payload = handler
            handler = lambda r: httpx.Response(200, json=payload)  # noqa: E731
        elif isinstance(handler, int):
            status = handler
            handler = lambda r: httpx.Response(status, json={"error": "boom"})  # noqa: E731
        self.handlers[host] = handler
        return self

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.handlers.get(request.url.host)
        if handler is None:
            raise httpx.ConnectError("no route", request=request)
        return handler(request)

    def bodies(self, host: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.host == host]


def make_config(tmp_path, **overrides) -> GatewayConfig:
    base = {
        "providers": {
            "oa": {"type": "openai", "base_url": "http://oa.test/v1"},
            "an": {"type": "anthropic", "base_url": "http://an.test/v1"},
            "ge": {"type": "gemini", "base_url": "http://ge.test/v1beta"},
        },
        "models": {
            "balanced": ["an:claude-sonnet-5-5", "oa:gpt-6.1-sol", "ge:gemini-3.8-flash"],
            "fast": ["oa:gpt-6-luna"],
            "gem": ["ge:gemini-3.8-flash"],
        },
        "pricing": {"claude-sonnet-5-5": {"input": 2.0, "output": 10.0}, "gpt-6-luna": {"input": 0.1, "output": 0.5}},
        "teams": [
            {"name": "product", "key_hashes": [hash_key(PRODUCT_KEY)], "monthly_budget_usd": 100},
            {"name": "data", "key_hashes": [hash_key(DATA_KEY)], "allowed_models": ["fast"]},
        ],
        "admin_key_hash": hash_key(ADMIN_KEY),
        "database": str(tmp_path / "usage.db"),
    }
    base.update(overrides)
    return GatewayConfig.model_validate(base)


@pytest.fixture
def upstream() -> Upstream:
    return Upstream()


@pytest.fixture
def gateway(tmp_path, upstream):
    """Factory: gateway(**config_overrides) -> TestClient."""
    clients = []

    def factory(**overrides) -> TestClient:
        config = make_config(tmp_path, **overrides)
        http = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle))
        client = TestClient(create_app(config, http_client=http))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


def auth(key: str = PRODUCT_KEY) -> dict:
    return {"Authorization": f"Bearer {key}"}


def chat(model="balanced", content="Hola", **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}
