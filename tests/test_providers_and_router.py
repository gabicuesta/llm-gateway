import httpx
import pytest

from llm_gateway.config import GatewayConfig
from llm_gateway.providers import BadRequest, build_providers
from llm_gateway.providers.anthropic import translate as to_anthropic
from llm_gateway.providers.gemini import translate as to_gemini
from llm_gateway.router import AllTargetsFailed, CircuitBreaker, Router

from .conftest import Upstream, anthropic_json, gemini_json, make_config, openai_json, sse

REQ = {
    "messages": [
        {"role": "system", "content": "Sé breve."},
        {"role": "user", "content": "Hola"},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}],
        },
        {"role": "assistant", "content": "¿Sí?"},
        {"role": "user", "content": "Adiós"},
    ],
    "temperature": 0.2,
    "max_tokens": 100,
    "stop": "FIN",
}


def test_anthropic_translation():
    body = to_anthropic(REQ, "claude-sonnet-5-5")
    assert body["system"] == "Sé breve."
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]  # consecutive users merged
    assert body["messages"][0]["content"][1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgo=",
    }
    assert body["max_tokens"] == 100 and body["stop_sequences"] == ["FIN"] and body["temperature"] == 0.2


def test_gemini_translation():
    body = to_gemini(REQ)
    assert body["systemInstruction"]["parts"][0]["text"] == "Sé breve."
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/png"
    assert body["generationConfig"] == {"maxOutputTokens": 100, "temperature": 0.2, "stopSequences": ["FIN"]}


def test_tools_rejected_outside_openai():
    with pytest.raises(BadRequest):
        to_anthropic({**REQ, "tools": [{"type": "function"}]}, "m")
    with pytest.raises(BadRequest):
        to_gemini({**REQ, "tools": [{"type": "function"}]})


def router_for(tmp_path, upstream: Upstream, breaker=None) -> Router:
    config: GatewayConfig = make_config(tmp_path)
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle))
    return Router(config, build_providers(config.providers, client), breaker)


async def test_fallback_on_server_error(tmp_path, upstream):
    upstream.on("an.test", 529).on("oa.test", openai_json("desde openai"))
    result, routed = await router_for(tmp_path, upstream).complete("balanced", REQ)
    assert result.text == "desde openai"
    assert str(routed.target) == "oa:gpt-6.1-sol"
    assert routed.attempts[0].target == "an:claude-sonnet-5-5" and "529" in routed.attempts[0].error


async def test_fallback_on_network_error(tmp_path, upstream):
    upstream.on("ge.test", gemini_json("gemini"))  # an.test and oa.test unreachable
    result, routed = await router_for(tmp_path, upstream).complete("balanced", REQ)
    assert result.text == "gemini" and len(routed.attempts) == 2
    assert result.input_tokens == 8


async def test_invalid_request_does_not_fall_back(tmp_path, upstream):
    upstream.on("an.test", 400).on("oa.test", openai_json())
    with pytest.raises(AllTargetsFailed) as exc:
        await router_for(tmp_path, upstream).complete("balanced", REQ)
    assert exc.value.status == 400
    assert not upstream.bodies("oa.test")


async def test_circuit_breaker_skips_failing_target(tmp_path, upstream):
    now = [0.0]
    breaker = CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: now[0])
    upstream.on("an.test", 503).on("oa.test", openai_json())
    router = router_for(tmp_path, upstream, breaker)
    for _ in range(2):
        await router.complete("balanced", REQ)
    assert breaker.state() == {"an:claude-sonnet-5-5": 30.0}

    before = len(upstream.bodies("an.test"))
    await router.complete("balanced", REQ)
    assert len(upstream.bodies("an.test")) == before  # skipped while open

    now[0] = 31
    await router.complete("balanced", REQ)
    assert len(upstream.bodies("an.test")) == before + 1  # retried after cooldown


async def test_streams_are_translated(tmp_path, upstream):
    upstream.on(
        "an.test",
        lambda r: httpx.Response(
            200,
            content=sse(
                [
                    ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
                    ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Ho"}}),
                    ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "la"}}),
                    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
                    ("message_stop", {"type": "message_stop"}),
                ]
            ),
        ),
    )  # fmt: skip
    chunks, routed = await router_for(tmp_path, upstream).stream("balanced", REQ)
    got = [c async for c in chunks]
    assert "".join(c.text for c in got) == "Hola"
    assert got[-1].finish_reason == "stop" and got[-1].input_tokens == 9 and got[-1].output_tokens == 2
    assert upstream.bodies("an.test")[0]["stream"] is True


async def test_stream_falls_back_before_first_token(tmp_path, upstream):
    upstream.on("an.test", 500)
    upstream.on(
        "oa.test",
        lambda r: httpx.Response(
            200,
            content=sse(
                [
                    (None, {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}),
                    (None, {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                    (None, {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}),
                    (None, "[DONE]"),
                ]
            ),
        ),
    )
    chunks, routed = await router_for(tmp_path, upstream).stream("balanced", REQ)
    got = [c async for c in chunks]
    assert str(routed.target) == "oa:gpt-6.1-sol" and "".join(c.text for c in got) == "Hi"
    assert got[-1].output_tokens == 1
    assert upstream.bodies("oa.test")[0]["stream_options"] == {"include_usage": True}


async def test_gemini_stream_uses_sse_and_last_usage(tmp_path, upstream):
    def handler(r: httpx.Request) -> httpx.Response:
        assert r.url.params["alt"] == "sse" and r.url.path.endswith(":streamGenerateContent")
        return httpx.Response(
            200,
            content=sse(
                [
                    (None, {"candidates": [{"content": {"parts": [{"text": "Bue"}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}}),
                    (None, {"candidates": [{"content": {"parts": [{"text": "nas"}]}, "finishReason": "MAX_TOKENS"}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3}}),
                ]
            ),
        )  # fmt: skip

    upstream.on("ge.test", handler)
    chunks, _ = await router_for(tmp_path, upstream).stream("gem", REQ)
    got = [c async for c in chunks]
    assert "".join(c.text for c in got) == "Buenas"
    assert got[-1].finish_reason == "length" and got[-1].output_tokens == 3


async def test_anthropic_complete_parsing(tmp_path, upstream):
    upstream.on("an.test", anthropic_json("ok", 20, 7))
    result, _ = await router_for(tmp_path, upstream).complete("balanced", REQ)
    assert (result.text, result.input_tokens, result.output_tokens) == ("ok", 20, 7)
    sent = upstream.requests[0]
    assert sent.url.path == "/v1/messages" and sent.headers["anthropic-version"]
