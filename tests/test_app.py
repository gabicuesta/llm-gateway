import json

import httpx
from openai import OpenAI

from .conftest import ADMIN_KEY, DATA_KEY, PRODUCT_KEY, anthropic_json, auth, chat, openai_json, sse

# ------------------------------------------------------------------ auth & policy


def test_requires_a_valid_key(gateway):
    gw = gateway()
    r = gw.post("/v1/chat/completions", json=chat())
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert gw.post("/v1/chat/completions", json=chat(), headers=auth("nope")).status_code == 401


def test_unknown_and_forbidden_models(gateway):
    gw = gateway()
    assert (
        gw.post("/v1/chat/completions", json=chat("gpt-x"), headers=auth()).json()["error"]["code"] == "model_not_found"
    )
    r = gw.post("/v1/chat/completions", json=chat("balanced"), headers=auth(DATA_KEY))
    assert r.status_code == 403 and r.json()["error"]["code"] == "model_not_allowed"


def test_models_endpoint_is_filtered_per_team(gateway):
    gw = gateway()
    assert [m["id"] for m in gw.get("/v1/models", headers=auth(DATA_KEY)).json()["data"]] == ["fast"]
    assert len(gw.get("/v1/models", headers=auth()).json()["data"]) == 3


def test_rate_limit(gateway, upstream):
    upstream.on("oa.test", openai_json())
    gw = gateway(teams=[{"name": "product", "key_hashes": [_h(PRODUCT_KEY)], "requests_per_minute": 2}])
    codes = [gw.post("/v1/chat/completions", json=chat("fast"), headers=auth()).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_budget_is_enforced_with_real_spend(gateway, upstream):
    upstream.on("an.test", anthropic_json("ok", tokens_in=1_000_000, tokens_out=0))  # $2.00 per request
    gw = gateway(teams=[{"name": "product", "key_hashes": [_h(PRODUCT_KEY)], "monthly_budget_usd": 3}])
    first = gw.post("/v1/chat/completions", json=chat(), headers=auth())
    assert first.headers["x-gateway-budget-remaining-usd"] == "3.0000"
    assert first.headers["x-gateway-cost-usd"] == "2.000000"
    assert gw.post("/v1/chat/completions", json=chat(), headers=auth()).status_code == 200  # $2 spent < $3
    blocked = gw.post("/v1/chat/completions", json=chat(), headers=auth())  # $4 spent
    assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "budget_exceeded"


def _h(key):
    from llm_gateway.teams import hash_key

    return hash_key(key)


# ------------------------------------------------------------------ completions


def test_completion_shape_and_headers(gateway, upstream):
    upstream.on("an.test", anthropic_json("Hola, ¿en qué te ayudo?", 12, 6))
    r = gateway().post("/v1/chat/completions", json=chat(), headers=auth())
    body = r.json()
    assert r.status_code == 200
    assert body["object"] == "chat.completion" and body["model"] == "balanced"
    assert body["choices"][0]["message"]["content"] == "Hola, ¿en qué te ayudo?"
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}
    assert r.headers["x-gateway-target"] == "an:claude-sonnet-5-5"
    assert r.headers["x-gateway-fallbacks"] == "0"


def test_fallback_is_visible_in_headers(gateway, upstream):
    upstream.on("an.test", 503).on("oa.test", openai_json("plan B"))
    r = gateway().post("/v1/chat/completions", json=chat(), headers=auth())
    assert r.json()["choices"][0]["message"]["content"] == "plan B"
    assert r.headers["x-gateway-target"] == "oa:gpt-6.1-sol" and r.headers["x-gateway-fallbacks"] == "1"


def test_all_targets_down_returns_502(gateway, upstream):
    r = gateway().post("/v1/chat/completions", json=chat(), headers=auth())
    assert r.status_code == 502 and "All targets failed" in r.json()["error"]["message"]


def test_pii_never_reaches_the_provider_but_comes_back(gateway, upstream):
    def echo(req: httpx.Request) -> httpx.Response:
        sent = json.loads(req.content)["messages"][0]["content"][0]["text"]
        return httpx.Response(200, json=anthropic_json(f"Recibido: {sent}"))

    upstream.on("an.test", echo)
    r = gateway().post("/v1/chat/completions", json=chat(content="Soy Ana, DNI 12345678Z, ana@acme.es"), headers=auth())
    sent = upstream.bodies("an.test")[0]["messages"][0]["content"][0]["text"]
    assert "12345678Z" not in sent and "ana@acme.es" not in sent
    assert "[DNI_1]" in sent and "[EMAIL_1]" in sent
    assert r.json()["choices"][0]["message"]["content"] == "Recibido: Soy Ana, DNI 12345678Z, ana@acme.es"
    assert r.headers["x-gateway-pii-redactions"] == "2"


def test_pii_block_mode(gateway, upstream):
    gw = gateway(pii={"mode": "block"})
    r = gw.post("/v1/chat/completions", json=chat(content="Mi IBAN es ES9121000418450200051332"), headers=auth())
    assert r.status_code == 400 and r.json()["error"]["code"] == "pii_detected"
    assert not upstream.requests


def test_cache_hits_for_deterministic_requests_and_is_per_team(gateway, upstream):
    upstream.on("oa.test", openai_json("cached?"))
    gw = gateway(teams=[
        {"name": "product", "key_hashes": [_h(PRODUCT_KEY)]},
        {"name": "data", "key_hashes": [_h(DATA_KEY)]},
    ])  # fmt: skip
    req = chat("fast", temperature=0)
    assert gw.post("/v1/chat/completions", json=req, headers=auth()).headers["x-gateway-cache"] == "miss"
    hit = gw.post("/v1/chat/completions", json=req, headers=auth())
    assert hit.headers["x-gateway-cache"] == "hit" and hit.json()["choices"][0]["message"]["content"] == "cached?"
    assert len(upstream.requests) == 1
    gw.post("/v1/chat/completions", json=req, headers=auth(DATA_KEY))  # other team: no shared cache
    assert len(upstream.requests) == 2
    gw.post("/v1/chat/completions", json=chat("fast", temperature=0.7), headers=auth())  # not deterministic
    assert len(upstream.requests) == 3


# ------------------------------------------------------------------ streaming


def test_streaming_with_pii_restore_and_usage(gateway, upstream):
    events = [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Te escribo a [EMA"}}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "IL_1] hoy."}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}),
    ]  # fmt: skip
    upstream.on("an.test", lambda r: httpx.Response(200, content=sse(events)))
    gw = gateway()
    req = chat(content="Mi correo es ana@acme.es", stream=True, stream_options={"include_usage": True})
    with gw.stream("POST", "/v1/chat/completions", json=req, headers=auth()) as r:
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(line[6:]) for line in lines[:-1]]
    text = "".join(c["delta"].get("content", "") for p in payloads for c in p["choices"])
    assert text == "Te escribo a ana@acme.es hoy."
    assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["usage"]["completion_tokens"] == 7

    usage = gw.get("/admin/usage", headers=auth(ADMIN_KEY)).json()
    assert usage["teams"][0]["requests"] == 1 and usage["teams"][0]["cost_usd"] > 0


# ------------------------------------------------------------------ admin & observability


def test_admin_endpoints_require_admin_key(gateway):
    gw = gateway()
    for path in ("/admin/usage", "/metrics"):
        assert gw.get(path, headers=auth()).status_code == 401
        assert gw.get(path, headers=auth(ADMIN_KEY)).status_code == 200
    assert "LLM Gateway" in gw.get("/dashboard").text


def test_usage_summary_and_metrics(gateway, upstream):
    upstream.on("an.test", 500).on("oa.test", openai_json(prompt=100, completion=50))
    gw = gateway()
    gw.post("/v1/chat/completions", json=chat(), headers=auth())
    gw.post("/v1/chat/completions", json=chat("fast"), headers=auth(DATA_KEY))

    summary = gw.get("/admin/usage", headers=auth(ADMIN_KEY)).json()
    teams = {t["team"]: t for t in summary["teams"]}
    assert teams["product"]["fallbacks"] == 1 and teams["product"]["monthly_budget_usd"] == 100
    assert {t["target"] for t in summary["targets"]} == {"oa:gpt-6.1-sol", "oa:gpt-6-luna"}
    assert summary["daily"][0]["requests"] == 2

    metrics = gw.get("/metrics", headers=auth(ADMIN_KEY)).text
    assert 'gateway_requests_total{team="product",alias="balanced",status="2xx"} 1' in metrics
    assert 'gateway_fallbacks_total{alias="balanced"} 1' in metrics
    assert 'gateway_tokens_total{target="oa:gpt-6-luna",direction="input"} 100' in metrics


# ------------------------------------------------------------------ drop-in compatibility


def test_official_openai_sdk_works_unchanged(gateway, upstream):
    upstream.on("an.test", anthropic_json("Funciona con el SDK de OpenAI"))
    gw = gateway()
    client = OpenAI(base_url="http://testserver/v1", api_key=PRODUCT_KEY, http_client=gw)
    resp = client.chat.completions.create(model="balanced", messages=[{"role": "user", "content": "Hola"}])
    assert resp.choices[0].message.content == "Funciona con el SDK de OpenAI"
    assert resp.usage.total_tokens == 18
