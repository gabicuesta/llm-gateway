"""OpenAI-compatible HTTP surface: /v1/chat/completions, /v1/models, plus admin and metrics."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .cache import ResponseCache, cache_key, cacheable
from .config import GatewayConfig, Target
from .pii import Redactor, StreamRestorer, Vault, restore
from .providers import ChatResult, build_providers
from .router import AllTargetsFailed, Router
from .store import UsageRecord, UsageStore
from .teams import KeyRing, RateLimiter
from .telemetry import Metrics, set_attrs, span

STATIC = Path(__file__).parent / "static"


def error(status: int, message: str, code: str, headers: dict | None = None) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "api_error"
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": code}}, status_code=status, headers=headers
    )


def bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key")


class Gateway:
    def __init__(self, config: GatewayConfig, client: httpx.AsyncClient):
        self.config = config
        self.router = Router(config, build_providers(config.providers, client))
        self.keys = KeyRing(config)
        self.limiter = RateLimiter()
        self.store = UsageStore(config.database)
        self.cache = ResponseCache(config.cache.ttl_s, config.cache.max_entries)
        self.redactor = Redactor(config.pii.entities)
        self.metrics = Metrics()

    def cost(self, target: Target | None, tokens_in: int, tokens_out: int) -> float:
        price = self.config.price_for(target.model) if target else None
        if not price:
            return 0.0
        return round((tokens_in * price.input + tokens_out * price.output) / 1_000_000, 8)

    async def account(self, rec: UsageRecord, latency_s: float) -> None:
        await self.store.record(rec)
        self.metrics.observe(
            team=rec.team, alias=rec.alias, status=rec.status, target=rec.target, input_tokens=rec.input_tokens,
            output_tokens=rec.output_tokens, cost=rec.cost_usd, latency_s=latency_s, fallbacks=rec.fallbacks,
            cache_hit=rec.cache_hit, pii=rec.pii_redactions,
        )  # fmt: skip


def completion_body(result: ChatResult, alias: str, request_id: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.text}
    if result.tool_calls:
        message["tool_calls"] = result.tool_calls
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": alias,
        "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason}],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
    }


def create_app(config: GatewayConfig, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        app.state.gw = Gateway(config, client)
        yield
        app.state.gw.store.close()
        if http_client is None:
            await client.aclose()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------ public

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "open_circuits": app.state.gw.router.breaker.state()}

    @app.get("/v1/models")
    async def models(request: Request):
        gw: Gateway = app.state.gw
        team = gw.keys.team_for(bearer(request))
        if not team:
            return error(401, "Invalid or missing API key.", "invalid_api_key")
        allowed = team.allowed_models or gw.router.aliases()
        return {"object": "list", "data": [{"id": a, "object": "model", "owned_by": "gateway"} for a in allowed]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        gw: Gateway = app.state.gw
        started = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        team = gw.keys.team_for(bearer(request))
        if not team:
            return error(401, "Invalid or missing API key.", "invalid_api_key")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return error(400, "Body must be JSON.", "invalid_json")
        alias = body.get("model")
        if alias not in gw.config.models:
            return error(
                404, f"Unknown model '{alias}'. Available: {', '.join(gw.router.aliases())}.", "model_not_found"
            )
        if team.allowed_models is not None and alias not in team.allowed_models:
            return error(403, f"Team '{team.name}' may not use '{alias}'.", "model_not_allowed")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            return error(400, "`messages` must be a non-empty list.", "invalid_messages")

        allowed, wait = gw.limiter.allow(team)
        if not allowed:
            return error(429, "Rate limit exceeded for your team.", "rate_limit_exceeded", {"Retry-After": str(wait)})

        spent = gw.store.month_spend(team.name)
        budget = team.monthly_budget_usd
        if budget is not None and spent >= budget:
            return error(429, f"Monthly budget of ${budget:.2f} reached for team '{team.name}'.", "budget_exceeded")

        vault = Vault()
        if gw.config.pii.enabled:
            body, vault = gw.redactor.redact_request(body)
            if vault.total and gw.config.pii.mode == "block":
                kinds = ", ".join(sorted(vault.counts))
                return error(400, f"Request blocked: it contains personal data ({kinds}).", "pii_detected")
        restore_pii = gw.config.pii.restore_in_response

        headers = {"x-gateway-request-id": request_id, "x-gateway-pii-redactions": str(vault.total)}
        if budget is not None:
            headers["x-gateway-budget-remaining-usd"] = f"{max(budget - spent, 0):.4f}"

        rec = UsageRecord(team=team.name, alias=alias, target=None, status=200, pii_redactions=vault.total)

        # ---------------------------------------------------- streaming
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            try:
                chunks, routed = await gw.router.stream(alias, body)
            except AllTargetsFailed as exc:
                rec.status, rec.stream = exc.status, True
                await gw.account(rec, time.perf_counter() - started)
                return error(exc.status, str(exc), "upstream_error")

            rec.target, rec.fallbacks, rec.stream = str(routed.target), len(routed.attempts), True
            headers |= {"x-gateway-target": rec.target, "x-gateway-fallbacks": str(rec.fallbacks)}

            async def sse():
                restorer = StreamRestorer(vault if restore_pii else Vault())
                created = int(time.time())

                def frame(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
                    payload: dict[str, Any] = {
                        "id": request_id, "object": "chat.completion.chunk", "created": created, "model": alias,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if usage is None else [],
                    }  # fmt: skip
                    if usage is not None:
                        payload["usage"] = usage
                    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                finish = "stop"
                yield frame({"role": "assistant", "content": ""})
                try:
                    with span("gen_ai.chat", **{"gen_ai.request.model": alias, "gen_ai.system": rec.target}) as s:
                        async for chunk in chunks:
                            if chunk.text and (text := restorer.feed(chunk.text)):
                                yield frame({"content": text})
                            if chunk.finish_reason:
                                finish = chunk.finish_reason
                            if chunk.input_tokens is not None:
                                rec.input_tokens = chunk.input_tokens
                            if chunk.output_tokens is not None:
                                rec.output_tokens = chunk.output_tokens
                        if tail := restorer.flush():
                            yield frame({"content": tail})
                        set_attrs(s, **{"gen_ai.usage.input_tokens": rec.input_tokens,
                                        "gen_ai.usage.output_tokens": rec.output_tokens})  # fmt: skip
                    yield frame({}, finish)
                    if include_usage:
                        yield frame({}, usage={
                            "prompt_tokens": rec.input_tokens, "completion_tokens": rec.output_tokens,
                            "total_tokens": rec.input_tokens + rec.output_tokens,
                        })  # fmt: skip
                    yield "data: [DONE]\n\n"
                except Exception as exc:  # mid-stream failure: tell the client, keep accounting
                    rec.status = 502
                    err = {"error": {"message": str(exc), "type": "api_error", "code": "upstream_error"}}
                    yield f"data: {json.dumps(err)}\n\n"
                finally:
                    rec.cost_usd = gw.cost(routed.target, rec.input_tokens, rec.output_tokens)
                    rec.latency_ms = int((time.perf_counter() - started) * 1000)
                    await gw.account(rec, time.perf_counter() - started)

            return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)

        # ------------------------------------------------ non-streaming
        key = None
        if gw.config.cache.enabled and cacheable(body, request.headers.get("x-gateway-cache")):
            key = cache_key(team.name, alias, body)
            if hit := gw.cache.get(key):
                result, target = hit
                rec.target, rec.cache_hit = target, True
                rec.latency_ms = int((time.perf_counter() - started) * 1000)
                await gw.account(rec, time.perf_counter() - started)
                out = completion_body(result, alias, request_id)
                if restore_pii:
                    out["choices"][0]["message"]["content"] = restore(result.text, vault)
                headers |= {"x-gateway-target": target, "x-gateway-cache": "hit", "x-gateway-cost-usd": "0"}
                return JSONResponse(out, headers=headers)

        with span("gen_ai.chat", **{"gen_ai.request.model": alias, "gen_ai.operation.name": "chat"}) as s:
            try:
                result, routed = await gw.router.complete(alias, body)
            except AllTargetsFailed as exc:
                rec.status, rec.fallbacks = exc.status, len(exc.attempts)
                rec.latency_ms = int((time.perf_counter() - started) * 1000)
                await gw.account(rec, time.perf_counter() - started)
                return error(exc.status, str(exc), "upstream_error")
            set_attrs(
                s, **{"gen_ai.response.model": routed.target.model, "gen_ai.usage.input_tokens": result.input_tokens,
                      "gen_ai.usage.output_tokens": result.output_tokens},
            )  # fmt: skip

        if key:
            gw.cache.put(key, result, str(routed.target))

        rec.target, rec.fallbacks = str(routed.target), len(routed.attempts)
        rec.input_tokens, rec.output_tokens = result.input_tokens, result.output_tokens
        rec.cost_usd = gw.cost(routed.target, result.input_tokens, result.output_tokens)
        rec.latency_ms = int((time.perf_counter() - started) * 1000)
        await gw.account(rec, time.perf_counter() - started)

        out = completion_body(result, alias, request_id)
        if restore_pii:
            out["choices"][0]["message"]["content"] = restore(result.text, vault)
        headers |= {
            "x-gateway-target": rec.target,
            "x-gateway-fallbacks": str(rec.fallbacks),
            "x-gateway-cache": "miss" if key else "skip",
            "x-gateway-cost-usd": f"{rec.cost_usd:.6f}",
        }
        return JSONResponse(out, headers=headers)

    # ------------------------------------------------------------- admin

    @app.get("/admin/usage")
    async def usage(request: Request, month: str | None = None):
        gw: Gateway = app.state.gw
        if not gw.keys.is_admin(bearer(request)):
            return error(401, "Admin key required.", "invalid_api_key")
        summary = gw.store.summary(month)
        budgets = {t.name: t.monthly_budget_usd for t in gw.config.teams}
        for row in summary["teams"]:
            row["monthly_budget_usd"] = budgets.get(row["team"])
        summary["open_circuits"] = gw.router.breaker.state()
        summary["cache"] = {"hits": gw.cache.hits, "misses": gw.cache.misses}
        return summary

    @app.get("/metrics")
    async def metrics(request: Request):
        gw: Gateway = app.state.gw
        if not gw.keys.is_admin(bearer(request)):
            return error(401, "Admin key required.", "invalid_api_key")
        return PlainTextResponse(gw.metrics.render(gw.router.breaker.state()))

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(STATIC / "dashboard.html")

    return app
