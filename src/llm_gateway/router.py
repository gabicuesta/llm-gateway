"""Alias → ordered targets, with fallback and a per-target circuit breaker.

Fallback happens on retryable failures (network, timeouts, 408/409/429, 5xx). A target that
keeps failing is skipped for a cooldown period so we don't pay its timeout on every request.
For streams, we can only fall back before the first token has been sent to the client.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .config import GatewayConfig, Target
from .providers import BadRequest, ChatResult, Provider, StreamChunk, UpstreamError


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float, clock=time.monotonic):
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def is_open(self, key: str) -> bool:
        return self._open_until.get(key, 0) > self._clock()

    def success(self, key: str) -> None:
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def failure(self, key: str) -> None:
        self._failures[key] = self._failures.get(key, 0) + 1
        if self._failures[key] >= self._threshold:
            self._open_until[key] = self._clock() + self._cooldown
            self._failures[key] = 0

    def state(self) -> dict[str, float]:
        now = self._clock()
        return {k: round(v - now, 1) for k, v in self._open_until.items() if v > now}


@dataclass
class Attempt:
    target: str
    error: str


@dataclass
class Routed:
    target: Target
    attempts: list[Attempt] = field(default_factory=list)


class AllTargetsFailed(Exception):
    def __init__(self, alias: str, attempts: list[Attempt], status: int = 502):
        self.alias, self.attempts, self.status = alias, attempts, status
        super().__init__(f"All targets failed for '{alias}': " + "; ".join(f"{a.target} → {a.error}" for a in attempts))


class Router:
    def __init__(self, config: GatewayConfig, providers: dict[str, Provider], breaker: CircuitBreaker | None = None):
        self._config = config
        self._providers = providers
        self.breaker = breaker or CircuitBreaker(config.circuit_breaker_failures, config.circuit_breaker_cooldown_s)

    def aliases(self) -> list[str]:
        return list(self._config.models)

    def _plan(self, alias: str) -> list[Target]:
        chain = self._config.models[alias]
        healthy = [t for t in chain if not self.breaker.is_open(str(t))]
        return healthy or chain  # if everything is tripped, try anyway rather than fail without trying

    def _handle(self, alias: str, target: Target, exc: UpstreamError, attempts: list[Attempt]) -> None:
        attempts.append(Attempt(str(target), str(exc)))
        if not exc.retryable:
            # The request itself is wrong (e.g. 400/401): another target won't fix it.
            raise AllTargetsFailed(alias, attempts, status=exc.status or 400) from exc
        self.breaker.failure(str(target))

    async def complete(self, alias: str, request: dict) -> tuple[ChatResult, Routed]:
        attempts: list[Attempt] = []
        bad_request = False
        for target in self._plan(alias):
            try:
                result = await self._providers[target.provider].complete(request, target.model)
            except BadRequest as exc:
                bad_request = True
                attempts.append(Attempt(str(target), str(exc)))
                continue
            except UpstreamError as exc:
                self._handle(alias, target, exc, attempts)
                continue
            self.breaker.success(str(target))
            return result, Routed(target, attempts)
        raise AllTargetsFailed(alias, attempts, status=400 if bad_request and len(attempts) == 1 else 502)

    async def stream(self, alias: str, request: dict) -> tuple[AsyncIterator[StreamChunk], Routed]:
        attempts: list[Attempt] = []
        bad_request = False
        for target in self._plan(alias):
            gen = self._providers[target.provider].stream(request, target.model)
            try:
                first = await anext(gen)
            except StopAsyncIteration:
                first = StreamChunk(finish_reason="stop")
            except BadRequest as exc:
                bad_request = True
                attempts.append(Attempt(str(target), str(exc)))
                continue
            except UpstreamError as exc:
                self._handle(alias, target, exc, attempts)
                continue
            self.breaker.success(str(target))

            async def chained(first=first, gen=gen) -> AsyncIterator[StreamChunk]:
                yield first
                async for chunk in gen:
                    yield chunk

            return chained(), Routed(target, attempts)
        raise AllTargetsFailed(alias, attempts, status=400 if bad_request and len(attempts) == 1 else 502)
