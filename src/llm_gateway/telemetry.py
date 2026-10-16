"""Metrics (Prometheus text format, no dependency) and optional OpenTelemetry traces.

Traces follow the OpenTelemetry GenAI semantic conventions (`gen_ai.*` attributes). They are
emitted only if `opentelemetry-sdk` is installed and configured (e.g. with
OTEL_EXPORTER_OTLP_ENDPOINT); otherwise every tracing call is a no-op.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

try:  # pragma: no cover - depends on optional install
    from opentelemetry import trace

    _tracer = trace.get_tracer("llm-gateway")
except ImportError:  # pragma: no cover
    _tracer = None


@contextmanager
def span(name: str, **attributes):
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as s:  # pragma: no cover
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v)
        yield s


def set_attrs(s, **attributes) -> None:
    if s is None:
        return
    for k, v in attributes.items():  # pragma: no cover
        if v is not None:
            s.set_attribute(k, v)


class Metrics:
    def __init__(self) -> None:
        self.requests: dict[tuple[str, str, str], int] = defaultdict(int)  # team, alias, status class
        self.tokens: dict[tuple[str, str], int] = defaultdict(int)  # target, direction
        self.cost: dict[str, float] = defaultdict(float)  # team
        self.fallbacks: dict[str, int] = defaultdict(int)  # alias
        self.cache_hits = 0
        self.pii_redactions = 0
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)

    def observe(
        self, *, team: str, alias: str, status: int, target: str | None, input_tokens: int, output_tokens: int,
        cost: float, latency_s: float, fallbacks: int, cache_hit: bool, pii: int,
    ) -> None:  # fmt: skip
        self.requests[(team, alias, f"{status // 100}xx")] += 1
        if target:
            self.tokens[(target, "input")] += input_tokens
            self.tokens[(target, "output")] += output_tokens
            self.latency_sum[target] += latency_s
            self.latency_count[target] += 1
        self.cost[team] += cost
        self.fallbacks[alias] += fallbacks
        self.cache_hits += int(cache_hit)
        self.pii_redactions += pii

    def render(self, breaker_open: dict[str, float]) -> str:
        lines = [
            "# HELP gateway_requests_total Requests by team, alias and status class.",
            "# TYPE gateway_requests_total counter",
        ]
        lines += [
            f'gateway_requests_total{{team="{t}",alias="{a}",status="{s}"}} {n}'
            for (t, a, s), n in self.requests.items()
        ]
        lines += ["# TYPE gateway_tokens_total counter"]
        lines += [f'gateway_tokens_total{{target="{t}",direction="{d}"}} {n}' for (t, d), n in self.tokens.items()]
        lines += ["# TYPE gateway_cost_usd_total counter"]
        lines += [f'gateway_cost_usd_total{{team="{t}"}} {c:.6f}' for t, c in self.cost.items()]
        lines += ["# TYPE gateway_fallbacks_total counter"]
        lines += [f'gateway_fallbacks_total{{alias="{a}"}} {n}' for a, n in self.fallbacks.items()]
        lines += ["# TYPE gateway_upstream_latency_seconds summary"]
        for t in self.latency_count:
            lines.append(f'gateway_upstream_latency_seconds_sum{{target="{t}"}} {self.latency_sum[t]:.3f}')
            lines.append(f'gateway_upstream_latency_seconds_count{{target="{t}"}} {self.latency_count[t]}')
        lines += [
            "# TYPE gateway_cache_hits_total counter",
            f"gateway_cache_hits_total {self.cache_hits}",
            "# TYPE gateway_pii_redactions_total counter",
            f"gateway_pii_redactions_total {self.pii_redactions}",
            "# TYPE gateway_circuit_open gauge",
        ]
        lines += [f'gateway_circuit_open{{target="{t}"}} 1' for t in breaker_open]
        return "\n".join(lines) + "\n"
