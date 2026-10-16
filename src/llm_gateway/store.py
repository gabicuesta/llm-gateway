"""Usage ledger in SQLite: every request, its cost and outcome. Source of truth for budgets."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    month TEXT NOT NULL,
    team TEXT NOT NULL,
    alias TEXT NOT NULL,
    target TEXT,
    status INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    fallbacks INTEGER NOT NULL DEFAULT 0,
    pii_redactions INTEGER NOT NULL DEFAULT 0,
    stream INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_team_month ON usage (team, month);
"""


@dataclass
class UsageRecord:
    team: str
    alias: str
    target: str | None
    status: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cache_hit: bool = False
    fallbacks: int = 0
    pii_redactions: int = 0
    stream: bool = False


def current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


class UsageStore:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._spend: dict[tuple[str, str], float] = {}
        for team, month, total in self._conn.execute(
            "SELECT team, month, SUM(cost_usd) FROM usage WHERE month = ? GROUP BY team, month", (current_month(),)
        ):
            self._spend[(team, month)] = total or 0.0

    def month_spend(self, team: str) -> float:
        return self._spend.get((team, current_month()), 0.0)

    def _insert(self, rec: UsageRecord) -> None:
        now = datetime.now(UTC)
        row = asdict(rec) | {"ts": now.isoformat(timespec="seconds"), "month": now.strftime("%Y-%m")}
        cols = ", ".join(row)
        with self._lock:
            self._conn.execute(f"INSERT INTO usage ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
            self._conn.commit()

    async def record(self, rec: UsageRecord) -> None:
        key = (rec.team, current_month())
        self._spend[key] = self._spend.get(key, 0.0) + rec.cost_usd  # update in memory first: budgets act now
        await asyncio.to_thread(self._insert, rec)

    def summary(self, month: str | None = None) -> dict:
        month = month or current_month()
        with self._lock:
            teams = self._conn.execute(
                """SELECT team, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd),
                          AVG(latency_ms), SUM(cache_hit), SUM(fallbacks > 0), SUM(status >= 400), SUM(pii_redactions)
                   FROM usage WHERE month = ? GROUP BY team ORDER BY SUM(cost_usd) DESC""",
                (month,),
            ).fetchall()
            models = self._conn.execute(
                """SELECT target, COUNT(*), SUM(cost_usd), AVG(latency_ms), SUM(status >= 400)
                   FROM usage WHERE month = ? AND target IS NOT NULL GROUP BY target ORDER BY COUNT(*) DESC""",
                (month,),
            ).fetchall()
            daily = self._conn.execute(
                "SELECT substr(ts, 1, 10), SUM(cost_usd), COUNT(*) FROM usage WHERE month = ? GROUP BY 1 ORDER BY 1",
                (month,),
            ).fetchall()
        return {
            "month": month,
            "teams": [
                {
                    "team": t[0], "requests": t[1], "input_tokens": t[2] or 0, "output_tokens": t[3] or 0,
                    "cost_usd": round(t[4] or 0, 6), "avg_latency_ms": round(t[5] or 0), "cache_hits": t[6] or 0,
                    "fallbacks": t[7] or 0, "errors": t[8] or 0, "pii_redactions": t[9] or 0,
                }
                for t in teams
            ],
            "targets": [
                {"target": m[0], "requests": m[1], "cost_usd": round(m[2] or 0, 6), "avg_latency_ms": round(m[3] or 0),
                 "errors": m[4] or 0}
                for m in models
            ],
            "daily": [{"day": d[0], "cost_usd": round(d[1] or 0, 6), "requests": d[2]} for d in daily],
        }  # fmt: skip

    def close(self) -> None:
        self._conn.close()
