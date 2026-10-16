"""Exact-match response cache, scoped per team.

Only deterministic requests are cached (temperature 0, or the client opts in with the
`x-gateway-cache: on` header). The key is computed on the *redacted* request, so no personal
data is ever used as a cache key, and it includes the team so answers never leak between teams.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict

from .providers import ChatResult

KEY_FIELDS = (
    "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed", "response_format",
    "tools", "tool_choice",
)  # fmt: skip


def cache_key(team: str, alias: str, request: dict) -> str:
    payload = {k: request.get(k) for k in KEY_FIELDS if request.get(k) is not None}
    blob = json.dumps({"team": team, "alias": alias, **payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def cacheable(request: dict, header: str | None) -> bool:
    if header == "off":
        return False
    return header == "on" or request.get("temperature") == 0


class ResponseCache:
    def __init__(self, ttl_s: int, max_entries: int, clock=time.monotonic):
        self._ttl = ttl_s
        self._max = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, ChatResult, str]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[ChatResult, str] | None:
        item = self._data.get(key)
        if item and self._clock() - item[0] < self._ttl:
            self._data.move_to_end(key)
            self.hits += 1
            return item[1], item[2]
        if item:
            del self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, result: ChatResult, target: str) -> None:
        self._data[key] = (self._clock(), result, target)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)
