"""Teams: virtual API keys, model permissions, monthly budgets and rate limits."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import deque

from .config import GatewayConfig, Team

KEY_PREFIX = "gw-"


def new_key() -> tuple[str, str]:
    """Return (plain key, sha256 hash). Only the hash goes in the config."""
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return key, hash_key(key)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyRing:
    def __init__(self, config: GatewayConfig):
        self._by_hash = {h: team for team in config.teams for h in team.key_hashes}
        self._admin = config.admin_key_hash

    def team_for(self, key: str | None) -> Team | None:
        if not key:
            return None
        return self._by_hash.get(hash_key(key))

    def is_admin(self, key: str | None) -> bool:
        return bool(key and self._admin and hmac.compare_digest(hash_key(key), self._admin))


class RateLimiter:
    """Sliding one-minute window per team, in memory (one gateway instance)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def allow(self, team: Team) -> tuple[bool, int]:
        """Return (allowed, seconds to wait)."""
        if not team.requests_per_minute:
            return True, 0
        now = self._clock()
        window = self._hits.setdefault(team.name, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= team.requests_per_minute:
            return False, int(60 - (now - window[0])) + 1
        window.append(now)
        return True, 0
