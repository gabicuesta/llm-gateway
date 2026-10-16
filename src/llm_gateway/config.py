"""Configuration: one YAML file, validated at startup so mistakes fail fast and loudly."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ProviderType = Literal["openai", "anthropic", "gemini"]


class ProviderConfig(BaseModel):
    type: ProviderType
    api_key_env: str | None = Field(None, description="Env var holding the upstream API key")
    base_url: str | None = Field(None, description="Override, e.g. a local vLLM/Ollama OpenAI-compatible server")
    timeout_s: float = 60.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class Target(BaseModel):
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> Target:
        provider, sep, model = spec.partition(":")
        if not sep or not model:
            raise ValueError(f"Target '{spec}' must look like 'provider:model'.")
        return cls(provider=provider, model=model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


class Price(BaseModel):
    input: float = Field(description="USD per million input tokens")
    output: float = Field(description="USD per million output tokens")


class Team(BaseModel):
    name: str
    key_hashes: list[str] = Field(description="sha256 hashes of the team's virtual keys")
    monthly_budget_usd: float | None = None
    requests_per_minute: int | None = None
    allowed_models: list[str] | None = Field(None, description="Aliases this team may use; None means all")


class PiiConfig(BaseModel):
    enabled: bool = True
    mode: Literal["mask", "block"] = "mask"
    restore_in_response: bool = True
    entities: list[str] = ["email", "dni_nie", "phone_es", "iban", "credit_card"]


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = 3600
    max_entries: int = 2000


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, list[Target]] = Field(description="Alias -> ordered fallback chain")
    pricing: dict[str, Price] = {}
    teams: list[Team] = []
    admin_key_hash: str | None = None
    pii: PiiConfig = PiiConfig()
    cache: CacheConfig = CacheConfig()
    database: str = "gateway.db"
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 30.0

    @field_validator("models", mode="before")
    @classmethod
    def _parse_targets(cls, value: dict) -> dict:
        return {
            alias: [
                Target.parse(t) if isinstance(t, str) else t for t in (chain if isinstance(chain, list) else [chain])
            ]
            for alias, chain in value.items()
        }

    @model_validator(mode="after")
    def _check_references(self) -> GatewayConfig:
        for alias, chain in self.models.items():
            if not chain:
                raise ValueError(f"Model alias '{alias}' has no targets.")
            for target in chain:
                if target.provider not in self.providers:
                    raise ValueError(f"Alias '{alias}' uses unknown provider '{target.provider}'.")
        for team in self.teams:
            for alias in team.allowed_models or []:
                if alias not in self.models:
                    raise ValueError(f"Team '{team.name}' allows unknown model alias '{alias}'.")
        return self

    def price_for(self, model: str) -> Price | None:
        return self.pricing.get(model)


_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def load_config(path: str | Path) -> GatewayConfig:
    text = Path(path).read_text(encoding="utf-8")
    text = _ENV.sub(lambda m: os.environ.get(m.group(1), ""), text)  # ${VAR} interpolation
    return GatewayConfig.model_validate(yaml.safe_load(text))
