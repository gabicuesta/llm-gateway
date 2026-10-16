from __future__ import annotations

import httpx

from ..config import ProviderConfig
from .anthropic import AnthropicProvider
from .base import BadRequest, ChatResult, Provider, StreamChunk, UpstreamError
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatProvider

ADAPTERS = {"openai": OpenAICompatProvider, "anthropic": AnthropicProvider, "gemini": GeminiProvider}


def build_providers(configs: dict[str, ProviderConfig], client: httpx.AsyncClient) -> dict[str, Provider]:
    return {
        name: ADAPTERS[cfg.type](name, client, cfg.api_key, cfg.base_url, cfg.timeout_s)
        for name, cfg in configs.items()
    }


__all__ = ["BadRequest", "ChatResult", "Provider", "StreamChunk", "UpstreamError", "build_providers"]
