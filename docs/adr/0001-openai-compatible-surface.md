# ADR 0001: OpenAI's Chat Completions API as the gateway's surface

- **Status:** Accepted
- **Date:** 2026-10-16

## Context

A gateway only helps if teams actually route their traffic through it. Every bit of friction, such as a new SDK, a new request format or code changes in every service, slows adoption.

## Decision

- Expose `/v1/chat/completions` and `/v1/models` exactly as OpenAI defines them, including SSE streaming and `stream_options.include_usage`.
- Clients use **aliases** (`smart`, `balanced`, `fast`) instead of vendor model names. Which model sits behind an alias is a platform decision that can change without touching application code.
- Each provider adapter translates the OpenAI request to its vendor's format and back: system prompts, alternating roles, images, stop sequences, finish reasons and usage.
- Gateway metadata travels in `x-gateway-*` response headers, never in the body, so responses stay schema-compatible.

## Consequences

- Adopting the gateway is a `base_url` change, and the official OpenAI SDK test proves it.
- Features without an equivalent in another vendor's API are limited in v1. Tool calling only works on OpenAI-compatible targets; on other targets the gateway skips to the next target or returns 400.
