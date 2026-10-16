# ADR 0003: Per-team budgets enforced on real spend, with a SQLite ledger

- **Status:** Accepted
- **Date:** 2026-10-16

## Context

AI cost scales with usage in ways that are hard to predict. Finance needs a cost per team, and teams need a guardrail that stops runaway spend, such as a loop calling a large model, before it shows up on the invoice.

## Decision

- Each request is priced from the **actual token usage the provider reports**, multiplied by the configured price for the target model.
- A ledger row is written for every request, including errors and cache hits: team, alias, target, tokens, cost, latency, fallbacks, PII count.
- An in-memory monthly counter per team is updated as soon as a request finishes, so the budget takes effect immediately. It is rebuilt from SQLite at startup.
- The budget is checked **before** calling the provider. Once spend reaches the budget, the gateway returns HTTP 429 `budget_exceeded`. The `x-gateway-budget-remaining-usd` header lets clients see how much is left.

## Consequences

- One request can overshoot the budget, because its cost is only known afterwards. We accept that in exchange for zero added latency.
- SQLite and in-memory counters limit v1 to a single instance. Postgres and Redis are the scaling path.
- Prices live in configuration and must be kept up to date. Models without a price are tracked at $0 and are easy to spot in the dashboard.
