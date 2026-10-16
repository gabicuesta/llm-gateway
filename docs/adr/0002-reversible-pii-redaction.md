# ADR 0002: Reversible PII redaction at the gateway

- **Status:** Accepted
- **Date:** 2026-10-16

## Context

Users paste customer data into prompts: names, DNIs, emails, phone numbers, IBANs. Sending that to a third-party model provider creates GDPR exposure. Simply masking the data, though, makes many answers useless, for example *"draft an email to ana@…"*.

## Decision

- Detect Spanish-relevant identifiers with regexes **plus checksum validation**, to keep false positives low:
  - DNI/NIE: control letter
  - IBAN: mod-97
  - Card numbers: Luhn
- Replace each value with a stable, numbered placeholder (`[EMAIL_1]`). The same value always gets the same placeholder within a request, so the model can still reason about who is who.
- Keep the mapping **only in memory, only for that request**. Put the original values back in the response.
- For streams, use a small look-ahead buffer so a placeholder split across chunks (`[EMA` + `IL_1]`) is still restored.
- Build the cache key from the redacted request, so personal data is never part of a key.
- Offer a `block` mode for teams that must not send personal data at all.

## Consequences

- The provider never sees the values. The user sees a normal answer.
- Regexes don't catch free-text personal data like names or addresses. That needs an NER model, which is on the roadmap.
- The redaction count appears in headers, metrics and the ledger, so compliance can see how often it happens.
