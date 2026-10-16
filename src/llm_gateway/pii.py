"""Reversible redaction of personal data before it leaves the company.

Detected values are replaced by stable placeholders ([EMAIL_1], [DNI_1]...) in the request,
and put back in the response, so users still get a useful answer while the provider never
sees the data. Detectors validate checksums (DNI/NIE letter, IBAN mod-97, card Luhn) to
keep false positives low.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass, field

DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def _valid_dni(m: re.Match) -> bool:
    raw = re.sub(r"[\s-]", "", m.group(0)).upper()
    number = raw[:-1].replace("X", "0").replace("Y", "1").replace("Z", "2")
    return number.isdigit() and DNI_LETTERS[int(number) % 23] == raw[-1]


def _valid_iban(m: re.Match) -> bool:
    raw = re.sub(r"\s", "", m.group(0)).upper()
    if not 15 <= len(raw) <= 34:
        return False
    rearranged = raw[4:] + raw[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return int(digits) % 97 == 1


def _valid_card(m: re.Match) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", m.group(0))]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class Detector:
    entity: str
    label: str
    pattern: re.Pattern
    validate: Callable[[re.Match], bool] = lambda m: True


# Order matters: earlier detectors win when matches overlap.
DETECTORS = [
    Detector("email", "EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    Detector("iban", "IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b"), _valid_iban),
    Detector("credit_card", "CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _valid_card),
    Detector("dni_nie", "DNI", re.compile(r"\b[XYZxyz]?\d{7,8}[\s-]?[A-Za-z]\b"), _valid_dni),
    Detector("phone_es", "PHONE", re.compile(r"(?<![\w+])(?:\+34|0034)?[\s.-]?[6789](?:[\s.-]?\d){8}\b")),
]


@dataclass
class Vault:
    """Placeholder ↔ original value, scoped to one request."""

    to_token: dict[str, str] = field(default_factory=dict)
    to_value: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def token_for(self, label: str, value: str) -> str:
        key = value.strip()
        if key not in self.to_token:
            self.counts[label] = self.counts.get(label, 0) + 1
            token = f"[{label}_{self.counts[label]}]"
            self.to_token[key] = token
            self.to_value[token] = key
        return self.to_token[key]

    @property
    def total(self) -> int:
        return len(self.to_value)


class Redactor:
    def __init__(self, entities: list[str]):
        self._detectors = [d for d in DETECTORS if d.entity in set(entities)]

    def redact_text(self, text: str, vault: Vault) -> str:
        spans: list[tuple[int, int, str]] = []
        for det in self._detectors:
            for m in det.pattern.finditer(text):
                start, end = m.start(), m.end()
                # Leading separators captured by the phone pattern are not part of the value.
                while start < end and text[start] in " .-":
                    start += 1
                if any(s < end and start < e for s, e, _ in spans) or not det.validate(m):
                    continue
                spans.append((start, end, det.label))
        for start, end, label in sorted(spans, reverse=True):
            text = text[:start] + vault.token_for(label, text[start:end]) + text[end:]
        return text

    def redact_request(self, request: dict) -> tuple[dict, Vault]:
        vault = Vault()
        redacted = copy.deepcopy(request)
        for message in redacted.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = self.redact_text(content, vault)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = self.redact_text(part.get("text", ""), vault)
        return redacted, vault


_TOKEN = re.compile(r"\[[A-Z]+_\d+\]")


def restore(text: str, vault: Vault) -> str:
    if not vault.total:
        return text
    return _TOKEN.sub(lambda m: vault.to_value.get(m.group(0), m.group(0)), text)


class StreamRestorer:
    """Restores placeholders in a stream, where a token like [EMAIL_1] can be split across chunks."""

    MAX_TOKEN = 24

    def __init__(self, vault: Vault):
        self._vault = vault
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        if not self._vault.total:
            return chunk
        self._buffer += chunk
        cut = self._buffer.rfind("[")
        if cut != -1 and "]" not in self._buffer[cut:] and len(self._buffer) - cut < self.MAX_TOKEN:
            ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        else:
            ready, self._buffer = self._buffer, ""
        return restore(ready, self._vault)

    def flush(self) -> str:
        out, self._buffer = restore(self._buffer, self._vault), ""
        return out
