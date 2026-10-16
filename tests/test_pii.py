import pytest

from llm_gateway.pii import Redactor, StreamRestorer, Vault, restore

ALL = ["email", "dni_nie", "phone_es", "iban", "credit_card"]


def redact(text: str) -> tuple[str, Vault]:
    vault = Vault()
    return Redactor(ALL).redact_text(text, vault), vault


@pytest.mark.parametrize(
    "text,label",
    [
        ("Escribe a ana.garcia@empresa.es mañana", "EMAIL"),
        ("Mi DNI es 12345678Z", "DNI"),
        ("NIE X1234567L por favor", "DNI"),
        ("Llámame al 612 345 678", "PHONE"),
        ("o al +34 912345678", "PHONE"),
        ("IBAN ES91 2100 0418 4502 0005 1332", "IBAN"),
        ("Tarjeta 4111 1111 1111 1111", "CARD"),
    ],
)
def test_detects_and_masks(text, label):
    out, vault = redact(text)
    assert f"[{label}_1]" in out
    assert vault.total == 1
    assert restore(out, vault) == text


@pytest.mark.parametrize(
    "text",
    [
        "Mi DNI es 12345678A",  # wrong control letter
        "IBAN ES91 2100 0418 4502 0005 1333",  # bad checksum
        "Pedido 4111 1111 1111 1112",  # fails Luhn
        "El presupuesto es 512345678 euros",  # 5 is not a Spanish mobile/landline prefix
        "Referencia 2026-10-16",
    ],
)
def test_ignores_lookalikes(text):
    out, vault = redact(text)
    assert out == text and vault.total == 0


def test_same_value_same_token_and_numbering():
    out, vault = redact("a@b.es, c@d.es y otra vez a@b.es")
    assert out == "[EMAIL_1], [EMAIL_2] y otra vez [EMAIL_1]"
    assert vault.counts == {"EMAIL": 2}


def test_redacts_text_parts_of_multimodal_messages():
    req = {
        "messages": [
            {"role": "system", "content": "Eres útil."},
            {"role": "user", "content": [{"type": "text", "text": "Soy 12345678Z"}, {"type": "image_url", "image_url": {"url": "x"}}]},
        ]
    }  # fmt: skip
    out, vault = Redactor(ALL).redact_request(req)
    assert out["messages"][1]["content"][0]["text"] == "Soy [DNI_1]"
    assert req["messages"][1]["content"][0]["text"] == "Soy 12345678Z"  # original untouched
    assert vault.total == 1


def test_stream_restorer_handles_tokens_split_across_chunks():
    _, vault = redact("ana@empresa.es")
    r = StreamRestorer(vault)
    pieces = ["Hola ", "[EMA", "IL_", "1]", ", ¿qué tal? [no es token"]
    out = "".join(r.feed(p) for p in pieces) + r.flush()
    assert out == "Hola ana@empresa.es, ¿qué tal? [no es token"


def test_only_enabled_entities():
    vault = Vault()
    out = Redactor(["email"]).redact_text("x@y.es 12345678Z", vault)
    assert out == "[EMAIL_1] 12345678Z"
