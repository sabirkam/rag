"""Offline tests: no download, no embedding model, no LLM call."""

import zlib
from datetime import date

import numpy as np
import pytest

import rag


def store(id, name, city, street, phone="257210978", hours="Mo 9:00-21:00;Tu 9:00-21:00;We 9:00-21:00;Th 9:00-21:00;Fr 9:00-21:00;Sa 9:00-21:00;Su 9:00-21:00", **extra):
    return {
        "id": id, "name": name, "description": "Prodejna O2",
        "address": {"city": city, "street": street},
        "emails": [{"email": f"{city}@o2.cz"}],
        "phones": [{"number": phone}] if phone else None,
        "openingHours": hours, **extra,
    }


STORES = [
    store("734", "OC Metropole Zličín, Řevnická 121/1, Praha", "Praha", "Řevnická 121/1"),
    store("1", "OC Tesco, Hrádecká 1310, Sušice", "Sušice", "Hrádecká 1310", phone="376555111",
          hours="Mo 8:00-18:00;Tu 8:00-18:00;We 8:00-18:00;Th 8:00-18:00;Fr 8:00-18:00;Sa 8:00-12:00;Su off"),
    store("794", "OC Tesco, Lidická 3341/137, Břeclav", "Břeclav", "Lidická 3341/137", phone="519322000"),
    store("914", "OC Novo Plaza, Novodvorská 1800/136, Praha", "Praha", "Novodvorská 1800/136", phone=None,
          description="U hlavního vchodu"),
]


class FakeEncoder:
    """Deterministic bag-of-tokens vectors standing in for E5."""

    def encode(self, texts, normalize_embeddings=True):
        single = isinstance(texts, str)
        vectors = np.zeros((1 if single else len(texts), 64))
        for row, text in enumerate([texts] if single else texts):
            for token in rag.tokenize(text):
                vectors[row, zlib.crc32(token.encode()) % 64] += 1
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors[0] if single else vectors


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DATA_DIR", tmp_path)  # keep the vector cache out of the repo
    return rag.Index([rag.to_chunk(s) for s in STORES], FakeEncoder())


def test_chunk_contains_contact_and_collapsed_hours():
    text = rag.to_chunk(STORES[1]).text
    assert text.startswith("Prodejna O2: OC Tesco, Hrádecká 1310, Sušice")
    assert "Telefon: +420 376 555 111" in text
    assert "Otevírací doba: pondělí–pátek 8:00–18:00, sobota 8:00–12:00, neděle zavřeno" in text


def test_chunk_handles_missing_phone_note_and_holiday_exception():
    record = dict(STORES[3], openingHoursRaw=[
        {"date": "2026-11-17", "from": "", "to": "", "status": "Zavřeno"},  # Tuesday, normally open
        {"date": "2026-11-18", "from": "9:00", "to": "21:00", "status": "Otevřeno"},  # regular, not listed
    ])
    text = rag.to_chunk(record).text
    assert "Telefon: neuveden" in text
    assert "Poznámka: U hlavního vchodu" in text
    assert "Upravená otevírací doba: úterý 17. 11. 2026 zavřeno" in text
    assert "18. 11." not in text


def test_tokenize_strips_diacritics_stopwords_and_inflection():
    assert rag.tokenize("Jaký je e-mail na Zličíně?") == ["email", "zlici"]
    assert rag.tokenize("Zličín") == rag.tokenize("zlicine")
    assert rag.tokenize("Sušici v Sušice", stem=False) == ["susici", "v", "susice"]


def test_rrf_rewards_documents_ranked_high_by_both_lists():
    fused = rag.rrf([[1, 2, 3], [3, 1, 4]])
    assert [doc for doc, _ in fused][:2] == [1, 3]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


@pytest.mark.parametrize("question, expected_id", [
    ("What is the phone number of the O2 store in OC Metropole Zličín?", "734"),
    ("What are the opening hours of the O2 store in OC Tesco, Hrádecká 1310, Sušice?", "1"),
])
def test_hybrid_retrieval_ranks_the_right_store_first(index, question, expected_id):
    assert index.search(question, k=2)[0][0].id == expected_id


def test_ask_sends_date_context_and_question_to_the_llm(index):
    calls = []

    def fake_llm(system, user):
        calls.append((system, user))
        return "Telefonní číslo je +420 257 210 978.\nZdroj: OC Metropole Zličín, Řevnická 121/1, Praha"

    answer, hits = rag.ask("Telefon na O2 Zličín?", index, k=2, llm=fake_llm, today=date(2026, 11, 14))
    system, user = calls[0]
    assert "Nevím, tuto informaci v dostupných datech nemám." in system
    assert user.startswith("Dnes je sobota 14. 11. 2026, zítra je neděle 15. 11. 2026.")
    assert "[1]\nProdejna O2: OC Metropole Zličín" in user and user.endswith("DOTAZ: Telefon na O2 Zličín?")
    assert answer.startswith("Telefonní číslo") and hits[0][0].id == "734"
