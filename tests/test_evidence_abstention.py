"""Regression checks for weak lexical evidence and exact source offsets."""

import pytest
from fastapi.testclient import TestClient

from rag_platform.api import create_app
from rag_platform.generator import grounded_answer
from rag_platform.store import Store


@pytest.fixture
def corpus(tmp_path):
    store = Store(tmp_path / "regression.db")
    store.upsert("async", "Async tasks", "https://docs.python.org/3/library/asyncio-task.html",
                 "The coroutine prints hello world. A coroutine won't run until awaited. " * 8)
    store.upsert("sqlite", "SQLite transactions", "https://docs.python.org/3/library/sqlite3.html",
                 "SQLite transactions are committed using commit. Rollback reverses a transaction. " * 8)
    store.upsert("release", "Release notes", "https://example.org/releases",
                 "The example toolkit release was published in 2024 with SQLite transaction support. " * 8)
    return store


@pytest.mark.parametrize("strategy", ["lexical", "vector", "hybrid"])
@pytest.mark.parametrize("query", ["Who won the 2099 World Cup?", "Who won the World Cup?"])
def test_accidental_world_and_contraction_are_not_evidence(corpus, strategy, query):
    assert corpus.search(query, strategy=strategy)
    result = corpus.answer(query, strategy=strategy)
    assert result["abstained"] and not result["citations"]


@pytest.mark.parametrize("strategy", ["lexical", "vector", "hybrid"])
def test_unsupported_number_rejected_without_rejecting_supported_number(corpus, strategy):
    assert corpus.answer("Toolkit release published in 2099", strategy=strategy)["abstained"]
    supported = corpus.answer("Toolkit release published in 2024", strategy=strategy)
    assert not supported["abstained"]
    assert all("2024" in c["quote"] for c in supported["citations"])


def test_terms_in_different_sentences_do_not_support_a_quote(tmp_path):
    store = Store(tmp_path / "split.db")
    store.upsert("separate", "Unrelated examples", "https://example.org/examples",
                 "A generic illustration contains the word planet. "
                 "A different illustration discusses the word recipe. " * 8)
    assert store.search("planet recipe", strategy="lexical")
    assert store.answer("planet recipe", strategy="lexical")["abstained"]


def test_each_citation_has_sufficient_coverage_and_original_offsets(corpus):
    result = corpus.answer("SQLite transactions commit rollback", strategy="lexical")
    assert not result["abstained"]
    assert all(c["doc"] == "sqlite" for c in result["citations"])
    with corpus.connect() as c:
        for citation in result["citations"]:
            chunk = c.execute("SELECT * FROM chunks WHERE id=?", (citation["chunk"],)).fetchone()
            start = citation["start"] - chunk["start"]
            end = citation["end"] - chunk["start"]
            assert chunk["text"][start:end] == citation["quote"]


def test_api_returns_abstention_and_generation_is_not_called(corpus):
    class MustNotGenerate:
        def generate(self, question, hits):
            raise AssertionError("Generation invoked without sufficient source evidence")

    response = TestClient(create_app(corpus)).post(
        "/query", json={"text": "Who won the 2099 World Cup?", "strategy": "lexical"})
    assert response.status_code == 200 and response.json()["abstained"]
    assert grounded_answer(corpus, MustNotGenerate(), "Who won the 2099 World Cup?",
                           strategy="lexical")["abstained"]
