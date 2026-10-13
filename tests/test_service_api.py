from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from faq.api import create_app
from faq.config import Settings
from faq.rerank import PairTooLong
from faq.search.answer_groups import EquivalenceRegistry, ScoredQuestion
from faq.service import SearchService, ServiceUnavailable


class MemoryStore:
    def __init__(self, metadata):
        self.meta = metadata

    def snapshot_lock(self):
        return nullcontext()

    def metadata(self):
        return self.meta

    def answer(self, ident):
        return SimpleNamespace(answer="Stored answer, unchanged.")

    def question(self, ident):
        return "O'qish qanday?"

    def group_texts(self, question_ids):
        if getattr(self, "missing", False):
            raise RuntimeError("Rerank candidate missing or inactive")
        return {ident: f"Savol {ident}\nJavob {ident}" for ident in question_ids}


class FixedIndex:
    def __init__(self, score=0.9):
        self.score = score

    def generation(self):
        return "generation1"

    def search(self, vector, **kwargs):
        return [ScoredQuestion(1, 2, self.score)]


class RecordingEncoder:
    def encode(self, texts):
        self.texts = texts
        return np.zeros((len(texts), 1024), dtype=np.float32)


def make_service(score=0.9, *, index=None, reranker=None, **overrides):
    settings = replace(
        Settings.load(),
        profile="uz_surface_v1",
        min_score=0.8,
        min_margin=0.02,
        allow_uncalibrated=True,
        **{"rerank_enabled": False, **overrides},
    )
    registry = EquivalenceRegistry("v1", ())
    meta = {
        "generation": "generation1",
        "profile": settings.profile,
        "model_name": settings.model_name,
        "model_revision": settings.model_revision,
        "registry_digest": registry.digest,
        "embedding_device": settings.device,
        "max_tokens": 512,
    }
    return SearchService(
        settings,
        storage=MemoryStore(meta),
        index=index or FixedIndex(score),
        encoder=RecordingEncoder(),
        registry=registry,
        reranker=reranker,
    )


def test_api_normalizes_query_and_returns_exact_stored_answer():
    service = make_service()
    with TestClient(create_app(service)) as client:
        result = client.post("/search", json={"question": "Oʻqish qanday?"})
        assert result.status_code == 200
        assert result.json()["answer"] == "Stored answer, unchanged."
        assert service.encoder.texts == ["O'qish qanday?"]
        assert result.json()["request_id"]
        assert client.get("/ready").status_code == 200


def test_no_match_is_distinct_from_dependency_failure():
    service = make_service(0.3)
    with TestClient(create_app(service)) as client:
        assert client.post("/search", json={"question": "question"}).json()["status"] == "no_match"
        service.storage.meta["generation"] = "broken"
        assert client.get("/ready").status_code == 503
        assert client.post("/search", json={"question": "question"}).status_code == 503
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("question", [" ", "x" * 2001, "a\x00b"])
def test_invalid_inputs_are_rejected(question):
    with TestClient(create_app(make_service())) as client:
        assert client.post("/search", json={"question": question}).status_code == 422


def test_profile_mismatch_never_serves_mixed_generations():
    service = make_service()
    service.storage.meta["profile"] = "different"
    with pytest.raises(ServiceUnavailable):
        service.search("question")


class BandIndex(FixedIndex):
    def search(self, vector, **kwargs):
        return [ScoredQuestion(1, 2, 0.70), ScoredQuestion(3, 4, 0.69)]


class FakeReranker:
    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.calls = scores, error, []

    def score(self, query, texts):
        self.calls.append((query, texts))
        if self.error:
            raise self.error
        return self.scores


RERANK = {"rerank_enabled": True, "rerank_floor": 0.6, "rerank_min_score": 0.5, "rerank_min_margin": 0.1}


def test_uncertain_match_is_reranked_and_answer_stays_verbatim():
    reranker = FakeReranker([0.2, 0.9])
    service = make_service(index=BandIndex(), reranker=reranker, **RERANK)
    with TestClient(create_app(service)) as client:
        body = client.post("/search", json={"question": "Oʻqish qanday?"}).json()
    assert (body["status"], body["stage"], body["answer_id"]) == ("matched", "rerank", 4)
    assert body["answer"] == "Stored answer, unchanged."
    assert body["score"] == pytest.approx(0.70) and body["rerank_score"] == 0.9
    assert reranker.calls == [("O'qish qanday?", ["Savol 1\nJavob 1", "Savol 3\nJavob 3"])]


def test_reranker_failure_is_unavailable_not_no_match():
    service = make_service(index=BandIndex(), reranker=FakeReranker(error=RuntimeError("boom")), **RERANK)
    with TestClient(create_app(service)) as client:
        assert client.post("/search", json={"question": "savol"}).status_code == 503


def test_missing_rerank_candidate_is_unavailable():
    service = make_service(index=BandIndex(), reranker=FakeReranker([0.9, 0.1]), **RERANK)
    service.storage.missing = True
    with pytest.raises(ServiceUnavailable):
        service.search("savol")


def test_too_long_pair_keeps_dense_no_match():
    service = make_service(index=BandIndex(), reranker=FakeReranker(error=PairTooLong("long")), **RERANK)
    result = service.search("savol")
    assert (result["status"], result["reason"], result["stage"]) == ("no_match", "low_similarity", "dense")


def test_below_floor_and_confident_matches_skip_reranker():
    reranker = FakeReranker([0.9])
    assert make_service(0.5, reranker=reranker, **RERANK).search("savol")["reason"] == "low_similarity"
    assert make_service(0.9, reranker=reranker, **RERANK).search("savol")["stage"] == "dense"
    assert reranker.calls == []


def test_disabled_reranker_keeps_dense_behavior():
    service = make_service(index=BandIndex())
    result = service.search("savol")
    assert (result["status"], result["reason"], result["stage"]) == ("no_match", "low_similarity", "dense")
    assert result["rerank_score"] is None and service.reranker is None
