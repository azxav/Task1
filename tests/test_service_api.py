from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from faq.api import create_app
from faq.config import Settings
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


def make_service(score=0.9):
    settings = replace(
        Settings.load(), profile="uz_surface_v1", min_score=0.8, min_margin=0.02, allow_uncalibrated=True
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
        index=FixedIndex(score),
        encoder=RecordingEncoder(),
        registry=registry,
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
