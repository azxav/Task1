import os
from pathlib import Path

import numpy as np
import pytest

from faq.data import load_dataset
from faq.index import VectorIndex
from faq.storage import Storage

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="requires local PostgreSQL and Qdrant"),
]


def test_real_snapshot_and_index_publication():
    store = Storage(
        os.getenv("TEST_DATABASE_URL", "postgresql://faq:faq_local@127.0.0.1:55432/faq"), schema="faq_test"
    )
    index = VectorIndex(os.getenv("TEST_QDRANT_URL", "http://127.0.0.1:56333"), alias="faq_test_questions")
    dataset = load_dataset(Path(__file__).parents[2] / "data/source")
    store.initialize()
    subset = dataset.indexed[:2]
    vectors = np.zeros((2, 1024), dtype=np.float32)
    vectors[:, 0] = 1
    index.create_generation("faq_test_generation", subset, vectors)
    metadata = {"generation": "faq_test_generation", "profile": "uz_surface_v1", "model_revision": "test"}
    store.replace(dataset, metadata)
    store.replace(dataset, metadata)
    assert store.counts() == {"questions": 3048, "answers": 505}
    assert store.answer(499).answer == dataset.answers[499].answer
    first = dataset.indexed[0]
    expected = first.question + "\n" + dataset.answers[first.answer_id].answer
    assert store.group_texts([first.id]) == {first.id: expected}
    with pytest.raises(RuntimeError, match="Rerank"):
        store.group_texts([10**9])
    index.publish("faq_test_generation")
    assert index.generation() == "faq_test_generation"
    hits = index.search(vectors[0], expected_generation="faq_test_generation")
    assert {hit.question_id for hit in hits} == {subset[0].id, subset[1].id}
    with pytest.raises(RuntimeError, match="generation"):
        index.search(vectors[0], expected_generation="another_generation")
