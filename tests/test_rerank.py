from types import SimpleNamespace

import numpy as np
import pytest

from faq.rerank import PairTooLong, Reranker


class FakeTokenizer:
    def __call__(self, query, text, add_special_tokens=True):
        return {"input_ids": list(range(len(query.split()) + len(text.split()) + 3))}


class FakeModel:
    tokenizer = FakeTokenizer()

    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def predict(self, pairs, **kwargs):
        self.calls.append(pairs)
        return np.array(self.scores[: len(pairs)], dtype=np.float32)


def make(scores):
    reranker = Reranker.__new__(Reranker)
    reranker.model = FakeModel(scores)
    reranker.settings = SimpleNamespace(reranker_name="m", reranker_revision="r", device="cpu")
    return reranker


def test_scores_follow_input_order():
    reranker = make([0.25, 0.75])
    assert reranker.score("savol", ["a", "b"]) == pytest.approx([0.25, 0.75])
    assert reranker.model.calls == [[("savol", "a"), ("savol", "b")]]
    assert reranker.score("savol", []) == []


def test_long_pair_is_rejected_not_truncated():
    reranker = make([0.5])
    with pytest.raises(PairTooLong):
        reranker.score("so'z " * 300, ["javob " * 300])
    assert reranker.model.calls == []


def test_non_finite_score_is_an_error():
    with pytest.raises(ValueError, match="Non-finite"):
        make([float("nan")]).score("savol", ["a"])


def test_evaluation_cache_reuses_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reranker = make([0.4])
    assert reranker.score("savol", ["a"], cache=True) == pytest.approx([0.4])
    assert reranker.score("savol", ["a"], cache=True) == pytest.approx([0.4])
    assert len(reranker.model.calls) == 1
