import pytest

from faq.evaluation.rerank_calibration import adopt, apply_rerank, select_rerank

DENSE = {"min_score": 0.76, "min_margin": 0.04}


def dev_row(expected, *, score, margin, predicted=None, groups=(), rerank=None, split="calibration"):
    return {
        "split": split,
        "slice": "base",
        "expected": list(expected),
        "ranked": [a for a, _, _ in groups],
        "score": score,
        "margin": margin,
        "predicted": predicted,
        "reason": None if predicted else "low_similarity",
        "top_groups": [{"answer_id": a, "question_id": q, "score": s} for a, q, s in groups],
        "rerank_scores": rerank,
    }


def rows():
    return [
        dev_row([1], score=0.90, margin=0.10, predicted=1, groups=[(1, 10, 0.90)]),
        dev_row([2], score=0.70, margin=0.01, groups=[(3, 30, 0.70), (2, 20, 0.69)], rerank=[0.1, 0.9]),
        dev_row([], score=0.72, margin=0.02, groups=[(4, 40, 0.72), (5, 50, 0.70)], rerank=[0.3, 0.2]),
    ]


def test_selection_recovers_band_answer_and_rejects_unsupported_request():
    choice = select_rerank(rows(), **DENSE)
    assert (choice["rerank_floor"], choice["rerank_min_score"], choice["rerank_min_margin"]) == (
        0.7,
        0.9,
        0.5,
    )
    assert choice["quality_target_met"] is True
    assert choice["calibration"]["correct_accepted"] == 2
    assert choice["calibration"]["false_accepted"] == 0


def test_selection_refuses_test_rows():
    with pytest.raises(ValueError, match="development"):
        select_rerank([*rows(), dev_row([1], score=0.9, margin=0.1, predicted=1, split="test")], **DENSE)


def test_apply_rerank_keeps_dense_outside_band_and_for_skipped_pairs():
    confident, band, _ = rows()
    values = {**DENSE, "floor": 0.7, "r_score": 0.9, "r_margin": 0.5}
    assert apply_rerank(confident, **values)["stage"] == "dense"
    reranked = apply_rerank(band, **values)
    assert (reranked["stage"], reranked["predicted"], reranked["rerank_score"]) == ("rerank", 2, 0.9)
    skipped = apply_rerank({**band, "rerank_scores": None}, **values)
    assert (skipped["stage"], skipped["predicted"], skipped["reason"]) == ("dense", None, "low_similarity")


def test_adoption_requires_more_correct_answers_within_quality_targets():
    dense = {"correct_accepted": 2, "accepted_precision": 1.0, "false_acceptance": 0.0}
    assert not adopt(dense, dense)
    assert not adopt(dense, {"correct_accepted": 3, "accepted_precision": 0.9, "false_acceptance": 0.0})
    assert not adopt(dense, {"correct_accepted": 3, "accepted_precision": 1.0, "false_acceptance": 0.1})
    assert adopt(dense, {"correct_accepted": 3, "accepted_precision": 0.96, "false_acceptance": 0.05})
