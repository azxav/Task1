import pytest

from faq.evaluation.labels import validate_cases
from faq.evaluation.metrics import metrics, select_thresholds


def test_threshold_selection_counts_each_authored_family_once():
    from faq.evaluation.metrics import calibration_families

    base = {"split": "calibration", "slice": "base", "family_id": "a"}
    variant = {**base, "slice": "casing"}
    held_out = {"split": "test", "slice": "base", "family_id": "b"}
    assert calibration_families([base, variant, held_out]) == [base]


def case(ident="a", split="calibration", expected=(1,), family="f"):
    return {
        "case_id": ident,
        "family_id": family,
        "split": split,
        "query": "Mustaqil savol",
        "acceptable_answer_ids": list(expected),
        "expected_status": "matched" if expected else "no_match",
        "coverage_rationale": "Source answer covers the requested intent",
        "slice": "base",
    }


def test_labels_reject_split_leakage_and_unknown_answers():
    with pytest.raises(ValueError, match="family"):
        validate_cases([case(), case("b", "test")], {1}, [])
    with pytest.raises(ValueError, match="answer"):
        validate_cases([case(expected=(99,))], {1}, [])
    with pytest.raises(ValueError, match="overlap"):
        validate_cases([case()], {1}, ["Mustaqil savol"])


def test_metrics_distinguish_precision_coverage_and_false_acceptance():
    rows = [
        {"expected": [1], "predicted": 1, "ranked": [1, 2]},
        {"expected": [2], "predicted": None, "ranked": [1, 2]},
        {"expected": [], "predicted": 1, "ranked": [1]},
    ]
    result = metrics(rows)
    assert result["accepted_precision"] == 0.5
    assert result["answer_coverage"] == 0.5
    assert result["false_acceptance"] == 1
    assert result["recall_at_5"] == 1


def test_calibration_does_not_choose_reject_everything():
    rows = [
        {"expected": [1], "ranked": [1], "score": 0.90, "margin": 0.10},
        {"expected": [], "ranked": [2], "score": 0.40, "margin": 0.05},
    ]
    thresholds = select_thresholds(rows)
    assert thresholds["min_score"] <= 0.90
    assert thresholds["min_score"] > 0.40 or thresholds["min_margin"] > 0.05
