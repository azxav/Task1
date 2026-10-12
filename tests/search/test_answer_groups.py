import pytest

from faq.search.answer_groups import EquivalenceRegistry, ScoredQuestion, decide, group_answer_candidates


def test_equivalent_ids_do_not_cause_false_ambiguity():
    registry = EquivalenceRegistry("v1", ((380, 494),))
    candidates = [ScoredQuestion(1, 380, 0.90), ScoredQuestion(2, 494, 0.89), ScoredQuestion(3, 22, 0.70)]
    groups = group_answer_candidates(candidates, registry)
    assert len(groups) == 2 and groups[0].best_answer_id == 380
    assert decide(groups, min_score=0.80, min_margin=0.05).status == "matched"


def test_partial_overlap_stays_distinct_and_rejection_boundaries():
    registry = EquivalenceRegistry("v1", ())
    groups = group_answer_candidates([ScoredQuestion(1, 22, 0.9), ScoredQuestion(2, 144, 0.89)], registry)
    assert decide(groups, min_score=0.8, min_margin=0.02).reason == "ambiguous"
    assert decide(groups, min_score=0.91, min_margin=0).reason == "low_similarity"
    assert decide([], min_score=0.8, min_margin=0).status == "no_match"
    assert decide(groups, min_score=0.9, min_margin=0).status == "matched"


def test_duplicates_and_ties_are_deterministic():
    registry = EquivalenceRegistry("v1", ((380, 494),))
    groups = group_answer_candidates(
        [ScoredQuestion(8, 494, 0.9), ScoredQuestion(9, 380, 0.9), ScoredQuestion(2, 380, 0.85)], registry
    )
    assert len(groups) == 1 and groups[0].best_answer_id == 380


def test_invalid_equivalence_registry_rejected():
    with pytest.raises(ValueError):
        EquivalenceRegistry("v1", ((1, 2), (2, 3)))
