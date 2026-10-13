import pytest

from faq.search.answer_groups import (
    AnswerGroup,
    EquivalenceRegistry,
    ScoredQuestion,
    decide,
    decide_reranked,
    group_answer_candidates,
    route,
)


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


def groups_of(*items):
    """(answer_id, question_id, dense_score) in dense order."""
    return [AnswerGroup(a, (a,), a, s, q) for a, q, s in items]


ROUTE = {"min_score": 0.76, "min_margin": 0.04, "floor": 0.62}


def test_route_boundaries():
    assert route(groups_of((1, 10, 0.76), (2, 20, 0.72)), **ROUTE) == "matched"
    assert route(groups_of((1, 10, 0.80), (2, 20, 0.79)), **ROUTE) == "rerank"
    assert route(groups_of((1, 10, 0.62)), **ROUTE) == "rerank"
    assert route(groups_of((1, 10, 0.6199)), **ROUTE) == "no_match"
    assert route([], **ROUTE) == "no_match"
    with pytest.raises(ValueError):
        route(groups_of((1, 10, 0.7)), min_score=0.76, min_margin=0.04, floor=0.8)


def test_rerank_decision_boundaries_keep_dense_context():
    groups = groups_of((1, 10, 0.74), (2, 20, 0.73), (3, 30, 0.70))
    decision = decide_reranked(groups, [0.2, 0.9, 0.5], min_score=0.9, min_margin=0.39)
    assert (decision.status, decision.answer_id, decision.question_id, decision.stage) == (
        "matched",
        2,
        20,
        "rerank",
    )
    assert decision.score == 0.74 and decision.margin == pytest.approx(0.01)
    assert decision.rerank_score == 0.9 and decision.rerank_margin == pytest.approx(0.4)
    low = decide_reranked(groups, [0.2, 0.9, 0.5], min_score=0.91, min_margin=0)
    assert (low.status, low.reason, low.answer_id) == ("no_match", "rerank_low_score", None)
    close = decide_reranked(groups, [0.2, 0.9, 0.5], min_score=0.5, min_margin=0.41)
    assert (close.status, close.reason) == ("no_match", "rerank_ambiguous")


def test_rerank_single_group_ties_equivalents_and_invalid_scores():
    single = decide_reranked(groups_of((7, 70, 0.7)), [0.6], min_score=0.5, min_margin=0.9)
    assert single.status == "matched" and single.rerank_margin is None
    tie = decide_reranked(groups_of((9, 90, 0.7), (4, 40, 0.69)), [0.8, 0.8], min_score=0.5, min_margin=0)
    assert tie.answer_id == 4 and tie.reason is None
    equivalent = [AnswerGroup(380, (380, 494), 494, 0.71, 5), AnswerGroup(22, (22,), 22, 0.70, 6)]
    assert decide_reranked(equivalent, [0.9, 0.2], min_score=0.5, min_margin=0.1).answer_id == 494
    with pytest.raises(ValueError):
        decide_reranked(groups_of((1, 10, 0.7)), [float("nan")], min_score=0.5, min_margin=0)
    with pytest.raises(ValueError):
        decide_reranked(groups_of((1, 10, 0.7)), [0.5, 0.4], min_score=0.5, min_margin=0)
    with pytest.raises(ValueError):
        decide_reranked([], [], min_score=0.5, min_margin=0)
