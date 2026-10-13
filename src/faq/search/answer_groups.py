import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoredQuestion:
    question_id: int
    answer_id: int
    score: float


@dataclass(frozen=True)
class EquivalenceRegistry:
    version: str
    classes: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        seen = set()
        for group in self.classes:
            if len(group) < 2 or len(set(group)) != len(group) or seen.intersection(group):
                raise ValueError("Equivalence classes must be disjoint with at least two unique IDs")
            if any(type(value) is not int or value <= 0 for value in group):
                raise ValueError("Equivalence IDs must be positive integers")
            seen.update(group)

    def group_id(self, answer_id: int) -> int:
        return next((min(group) for group in self.classes if answer_id in group), answer_id)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps([self.version, sorted([sorted(g) for g in self.classes])]).encode()
        ).hexdigest()

    @classmethod
    def load(cls, path: str | Path, answer_ids=None):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls(raw["version"], tuple(tuple(g["answer_ids"]) for g in raw["groups"]))
        if answer_ids is not None and any(i not in answer_ids for group in registry.classes for i in group):
            raise ValueError("Registry references absent answers")
        return registry


@dataclass(frozen=True)
class AnswerGroup:
    group_id: int
    member_answer_ids: tuple[int, ...]
    best_answer_id: int
    best_score: float
    question_id: int


def group_answer_candidates(candidates, registry: EquivalenceRegistry) -> list[AnswerGroup]:
    buckets = {}
    for candidate in candidates:
        if not math.isfinite(candidate.score):
            raise ValueError("Non-finite similarity")
        buckets.setdefault(registry.group_id(candidate.answer_id), []).append(candidate)
    groups = []
    for ident, bucket in buckets.items():
        best = min(bucket, key=lambda c: (-c.score, c.answer_id, c.question_id))
        groups.append(
            AnswerGroup(
                ident,
                tuple(sorted({c.answer_id for c in bucket})),
                best.answer_id,
                best.score,
                best.question_id,
            )
        )
    return sorted(groups, key=lambda g: (-g.best_score, g.best_answer_id, g.question_id))


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str | None
    answer_id: int | None = None
    score: float | None = None
    margin: float | None = None
    question_id: int | None = None
    stage: str = "dense"
    rerank_score: float | None = None
    rerank_margin: float | None = None


def decide(groups, *, min_score: float, min_margin: float) -> Decision:
    if not (-1 <= min_score <= 1 and 0 <= min_margin <= 2):
        raise ValueError("Invalid cosine threshold or margin")
    if not groups:
        return Decision("no_match", "low_similarity")
    best = groups[0]
    margin = best.best_score - groups[1].best_score if len(groups) > 1 else None
    if best.best_score < min_score:
        return Decision("no_match", "low_similarity", score=best.best_score, margin=margin)
    if margin is not None and margin < min_margin:
        return Decision("no_match", "ambiguous", score=best.best_score, margin=margin)
    return Decision("matched", None, best.best_answer_id, best.best_score, margin, best.question_id)


def route_scores(score, margin, *, min_score: float, min_margin: float, floor: float) -> str:
    """Dense routing: confident accept, reject below the floor, otherwise send to the reranker."""
    if not floor <= min_score:
        raise ValueError("Rerank floor must not exceed the dense threshold")
    if score is None or score < floor:
        return "no_match"
    if score >= min_score and (margin is None or margin >= min_margin):
        return "matched"
    return "rerank"


def route(groups, *, min_score: float, min_margin: float, floor: float) -> str:
    if not groups:
        return "no_match"
    margin = groups[0].best_score - groups[1].best_score if len(groups) > 1 else None
    return route_scores(groups[0].best_score, margin, min_score=min_score, min_margin=min_margin, floor=floor)


def decide_reranked(groups, scores, *, min_score: float, min_margin: float) -> Decision:
    """Second-stage decision over dense-ordered groups; reranker scores are aligned with `groups`."""
    if not (0 <= min_score <= 1 and 0 <= min_margin <= 1):
        raise ValueError("Invalid reranker threshold or margin")
    if not groups or len(groups) != len(scores):
        raise ValueError("Each reranked group needs exactly one score")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Non-finite rerank score")
    order = sorted(
        range(len(groups)), key=lambda i: (-scores[i], groups[i].best_answer_id, groups[i].question_id)
    )
    best, top = groups[order[0]], scores[order[0]]
    gap = top - scores[order[1]] if len(order) > 1 else None
    context = {
        "score": groups[0].best_score,
        "margin": groups[0].best_score - groups[1].best_score if len(groups) > 1 else None,
        "stage": "rerank",
        "rerank_score": top,
        "rerank_margin": gap,
    }
    if top < min_score:
        return Decision("no_match", "rerank_low_score", **context)
    if gap is not None and gap < min_margin:
        return Decision("no_match", "rerank_ambiguous", **context)
    return Decision("matched", None, best.best_answer_id, question_id=best.question_id, **context)
