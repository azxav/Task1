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
