"""Diagnostics only: script flags never rewrite source text."""

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Sequence, TypedDict
import unicodedata as ud

from .normalization import normalize


class QuestionRow(TypedDict):
    id: int
    answer_id: int
    question: str


@dataclass(frozen=True)
class ScriptDiagnostics:
    aggregate_script: str
    mixed_tokens: tuple[str, ...]


def _scripts(text: str) -> set[str]:
    return {script for char in text for script in ("LATIN", "CYRILLIC") if script in ud.name(char, "")}


def classify_scripts(text: str) -> ScriptDiagnostics:
    scripts = _scripts(text)
    mixed = tuple(token for token in re.findall(r"[^\W\d_]+(?:['ʻʼ‘’][^\W\d_]+)*", text)
                  if len(_scripts(token)) > 1)
    aggregate = "mixed" if len(scripts) > 1 else next(iter(scripts), "OTHER").lower()
    return ScriptDiagnostics(aggregate, mixed)


@dataclass(frozen=True)
class Collision:
    comparison_key: str
    question_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    cross_answer: bool


def find_collisions(rows: Sequence[QuestionRow]) -> list[Collision]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[normalize(row["question"]).comparison_key].append(row)
    result = []
    for key, group in grouped.items():
        if len(group) > 1:
            answers = tuple(sorted({row["answer_id"] for row in group}))
            result.append(Collision(key, tuple(sorted(row["id"] for row in group)), answers, len(answers) > 1))
    return result
