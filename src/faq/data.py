"""Validate complete input before touching either database."""

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .text.normalization import normalize, validate_text
from .text.scripts import classify_scripts, find_collisions


class DataValidationError(ValueError):
    """An input file/sheet contains an invalid row."""


@dataclass(frozen=True)
class Answer:
    id: int
    category_id: int
    topic_id: int
    answer: str
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Question:
    id: int
    answer_id: int
    question: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    embedding_text: str


@dataclass
class Dataset:
    questions: dict[int, Question]
    answers: dict[int, Answer]
    hashes: dict[str, str]
    warnings: list[dict]

    @property
    def canonical_digest(self) -> str:
        # Parsed values, stable row order, independent of CSV/Excel serialization.
        questions = []
        for q in sorted(self.questions.values(), key=lambda row: row.id):
            row = asdict(q)
            row.pop("embedding_text")  # Derived profile has separate provenance.
            questions.append(row)
        corpus = {
            "version": 1,
            "questions": questions,
            "answers": [asdict(a) for a in sorted(self.answers.values(), key=lambda row: row.id)],
        }
        serialized = json.dumps(
            corpus,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=lambda value: value.isoformat(),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def indexed(self) -> list[Question]:
        return [
            q for q in self.questions.values() if q.is_active and self.answers[q.answer_id].status == "active"
        ]


ANSWER_COLUMNS = {"id", "category_id", "topic_id", "answer", "status", "created_at", "updated_at"}
QUESTION_COLUMNS = {"id", "answer_id", "question", "is_active", "created_at", "updated_at"}


def _rows(source: Path, name: str):
    required = ANSWER_COLUMNS if name == "answers" else QUESTION_COLUMNS
    if source.is_dir():
        path = source / f"{name}.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            rows = list(reader)
    else:
        from openpyxl import load_workbook

        with source.open("rb") as stream:
            workbook = load_workbook(stream, read_only=True, data_only=True)
            try:
                sheet = workbook[name]
                values = iter(sheet.values)
                headers = list(next(values, ()))
                rows = [
                    dict(zip(headers, row, strict=False)) for row in values if any(v is not None for v in row)
                ]
            except KeyError as exc:
                raise DataValidationError(f"{source}/{name}: missing sheet") from exc
            finally:
                workbook.close()
    if len(headers) != len(set(headers)) or not required.issubset(headers):
        raise DataValidationError(
            f"{source}/{name}: missing or duplicate columns; required {sorted(required)}"
        )
    if not rows:
        raise DataValidationError(f"{source}/{name}: empty input")
    return rows


def _positive(value):
    # Prevent silently rounding Excel floats or accepting booleans as identifiers.
    if isinstance(value, bool) or str(value).strip() != str(int(value)) or int(value) <= 0:
        raise ValueError("IDs must be positive integers")
    return int(value)


def _date(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def load_dataset(source: str | Path) -> Dataset:
    source = Path(source)
    if not source.is_dir() and source.suffix.lower() != ".xlsx":
        raise DataValidationError("Use a directory with questions.csv/answers.csv or an .xlsx workbook")
    answers, questions, warnings = {}, {}, []
    for name, output in (("answers", answers), ("questions", questions)):
        for row_number, row in enumerate(_rows(source, name), start=2):
            try:
                ident = _positive(row["id"])
                if ident in output:
                    raise ValueError(f"duplicate ID {ident}")
                dates = {key: _date(row[key]) for key in ("created_at", "updated_at")}
                if name == "answers":
                    text = row["answer"]
                    validate_text(text)
                    if row["status"] not in ("active", "inactive"):
                        raise ValueError("status must be active or inactive")
                    output[ident] = Answer(
                        ident,
                        _positive(row["category_id"]),
                        _positive(row["topic_id"]),
                        text,
                        row["status"],
                        **dates,
                    )
                else:
                    reference = _positive(row["answer_id"])
                    if reference not in answers:
                        raise ValueError(f"unknown answer_id {reference}")
                    active = str(row["is_active"]).casefold()
                    if active not in ("true", "false"):
                        raise ValueError("is_active must be true or false")
                    text = row["question"]
                    prepared = normalize(text)
                    output[ident] = Question(
                        ident,
                        reference,
                        text,
                        active == "true",
                        **dates,
                        embedding_text=prepared.embedding_text,
                    )
                script = classify_scripts(text)
                if script.mixed_tokens:
                    warnings.append(
                        {"kind": "mixed_script", "table": name, "id": ident, "tokens": script.mixed_tokens}
                    )
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise DataValidationError(f"{source}/{name} row {row_number}: {exc}") from exc
    for collision in find_collisions(
        [{"id": q.id, "answer_id": q.answer_id, "question": q.question} for q in questions.values()]
    ):
        warnings.append({"kind": "collision", **asdict(collision)})
    paths = [source / "questions.csv", source / "answers.csv"] if source.is_dir() else [source]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return Dataset(questions, answers, hashes, warnings)
