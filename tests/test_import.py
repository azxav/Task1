import csv
from pathlib import Path

import pytest
from openpyxl import Workbook

from faq.data import DataValidationError, load_dataset

SOURCE = Path(__file__).parents[1] / "data/source"


def test_actual_data_validated_and_originals_preserved():
    dataset = load_dataset(SOURCE)
    assert len(dataset.questions) == 3048 and len(dataset.answers) == 505
    assert dataset.answers[499].answer.endswith("rasmiylashtiradi.")
    assert "egа" in dataset.answers[499].answer
    assert dataset.questions[2419].embedding_text.endswith("qanday olinadi?")
    assert any(w["id"] == 499 for w in dataset.warnings)


def write_small(directory, *, reference=1, active="true", answer="Javob", duplicate=False):
    directory.mkdir(exist_ok=True)
    with (directory / "answers.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "category_id", "topic_id", "answer", "status", "created_at", "updated_at"])
        writer.writerow([1, 1, 1, answer, "active", "2026-08-04 00:00:00", "2026-08-04 00:00:00"])
    with (directory / "questions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "answer_id", "question", "is_active", "created_at", "updated_at"])
        row = [1, reference, "Oʻqish qanday?", active, "2026-08-04 00:00:00", "2026-08-04 00:00:00"]
        writer.writerow(row)
        if duplicate:
            writer.writerow(row)


@pytest.mark.parametrize(
    "kwargs", [{"reference": 99}, {"active": "perhaps"}, {"answer": " "}, {"duplicate": True}]
)
def test_bad_rows_rejected_before_database_changes(tmp_path, kwargs):
    write_small(tmp_path, **kwargs)
    with pytest.raises(DataValidationError, match="row"):
        load_dataset(tmp_path)


def test_excel_has_same_semantics(tmp_path):
    write_small(tmp_path)
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name in ("questions", "answers"):
        sheet = workbook.create_sheet(name)
        with (tmp_path / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                sheet.append(row)
    path = tmp_path / "input.xlsx"
    workbook.save(path)
    dataset = load_dataset(path)
    assert dataset.questions[1].question == "Oʻqish qanday?"
    assert dataset.questions[1].embedding_text == "O'qish qanday?"
    assert dataset.canonical_digest == load_dataset(tmp_path).canonical_digest
