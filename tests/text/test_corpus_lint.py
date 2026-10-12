import csv
from pathlib import Path

from faq.text.normalization import letter_marker_context, normalize
from faq.text.scripts import classify_scripts, find_collisions

DATA = Path(__file__).parents[2] / "data/source"


def test_script_diagnostics_preserve_mixed_tokens():
    assert classify_scripts("ўқиш").aggregate_script == "cyrillic"
    assert classify_scripts("o'qish").aggregate_script == "latin"
    mixed = classify_scripts("Yangi egа my.gov.uz")
    assert mixed.aggregate_script == "mixed"
    assert mixed.mixed_tokens == ("egа",)
    assert classify_scripts("Ўқиш my.gov.uz").mixed_tokens == ()


def test_collision_report_keeps_all_source_ids():
    rows = [
        {"id": 1, "answer_id": 2, "question": "Oʻqish"},
        {"id": 2, "answer_id": 3, "question": "O'qish"},
        {"id": 3, "answer_id": 3, "question": "O'qish joyi"},
    ]
    (collision,) = find_collisions(rows)
    assert collision.question_ids == (1, 2)
    assert collision.answer_ids == (2, 3)
    assert collision.cross_answer
    assert rows[0]["question"] == "Oʻqish"


def test_actual_corpus_and_keyboard_variants_keep_meaning():
    with (DATA / "questions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3048
    for row in rows:
        source = row["question"]
        assert normalize(source).embedding_text == source, row["id"]
        for mark in "ʻʼ‘’`":
            mutant = "".join(
                mark if c == "'" and letter_marker_context(source, pos) else c for pos, c in enumerate(source)
            )
            assert normalize(mutant).embedding_text == source, (row["id"], mark)
    assert find_collisions([{**r, "id": int(r["id"]), "answer_id": int(r["answer_id"])} for r in rows]) == []
    with (DATA / "answers.csv").open(encoding="utf-8-sig", newline="") as stream:
        answer = next(r for r in csv.DictReader(stream) if r["id"] == "499")
    assert classify_scripts(answer["answer"]).mixed_tokens == ("egа",)
