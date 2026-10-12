"""Materialize reviewed base queries and controlled variants; no model predictions used."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from faq.data import load_dataset
from faq.evaluation.labels import validate_cases
from faq.text.normalization import letter_marker_context

root = Path("evaluation/uzbek")
dataset = load_dataset("data/source")
counts, cases, parents = defaultdict(int), [], {}
review = json.loads((root / "label_review.json").read_text(encoding="utf-8"))
alternatives = {int(k): v["alternative_ids"] for k, v in review["reviewed_alternatives"].items()}


def add(parent, query, slice_name):
    if query == parent["query"]:
        return
    cases.append(
        {
            **parent,
            "case_id": parent["case_id"] + "-" + slice_name,
            "query": query,
            "slice": slice_name,
            "parent_case_id": parent["case_id"],
            "coverage_rationale": parent["coverage_rationale"] + "; reviewed same-intent writing variation",
        }
    )


with (root / "base_queries.tsv").open(encoding="utf-8", newline="") as stream:
    for n, row in enumerate(csv.DictReader(stream, delimiter="\t"), 1):
        expected = [int(i) for i in row["answer_ids"].split(",")]
        primary = expected[0]
        expected += [i for i in alternatives.get(primary, []) if i not in expected]
        category = dataset.answers[primary].category_id
        split = "calibration"  # All v1 families inspected during review; no longer held out.
        counts[category] += 1
        query = row["query"]
        parent = {
            "case_id": f"q{n:03}",
            "family_id": f"q{n:03}",
            "split": split,
            "query": query,
            "category_id": category,
            "acceptable_answer_ids": expected,
            "expected_status": "matched",
            "slice": "base",
            "coverage_rationale": f"Read full source answers {expected}; requested intent is covered. Equivalent/overlapping IDs explicitly listed, not inferred from predictions.",
        }
        cases.append(parent)
        parents[primary] = parent
        mark_query = "".join(
            ("ʻ" if pos and query[pos - 1].lower() in "og" else "ʼ")
            if c == "'" and letter_marker_context(query, pos)
            else c
            for pos, c in enumerate(query)
        )
        add(parent, mark_query, "typography")
        add(parent, query.upper(), "casing")
        add(parent, "\ufeff  " + query.replace(" ", "\u00a0\t") + "  ", "whitespace")
        add(
            parent,
            "".join(c for pos, c in enumerate(query) if not (c == "'" and letter_marker_context(query, pos))),
            "missing_apostrophe",
        )
        for source, typo in (
            ("qanday", "qande"),
            ("hujjat", "hujat"),
            ("qayer", "qayerd"),
            ("kerak", "kere"),
        ):
            if source in query:
                add(parent, query.replace(source, typo, 1), "typo")
                break
with (root / "cyrillic_variants.tsv").open(encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        add(parents[int(row["answer_id"])], row["query"], "cyrillic")
for n, query in enumerate((root / "no_match_queries.txt").read_text(encoding="utf-8").splitlines(), 1):
    cases.append(
        {
            "case_id": f"n{n:03}",
            "family_id": f"n{n:03}",
            "split": "calibration",
            "query": query,
            "acceptable_answer_ids": [],
            "expected_status": "no_match",
            "slice": "unrelated" if n <= 18 else "unsupported_action",
            "coverage_rationale": "No stored FAQ covers this unrelated request or performs the requested private lookup/action. A related how-to answer is insufficient.",
        }
    )
assert dict(counts) == {i: 8 for i in range(1, 10)}, counts
fresh_counts = defaultdict(int)
with (root / "fresh_test.tsv").open(encoding="utf-8", newline="") as stream:
    for n, row in enumerate(csv.DictReader(stream, delimiter="\t"), 1):
        expected = [int(i) for i in row["answer_ids"].split(",")]
        category = dataset.answers[expected[0]].category_id
        fresh_counts[category] += 1
        parent = {
            "case_id": f"t{n:03}",
            "family_id": f"t{n:03}",
            "split": "test",
            "query": row["query"],
            "category_id": category,
            "acceptable_answer_ids": expected,
            "expected_status": "matched",
            "slice": "base",
            "coverage_rationale": f"Fresh family authored from full source answer(s) {expected} before any v2 prediction; source coverage, not current legal verification.",
        }
        cases.append(parent)
        parents[expected[0]] = parent
        add(parent, row["query"].replace("'", "ʼ"), "typography")
        add(parent, row["query"].upper(), "casing")
        add(parent, "\ufeff \t" + row["query"].replace(" ", "\u00a0") + "  ", "whitespace")
        add(parent, row["query"].replace("'", ""), "missing_apostrophe")
        add(parent, row["query"].replace("a", "а", 1), "mixed_script")
with (root / "fresh_cyrillic.tsv").open(encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        add(parents[int(row["answer_id"])], row["query"], "cyrillic")
with (root / "fresh_no_match.tsv").open(encoding="utf-8", newline="") as stream:
    for n, row in enumerate(csv.DictReader(stream, delimiter="\t"), 1):
        cases.append(
            {
                "case_id": f"tn{n:03}",
                "family_id": f"tn{n:03}",
                "split": "test",
                "query": row["query"],
                "acceptable_answer_ids": [],
                "expected_status": "no_match",
                "slice": row["slice"],
                "coverage_rationale": "Fresh unsupported family; corpus cannot perform this action or answer unrelated request.",
            }
        )
assert dict(fresh_counts) == {i: 4 for i in range(1, 10)}, fresh_counts
validation = validate_cases(
    cases,
    {a.id for a in dataset.answers.values() if a.status == "active"},
    [q.question for q in dataset.indexed],
)
(root / "queries.jsonl").write_text(
    "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8"
)
print(json.dumps(validation))
