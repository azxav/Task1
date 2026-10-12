"""Read-only source audit; writes analysis JSON, not normalized source data.

Standard-library only. Requires tar with RAR support. The diagnostic key
is an exploratory collision probe, NOT the proposed production normalizer.
"""

import argparse
import collections
import csv
import difflib
import hashlib
import io
import itertools
import json
import re
import subprocess
import sys
import unicodedata as ud
from pathlib import Path

DIAGNOSTIC_MARKS = "'\u2018\u2019\u02bb\u02bc`\u00b4\u02b9\uff07\u2032\u201b\u02bd"


def load(archive, member):
    raw = subprocess.check_output(["tar", "-xOf", str(archive), member])
    return raw, list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def diagnostic_key(text):
    text = ud.normalize("NFC", text).casefold()
    for mark in DIAGNOSTIC_MARKS:
        text = re.sub(r"(?<=[^\W\d_])" + re.escape(mark) + r"(?=[^\W\d_])", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze(rows, field):
    counts = collections.Counter("".join(row[field] for row in rows))
    characters = []
    for char, count in sorted(counts.items(), key=lambda item: ord(item[0])):
        if (
            char in DIAGNOSTIC_MARKS
            or (ord(char) > 127 and not ud.category(char).startswith("L"))
            or ud.category(char).startswith("C")
        ):
            samples = []
            for row in rows:
                if char in row[field]:
                    pos = row[field].index(char)
                    samples.append({"id": row["id"], "context": row[field][max(0, pos - 20) : pos + 25]})
                    if len(samples) == 3:
                        break
            characters.append(
                {
                    "char": char,
                    "code": f"U+{ord(char):04X}",
                    "name": ud.name(char, "UNNAMED"),
                    "count": count,
                    "rows": sum(char in row[field] for row in rows),
                    "samples": samples,
                }
            )
    scripts = collections.Counter()
    mixed = []
    groups = collections.defaultdict(list)
    for row in rows:
        latin = any("LATIN" in ud.name(char, "") for char in row[field])
        cyrillic = any("CYRILLIC" in ud.name(char, "") for char in row[field])
        scripts[
            "mixed" if latin and cyrillic else "cyrillic" if cyrillic else "latin" if latin else "other"
        ] += 1
        if cyrillic:
            mixed.append({"id": row["id"], "text": row[field]})
        groups[diagnostic_key(row[field])].append(row)
    duplicates, collisions = [], []
    for text, group in groups.items():
        if len(group) <= 1:
            continue
        item = {"normalized": text, "ids": [row["id"] for row in group]}
        if field == "question":
            item["answer_ids"] = sorted({row["answer_id"] for row in group}, key=int)
        (collisions if field == "question" and len(item["answer_ids"]) > 1 else duplicates).append(item)
    formatting = {
        "leading_trailing": sum(row[field] != row[field].strip() for row in rows),
        "repeated_whitespace": sum(bool(re.search(r"\s{2,}", row[field])) for row in rows),
        "casefold_changes": sum(row[field] != row[field].casefold() for row in rows),
        "nfc_changes": sum(row[field] != ud.normalize("NFC", row[field]) for row in rows),
        "nfkc_changes": sum(row[field] != ud.normalize("NFKC", row[field]) for row in rows),
        "rows_with_digits": sum(bool(re.search(r"\d", row[field])) for row in rows),
        "max_length": max(len(row[field]) for row in rows),
    }
    marker_contexts = collections.Counter()
    context_examples = collections.defaultdict(list)
    for row in rows:
        text = row[field]
        for pos, char in enumerate(text):
            if char != "'":
                continue
            before = text[pos - 1] if pos else ""
            after = text[pos + 1] if pos + 1 < len(text) else ""
            role = (
                "between_letters"
                if before.isalpha() and after.isalpha()
                else "after_o_g"
                if before and before in "oOgG"
                else "other"
            )
            marker_contexts[role] += 1
            if len(context_examples[role]) < 3:
                context_examples[role].append({"id": row["id"], "text": text})
    return {
        "count": len(rows),
        "characters": characters,
        "scripts": dict(scripts),
        "non_latin_samples": mixed[:12],
        "formatting": formatting,
        "apostrophe_contexts": dict(marker_contexts),
        "apostrophe_context_examples": dict(context_examples),
        "cross_answer_collisions": collisions,
        "duplicate_groups": duplicates,
    }


def semantic_audit(questions, answers, groups):
    report = {
        "extra_variants": {key: value for key, value in groups.items() if len(value) != 6},
        "categories": dict(collections.Counter(a["category_id"] for a in answers)),
        "topics": dict(collections.Counter(a["topic_id"] for a in answers)),
        "numeric_questions": [q for q in questions if re.search(r"\d", q["question"])],
    }
    answer_pairs = []
    for a, b in itertools.combinations(answers, 2):
        matcher = difflib.SequenceMatcher(
            None, a["answer"].casefold(), b["answer"].casefold(), autojunk=False
        )
        if matcher.quick_ratio() < 0.88:
            continue
        score = matcher.ratio()
        if score >= 0.88:
            answer_pairs.append(
                {
                    "ratio": round(score, 4),
                    "a": a["id"],
                    "b": b["id"],
                    "texts": [a["answer"], b["answer"]],
                    "questions": [groups[a["id"]][0]["question"], groups[b["id"]][0]["question"]],
                }
            )
    report["similar_answers"] = sorted(answer_pairs, key=lambda p: (-p["ratio"], int(p["a"]), int(p["b"])))[
        :20
    ]
    trigrams, inverted = [], collections.defaultdict(list)
    for index, question in enumerate(questions):
        text = question["question"].casefold()
        grams = {text[pos : pos + 3] for pos in range(len(text) - 2)}
        trigrams.append(grams)
        for gram in grams:
            inverted[gram].append(index)
    pairs = []
    for index, question in enumerate(questions):
        intersections = collections.Counter(
            other
            for gram in trigrams[index]
            for other in inverted[gram]
            if other > index and questions[other]["answer_id"] != question["answer_id"]
        )
        for other, intersection in intersections.items():
            score = intersection / (len(trigrams[index]) + len(trigrams[other]) - intersection)
            if score >= 0.82:
                neighbor = questions[other]
                pairs.append(
                    {
                        "jaccard": round(score, 4),
                        "question_ids": [question["id"], neighbor["id"]],
                        "answer_ids": [question["answer_id"], neighbor["answer_id"]],
                        "texts": [question["question"], neighbor["question"]],
                    }
                )
    report["similar_questions_cross_answer"] = sorted(
        pairs, key=lambda p: (-p["jaccard"], *(int(value) for value in p["question_ids"]))
    )[:30]
    report["similar_question_pair_count_over_082"] = len(pairs)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/research"))
    args = parser.parse_args()
    qb, questions = load(args.archive, "Files Task_1/questions.csv")
    ab, answers = load(args.archive, "Files Task_1/answers.csv")
    groups = collections.defaultdict(list)
    for question in questions:
        groups[question["answer_id"]].append(question)
    lookup = {answer["id"]: answer for answer in answers}
    audit = {"questions": analyze(questions, "question"), "answers": analyze(answers, "answer")}
    for collision in audit["questions"]["cross_answer_collisions"]:
        collision["answers"] = {key: lookup[key]["answer"] for key in collision["answer_ids"]}
    patterns = {
        "malformed_ma_lumotnomanini": r"ma'lumotnomanini",
        "ma_lumot_bering": r"haqida ma'lumot bering",
        "ma_lumot_kerak": r"bo'yicha ma'lumot kerak",
        "qanaqa": r"\bqanaqa\b",
        "id_karta": r"(?i)ID-karta",
        "my_gov": r"my\.gov\.uz",
        "latin_x_vs_h": r"(?i)xizmat|hujjat",
        "negative_or_missing": r"(?i)emas|bo'lmay|yo'q|olmay|rad et|bekor",
        "acronyms": r"\b[A-Z][A-Z0-9]{1,}\b",
    }
    targeted = {
        "sha256": {
            "questions.csv": hashlib.sha256(qb).hexdigest(),
            "answers.csv": hashlib.sha256(ab).hexdigest(),
        },
        "selected_answers": [
            lookup[key] for key in ["22", "27", "56", "144", "181", "279", "380", "494", "499"]
        ],
        "question_patterns": {},
        "samples": {},
        "source_dates": dict(collections.Counter(q["created_at"] for q in questions)),
    }
    targeted["audit_environment"] = {
        "python": sys.version.split()[0],
        "unicode_version": ud.unidata_version,
        "answer_similarity": "SequenceMatcher >= 0.88",
        "question_similarity": "character-trigram Jaccard >= 0.82",
    }
    for name, pattern in patterns.items():
        matching = [q for q in questions if re.search(pattern, q["question"])]
        targeted["question_patterns"][name] = len(matching)
        targeted["samples"][name] = matching[:4]
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, data in [
        ("task1-data-audit.json", audit),
        ("task1-semantic-audit.json", semantic_audit(questions, answers, groups)),
        ("task1-targeted-audit.json", targeted),
    ]:
        (args.output / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "questions": len(questions),
                "answers": len(answers),
                "unique_question_ids": len({q["id"] for q in questions}),
                "unique_answer_ids": len(lookup),
                "orphan_questions": sum(q["answer_id"] not in lookup for q in questions),
                "question_states": dict(collections.Counter(q["is_active"] for q in questions)),
                "answer_states": dict(collections.Counter(a["status"] for a in answers)),
                "output": str(args.output),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
