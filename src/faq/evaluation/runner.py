"""Offline exact cosine evaluation, plus deployed-Qdrant parity checks.

Profile/threshold selection uses calibration only. Test metrics are computed
after the selected configuration is frozen on disk. No generations are published here.
"""

import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ..data import load_dataset
from ..embeddings import Encoder
from ..index import VectorIndex
from ..search.answer_groups import EquivalenceRegistry, ScoredQuestion, group_answer_candidates
from ..storage import Storage
from .labels import validate_cases
from .metrics import apply_threshold, calibration_families, metrics, select_thresholds
from .rerank_calibration import adopt, apply_rerank, needs_rerank, select_rerank
from .text_profiles import prepare_profile


def rank_queries(query_vectors, document_vectors, questions, registry, *, limit=30, groups_limit=5):
    scores = query_vectors @ document_vectors.T
    output = []
    for row in scores:
        order = np.argsort(-row, kind="stable")[:limit]
        candidates = [
            ScoredQuestion(questions[int(i)].id, questions[int(i)].answer_id, float(row[i])) for i in order
        ]
        groups = group_answer_candidates(candidates, registry)
        output.append(
            {
                "ranked": [g.best_answer_id for g in groups],
                "score": groups[0].best_score if groups else -1,
                "margin": groups[0].best_score - groups[1].best_score if len(groups) > 1 else None,
                "top_groups": [
                    {"answer_id": g.best_answer_id, "question_id": g.question_id, "score": g.best_score}
                    for g in groups[:groups_limit]
                ],
            }
        )
    return output


def _labeled(ranks, cases):
    return [
        {
            **row,
            "case_id": c["case_id"],
            "family_id": c["family_id"],
            "split": c["split"],
            "slice": c["slice"],
            "expected": c["acceptable_answer_ids"],
            "predicted": None,
        }
        for row, c in zip(ranks, cases, strict=True)
    ]


def _summary(rows):
    slices = defaultdict(list)
    for row in rows:
        slices[row["slice"]].append(row)
    primary = [{**row, "expected": row["expected"][:1]} for row in rows]
    return {
        "all": metrics(rows),
        "base_and_no_match": metrics(
            [r for r in rows if r["slice"] in ("base", "unrelated", "unsupported_action")]
        ),
        "primary_id": metrics(primary),
        "slices": {name: metrics(group) for name, group in slices.items()},
    }


def _rerank_stage(
    settings, selected, rows, cases, frozen, *, encoder, document_vectors, questions, source, registry
):
    """Second stage for the selected profile; dense values in `frozen` are never changed here."""
    from ..rerank import PairTooLong, Reranker

    reranker = Reranker(settings)
    texts = {q.id: q.question + "\n" + source.answers[q.answer_id].answer for q in questions}
    dense = {"min_score": frozen["min_score"], "min_margin": frozen["min_margin"]}
    reranked = skipped = 0
    for row, case in zip(rows, cases, strict=True):
        if not needs_rerank(row, **dense):
            continue
        query = prepare_profile(case["query"], selected)
        try:
            row["rerank_scores"] = reranker.score(
                query, [texts[g["question_id"]] for g in row["top_groups"]], cache=True
            )
            reranked += 1
        except PairTooLong:
            row["rerank_scores"] = None
            skipped += 1
    development = calibration_families(rows)
    choice = select_rerank(development, **dense)
    dense_development = metrics(development)
    adopted = adopt(dense_development, choice["calibration"])
    values = {key: choice[key] for key in ("rerank_floor", "rerank_min_score", "rerank_min_margin")}
    combined = [
        apply_rerank(
            row,
            **dense,
            floor=values["rerank_floor"],
            r_score=values["rerank_min_score"],
            r_margin=values["rerank_min_margin"],
        )
        for row in rows
    ]
    latencies = []
    for row, case in [(r, c) for r, c in zip(rows, cases, strict=True) if r.get("rerank_scores")][:12]:
        started = perf_counter()
        query = prepare_profile(case["query"], selected)
        ranked = rank_queries(
            encoder.encode([query]),
            document_vectors,
            questions,
            registry,
            limit=settings.top_k,
            groups_limit=settings.rerank_groups,
        )[0]
        reranker.score(query, [texts[g["question_id"]] for g in ranked["top_groups"]])
        latencies.append(1000 * (perf_counter() - started))
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 2**30
    except ImportError:
        rss = None
    return {
        "rows": combined,
        "manifest": {
            "rerank_enabled": adopted,
            "reranker_name": settings.reranker_name,
            "reranker_revision": settings.reranker_revision,
            "rerank_groups": settings.rerank_groups,
            **values,
            "test_status": "reused",
            "rerank_selection_basis": (
                "Dense 0.76/0.04 frozen. Floor, reranker score and margin chosen on 108 development "
                "families only; adopt only if correct accepted rises with >=95% precision and <=5% "
                "false acceptance. Test split reused, not fresh."
            ),
        },
        "report": {
            "model": settings.reranker_name,
            "revision": settings.reranker_revision,
            "adopted": adopted,
            "test_status": "reused, not fresh",
            "selection": choice,
            "dense_development": dense_development,
            "reranked_rows": reranked,
            "skipped_length": skipped,
            "warm_rerank_path_ms": {
                "samples": len(latencies),
                "p50": float(np.percentile(latencies, 50)) if latencies else None,
                "p95": float(np.percentile(latencies, 95)) if latencies else None,
            },
            "reranker_startup_seconds": reranker.startup_seconds,
            "process_rss_gib": rss,
        },
    }


def run_evaluation(settings):
    source = load_dataset("data/source")
    root, out = Path("evaluation/uzbek"), Path("evaluation/results")
    out.mkdir(parents=True, exist_ok=True)
    case_path = root / "queries.jsonl"
    cases = [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines()]
    questions = source.indexed
    validate_cases(
        cases,
        {a.id for a in source.answers.values() if a.status == "active"},
        [q.question for q in questions],
    )
    registry = EquivalenceRegistry.load(settings.registry_path, source.answers)
    encoder = Encoder(settings)
    profiles, prepared_indexes, query_vectors = {}, {}, {}
    for profile in ("raw", "uz_surface_v1", "uz_surface_casefold_v1"):
        print(f"Evaluating calibration profile: {profile}", flush=True)
        indexed = encoder.encode([prepare_profile(q.question, profile) for q in questions], cache=True)
        queries = encoder.encode([prepare_profile(c["query"], profile) for c in cases], cache=True)
        rows = _labeled(
            rank_queries(
                queries,
                indexed,
                questions,
                registry,
                limit=settings.top_k,
                groups_limit=settings.rerank_groups,
            ),
            cases,
        )
        calibration = calibration_families(rows)
        thresholds = select_thresholds(calibration)
        transformed = [
            apply_threshold(row, thresholds["min_score"], thresholds["min_margin"]) for row in rows
        ]
        profiles[profile] = {"thresholds": thresholds, "rows": transformed}
        prepared_indexes[profile], query_vectors[profile] = indexed, queries
    # Text transformations and all thresholds are chosen solely against calibration labels.
    preference = {"uz_surface_v1": 2, "uz_surface_casefold_v1": 1, "raw": 0}
    selected = max(
        profiles,
        key=lambda p: (
            profiles[p]["thresholds"]["quality_target_met"],
            profiles[p]["thresholds"]["calibration"]["correct_accepted"],
            preference[p],
        ),
    )
    frozen = {
        "profile": selected,
        "min_score": profiles[selected]["thresholds"]["min_score"],
        "min_margin": profiles[selected]["thresholds"]["min_margin"],
        "top_k": settings.top_k,
        "model_name": settings.model_name,
        "model_revision": settings.model_revision,
        "registry_digest": registry.digest,
        "corpus_hashes": source.hashes,
        "corpus_digest": source.canonical_digest,
        "cases_sha256": hashlib.sha256(case_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        "cases_hash_basis": "UTF-8 text with LF line endings",
        "selection_basis": "Calibration only; >=95% accepted precision and <=5% no-match false acceptance when feasible; maximize correct accepted, prefer conservative surface on ties",
        "device": settings.device,
        "max_tokens": 512,
    }
    frozen["benchmark_version"] = "source_coverage_v2"
    frozen["selection_basis"] = (
        "108 independent development families only (variants excluded); >=95% accepted precision and <=5% no-match false acceptance when feasible; maximize correct accepted, prefer conservative surface on ties. Fresh test uninspected before freezing."
    )
    rerank = _rerank_stage(
        settings,
        selected,
        profiles[selected]["rows"],
        cases,
        frozen,
        encoder=encoder,
        document_vectors=prepared_indexes[selected],
        questions=questions,
        source=source,
        registry=registry,
    )
    frozen.update(rerank["manifest"])
    (root / "manifest.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    report = {
        "selected": frozen,
        "profiles": {},
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": settings.device,
        },
        "model_startup_seconds": encoder.startup_seconds,
        "independent_families": len({c["family_id"] for c in cases}),
        "cases": len(cases),
        "warnings": source.warnings,
    }
    for profile, data in profiles.items():
        rows = data["rows"]
        report["profiles"][profile] = {
            "thresholds": data["thresholds"],
            "calibration": _summary([r for r in rows if r["split"] == "calibration"]),
            "test": _summary([r for r in rows if r["split"] == "test"]),
        }
        (out / f"{profile}-predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    (out / "rerank-predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rerank["rows"]), encoding="utf-8"
    )
    report["rerank"] = {
        **rerank["report"],
        "development": _summary([r for r in rerank["rows"] if r["split"] == "calibration"]),
        "test": _summary([r for r in rerank["rows"] if r["split"] == "test"]),
    }
    # Character lexical baseline: retrieval metrics only; no uncalibrated rejection claim.
    from sklearn.feature_extraction.text import TfidfVectorizer

    from ..text.normalization import normalize

    lexical = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False)
    index_matrix = lexical.fit_transform([normalize(q.question).comparison_key for q in questions])
    query_matrix = lexical.transform([normalize(c["query"]).comparison_key for c in cases])
    lexical_scores = (query_matrix @ index_matrix.T).toarray()
    lexical_rows = []
    for score_row, case in zip(lexical_scores, cases, strict=True):
        order = np.argsort(-score_row, kind="stable")[: settings.top_k]
        groups = group_answer_candidates(
            [ScoredQuestion(questions[i].id, questions[i].answer_id, float(score_row[i])) for i in order],
            registry,
        )
        lexical_rows.append(
            {
                "expected": case["acceptable_answer_ids"],
                "ranked": [g.best_answer_id for g in groups],
                "predicted": None,
                "split": case["split"],
                "slice": case["slice"],
            }
        )
    report["lexical_baseline"] = _summary([r for r in lexical_rows if r["split"] == "test"])
    # Separate single-variant diagnostic; siblings stay indexed. It is intentionally optimistic.
    first = {}
    for pos, question in enumerate(questions):
        first.setdefault(question.answer_id, pos)
    held_out = sorted(first.values())
    retained = [i for i in range(len(questions)) if i not in set(held_out)]
    diagnostic = rank_queries(
        prepared_indexes[selected][held_out],
        prepared_indexes[selected][retained],
        [questions[i] for i in retained],
        registry,
        limit=settings.top_k,
    )
    diagnostic_rows = []
    for result, pos in zip(diagnostic, held_out, strict=True):
        ident = questions[pos].answer_id
        acceptable = next((list(g) for g in registry.classes if ident in g), [ident])
        diagnostic_rows.append({**result, "expected": acceptable, "predicted": None})
    report["supplied_variant_holdout"] = metrics(diagnostic_rows)
    # Actual single-query model + exact retrieval warm latency, not batch throughput.
    latencies = []
    for case in [c for c in cases if c["slice"] == "base" and c["split"] == "test"][:12]:
        started = perf_counter()
        vector = encoder.encode([prepare_profile(case["query"], selected)])
        rank_queries(vector, prepared_indexes[selected], questions, registry, limit=settings.top_k)
        latencies.append(1000 * (perf_counter() - started))
    report["warm_model_search_ms"] = {
        "samples": len(latencies),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    }
    # Preservation diagnostics: distinct token sequences are necessary, but high
    # cosine still does not establish that the model understands a negated intent.
    contrast_pairs = [
        ("Haydovchilik uchun 16 yosh", "Haydovchilik uchun 6 yosh"),
        ("Nikoh uchun 18 yosh", "Nikoh uchun 17 yosh"),
        ("Ro'yxatdan o'tish mumkin", "Ro'yxatdan o'tish mumkin emas"),
        ("To'lov qilinadi", "To'lov qilinmaydi"),
        ("she'r", "sher"),
    ]
    report["preservation_diagnostics"] = []
    for left, right in contrast_pairs:
        texts = [prepare_profile(t, selected) for t in (left, right)]
        tokens = [encoder.model.tokenizer.encode(t) for t in texts]
        vectors = encoder.encode(texts, cache=True)
        report["preservation_diagnostics"].append(
            {
                "left": left,
                "right": right,
                "prepared_distinct": texts[0] != texts[1],
                "tokens_distinct": tokens[0] != tokens[1],
                "cosine": float(vectors[0] @ vectors[1]),
                "scope": "Diagnostic only; no expected answer or correctness claim",
            }
        )
    # Compare offline cosine with real Qdrant for the currently published profile.
    store, index = Storage(settings.database_url), VectorIndex(settings.qdrant_url)
    meta = store.metadata()
    parity = []
    if meta.get("profile") in profiles:
        profile = meta["profile"]
        for pos in [i for i, c in enumerate(cases) if c["slice"] == "base"][:9]:
            hits = index.search(
                query_vectors[profile][pos], expected_generation=meta["generation"], limit=settings.top_k
            )
            groups = group_answer_candidates(hits, registry)
            offline = profiles[profile]["rows"][pos]
            parity.append(
                {
                    "case_id": cases[pos]["case_id"],
                    "same_best_answer": groups[0].best_answer_id == offline["ranked"][0],
                    "score_delta": abs(groups[0].best_score - offline["score"]),
                }
            )
    report["qdrant_parity"] = parity
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_report(report)
    return {
        "profile": selected,
        "thresholds": {k: frozen[k] for k in ("min_score", "min_margin")},
        "test": report["profiles"][selected]["test"]["base_and_no_match"],
        "rerank_adopted": report["rerank"]["adopted"],
        "report": str(out / "report.json"),
    }


def _write_report(report):
    lines = [
        "# Uzbek FAQ measured quality",
        "",
        "Measured with pinned BGE-M3 on local CPU; no generative model or external inference.",
        "",
        (
            "Version 2: 108 development families (72 answerable, 36 no-match), including the inspected v1 test. "
            "Fresh held-out test: 54 families (36 answerable, four/category; 18 no-match). Synthetic variants stay with their family and do not influence threshold selection. "
            f"{report['cases']} total cases; configuration frozen before fresh test metrics. Labels are source-coverage judgments, not an external adjudication. Original flawed labels/results remain in evaluation/results/v1."
        ),
        "",
        "| Profile | Test top-1 | Accepted precision | Answer coverage | No-match false acceptance |",
        "|---|---:|---:|---:|---:|",
    ]

    def percent(value):
        return "n/a" if value is None else f"{100 * value:.1f}%"

    for name, data in report["profiles"].items():
        m = data["test"]["base_and_no_match"]
        lines.append(
            f"| {name} | {percent(m['top1_accuracy'])} | {percent(m['accepted_precision'])} | {percent(m['answer_coverage'])} | {percent(m['false_acceptance'])} |"
        )
    selected = report["selected"]
    final = report["profiles"][selected["profile"]]["test"]["base_and_no_match"]
    lines += [
        "",
        f"Selected on calibration: **{selected['profile']}**, minimum score **{selected['min_score']}**, ambiguity margin **{selected['min_margin']}**.",
        f"Fresh base test accepts {final['correct_accepted']} correct answers and {final['wrong_accepted']} wrong answers, rejects {final['answerable'] - final['correct_accepted']} of {final['answerable']} answerable questions, and falsely accepts {final['false_accepted']} of {final['no_match']} unsupported requests. The 95% precision / 5% false-acceptance development targets were not achieved on this small held-out test. This is an evaluated prototype, not evidence of production readiness.",
        "",
        "Full counts, per-slice metrics, primary-ID metrics, calibration figures, source-variant holdout and Qdrant parity are in evaluation/results/report.json. Per-case predictions are beside it.",
        "",
        (
            f"Warm CPU model plus exact vector search: p50 {report['warm_model_search_ms']['p50']:.1f} ms; p95 {report['warm_model_search_ms']['p95']:.1f} ms over 12 single-query samples. This excludes HTTP/SQL overhead. "
            f"Model startup: {report['model_startup_seconds']:.1f} seconds with cached weights; initial download/import is separate."
        ),
        "",
        "## Limits and next steps",
        "",
        (
            "Small balanced benchmark is not production traffic. Variants are correlated within families and cannot be counted as independent evidence. Primary-ID results can penalize valid alternative answers; semantic labels and equivalence groups are reported separately. "
            "The variant holdout retains closely related sibling templates and is optimistic. Corpus statements/fees have not been checked for legal currency. Source answer 499 retains a Cyrillic lookalike by design."
        ),
        "",
        (
            "A similarity score is not a probability. Closely related unsupported private lookups/actions can still retrieve a plausible FAQ; rejection rates must be read alongside answer coverage. "
            "No automatic typo correction, stemming or transliteration has been added. Numeric/negation token and cosine diagnostics appear in report.json; preservation does not prove semantic correctness. Answer-vector reranking was tested on development data and not selected; see answer-support-experiment.json. Any further tuning needs another fresh held-out set."
        ),
        "",
    ]
    rr = report["rerank"]

    def line(name, m):
        return (
            f"| {name} | {percent(m['accepted_precision'])} | {percent(m['correct_answer_coverage'])} | "
            f"{percent(m['false_acceptance'])} | {m['correct_accepted']} | {m['wrong_accepted']} |"
        )

    sel, warm = rr["selection"], rr["warm_rerank_path_ms"]
    rss = "not measured" if rr["process_rss_gib"] is None else f"{rr['process_rss_gib']:.2f} GiB"
    latency = "n/a" if warm["p50"] is None else f"p50 {warm['p50']:.1f} ms, p95 {warm['p95']:.1f} ms"
    lines += [
        "## Reranker stage",
        "",
        (
            f"Model {rr['model']} (revision {rr['revision'][:12]}), local CPU. Uncertain dense matches "
            f"(ambiguous, or dense score from the floor up to {selected['min_score']}) are reranked over the top "
            f"{selected['rerank_groups']} answer groups. Chosen on development families only: floor "
            f"**{sel['rerank_floor']}**, reranker score **{sel['rerank_min_score']}**, reranker margin "
            f"**{sel['rerank_min_margin']}**. Adopted for runtime: **{'yes' if rr['adopted'] else 'no'}**."
        ),
        "",
        (
            "**The test split is reused, not fresh.** It was measured once before this change and motivated it. "
            "No value was tuned on it, but it is not independent evidence; a fresh held-out set is still required."
        ),
        "",
        "| Data | Accepted precision | Correct coverage | False acceptance | Correct | Wrong |",
        "|---|---:|---:|---:|---:|---:|",
        line("Development, dense only", rr["dense_development"]),
        line("Development, dense + rerank", rr["development"]["base_and_no_match"]),
        line("Test (reused), dense only", final),
        line("Test (reused), dense + rerank", rr["test"]["base_and_no_match"]),
        "",
        (
            f"Reranked cases: {rr['reranked_rows']} of {report['cases']}; skipped for length: "
            f"{rr['skipped_length']}. Warm rerank path (query embedding, exact search, reranking): {latency} over "
            f"{warm['samples']} samples. Reranker startup {rr['reranker_startup_seconds']:.1f} s. Evaluation "
            f"process memory with both models loaded: {rss}. Three values on 108 development families can overfit."
        ),
        "",
    ]
    destination = Path("docs/evaluation/uzbek-normalization.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
