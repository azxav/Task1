# Reranker stage for uncertain matches

Status: approved design, 2026-10-13. Not yet implemented.

## Problem

The current system refuses about half of answerable questions: the fresh test showed 47.2% correct
answer coverage. On development data (72 answerable families), 37 answerable questions are
refused:

| Dense reason | Count | Correct answer at rank 1 | In top 5 | Not in top 5 |
|---|---:|---:|---:|---:|
| ambiguous | 13 | 7 | 6 | 0 |
| low_similarity | 24 | 15 | 7 | 2 |

Low-similarity refusals score 0.62–0.758, just under the 0.76 threshold. Must-refuse development
questions score 0.40–0.81; 11 of them fall in 0.62–0.76. Dense scores alone cannot separate these
cases. A cross-encoder reads the query and the stored text together and may separate them.

## Goal and success rule

Raise correct answer coverage without losing precision.

Adoption rule, evaluated on development data only:
- correct accepted answers increase compared with dense-only;
- accepted precision stays ≥ 95%;
- must-refuse false acceptance stays ≤ 5%.

If the rule fails, write `rerank_enabled: false`. Runtime behavior then stays identical to the
current system; the experiment code and report remain. Latency and RAM are measured and reported,
with no limit.

## Decision flow

```text
dense groups (top 30 stored questions → grouped by answer, as now)
  s1 ≥ 0.76 and margin ≥ 0.04        → matched                 stage "dense"
  s1 < X                             → no_match low_similarity stage "dense"
  otherwise (ambiguous, or X ≤ s1 < 0.76)
     → rerank the top 5 distinct answer groups
       r1 ≥ R_score and r1 − r2 ≥ R_margin → matched           stage "rerank"
       r1 < R_score                        → no_match rerank_low_score
       r1 − r2 < R_margin                  → no_match rerank_ambiguous
```

- Dense thresholds 0.76 / 0.04 stay fixed. The confident path does not change.
- 0.76 is the "accept without reranker" line. X is the new reject line.
- Pair per group: `(user query, best stored question + "\n" + best answer text)`. "Best" means
  the group's highest dense-scoring question and its answer, as `group_answer_candidates` selects.
- Reranker score = sigmoid of the cross-encoder logit, from 0 to 1. It is not a probability.
- With one group: score check only, no margin check (the same as dense `decide`).
- The final answer is the reranked best group's `best_answer_id`, fetched verbatim as now.
- With `rerank_enabled: false`, X is ignored; `decide()` behaves exactly as today.

## Model

- `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- Loaded with sentence-transformers `CrossEncoder` (already a dependency), CPU, `max_length=512`.
- About 568M parameters, about 2.2 GB RAM in fp32. Together with BGE-M3, the API needs about 5 GB.
- Weights are cached under `HF_HOME`, so Docker uses the existing `modelcache` volume.

## Calibration

The values to calibrate are X, R_score and R_margin. Everything else is frozen.

1. Run the current dense evaluation unchanged. It selects the profile and the 0.76 / 0.04
   thresholds as now.
2. For every development and test case not accepted by the confident dense path and with
   s1 ≥ 0.50, compute reranker scores for its top 5 groups once. Cache them on disk, keyed by
   model, revision and exact pair text.
3. Grid search on development base and must-refuse families only (variants excluded, as now):
   - X: 0.50–0.76, step 0.01;
   - R_score: 0.00–0.99, step 0.01;
   - R_margin: 0.00–0.50, step 0.01.
   Metric: combined dense + rerank decision for each row.
4. Choose the candidate with the existing objective: feasible (precision ≥ 95%, false
   acceptance ≤ 5%) first, then utility `correct − 8 × wrong`, then correct accepted. Ties prefer a
   higher X (fewer reranks), then a higher R_score, then a higher R_margin.
5. Apply the adoption rule. Write the result to `evaluation/uzbek/manifest.json`.
6. Only then compute results on the test split.

An assertion confirms that no test-split row enters steps 3–5.

Risk: three new values on 108 development families can overfit. The report states this.

## Test set status

The existing test split has already been used once, and its result motivated this change. It is
reused for this report and labeled **"reused, not fresh"** in the manifest
(`test_status: "reused"`) and in every report. No value is tuned on it. A fresh held-out set is
still needed before any claim of generalization.

## Components

| File | Change |
|---|---|
| `src/faq/rerank.py` (new) | `Reranker`: loads the pinned model, checks pair length, `score(query, texts) -> list[float]` (sigmoid). Optional disk cache for evaluation. |
| `src/faq/search/answer_groups.py` | `route(groups, *, min_score, min_margin, floor) -> "matched" / "rerank" / "no_match"`; `decide_reranked(groups, scores, *, min_score, min_margin) -> Decision`. `decide()` is unchanged. |
| `src/faq/storage.py` | `group_texts(answer_ids, question_ids)`: one query returns the answer and question texts. |
| `src/faq/service.py` | Rerank branch inside the existing shared snapshot lock; separate reranker lock. Response adds `stage`, `rerank_score`, `rerank_margin`. |
| `src/faq/config.py` | New settings with defaults from the manifest: `rerank_enabled`, `reranker_name`, `reranker_revision`, `rerank_groups` (5), `rerank_floor`, `rerank_min_score`, `rerank_min_margin`. Env overrides: `FAQ_RERANK_ENABLED`, `RERANKER_NAME`, `RERANKER_REVISION`, `FAQ_RERANK_FLOOR`, `FAQ_RERANK_MIN_SCORE`, `FAQ_RERANK_MIN_MARGIN`. The calibration guard compares all of them. |
| `src/faq/api.py` | Loads the reranker at startup only when enabled. Endpoints are unchanged. |
| `src/faq/evaluation/runner.py`, `metrics.py` | Calibration steps 2–6, adoption rule, report section, warm latency of the rerank path, process RAM after loading both models. |
| `scripts/verify_live_api.py` | Compares live decisions with frozen offline predictions, including the rerank path; reports HTTP p50/p95 for reranked queries. |
| `pyproject.toml` | `psutil` in dev extras, for RAM measurement only. |
| Docs | README quality table; GUIDE §7, §9, §12–14, §17; `docs/evaluation/uzbek-normalization.md`. |

`bot.py`, import, the Qdrant index, `compose.yaml` and `Dockerfile` do not change. The reranker
works only at query time, so no re-import is needed.

## Error handling

| Case | Behavior |
|---|---|
| Reranker enabled, but the model fails to load | API startup fails with a clear error. |
| Reranker raises during search | 503 "temporarily unavailable"; never a silent no_match. |
| Score is NaN or infinite | Error → 503. |
| Pair longer than 512 tokens | No truncation. Skip reranking; return the dense decision (no_match); log `rerank_skipped_length`. |
| Fewer than 5 groups | Rerank the groups that exist. |
| Reranker settings differ from the manifest | Calibration guard: `/ready` 503, import refuses. |
| Concurrency | Texts are fetched under the same shared snapshot lock (one generation). Slower queries can cause more "busy" 503s under load; the admission limit stays 2. |

## Tests

- `route`: s1 = 0.76 with margin = 0.04 → matched; ambiguous above 0.76 → rerank; s1 = X → rerank;
  s1 just below X → no_match; disabled → same output as `decide()`.
- `decide_reranked`: score and margin boundaries, one group, deterministic ties, non-finite score.
- Service with a fake reranker: verbatim answer and `stage: "rerank"`; reranker exception → 503;
  long pair → dense no_match; disabled → current behavior unchanged.
- Calibration guard: a changed reranker revision or threshold is refused.
- Calibration: selection receives development rows only (assertion); "reject everything" is never
  chosen; a failed adoption rule writes `rerank_enabled: false`.
- All current tests stay green. Unit tests use a fake reranker; the real model is covered by
  `faq evaluate` and `scripts/verify_live_api.py`.

## Out of scope

Recalibrating 0.76 / 0.04, hybrid sparse retrieval, transliteration, upper-case handling, GPU
tuning, and a new fresh test set.
