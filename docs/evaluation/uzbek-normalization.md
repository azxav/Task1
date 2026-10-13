# Uzbek FAQ measured quality

Measured with pinned BGE-M3 on local CPU; no generative model or external inference.

Version 2: 108 development families (72 answerable, 36 no-match), including the inspected v1 test. Fresh held-out test: 54 families (36 answerable, four/category; 18 no-match). Synthetic variants stay with their family and do not influence threshold selection. 625 total cases; configuration frozen before fresh test metrics. Labels are source-coverage judgments, not an external adjudication. Original flawed labels/results remain in evaluation/results/v1.

| Profile | Test top-1 | Accepted precision | Answer coverage | No-match false acceptance |
|---|---:|---:|---:|---:|
| raw | 75.0% | 94.4% | 47.2% | 5.6% |
| uz_surface_v1 | 75.0% | 94.4% | 47.2% | 5.6% |
| uz_surface_casefold_v1 | 75.0% | 90.5% | 55.6% | 5.6% |

Selected on calibration: **uz_surface_v1**, minimum score **0.76**, ambiguity margin **0.04**.
Fresh base test accepts 17 correct answers and 1 wrong answers, rejects 19 of 36 answerable questions, and falsely accepts 1 of 18 unsupported requests. The 95% precision / 5% false-acceptance development targets were not achieved on this small held-out test. This is an evaluated prototype, not evidence of production readiness.

Full counts, per-slice metrics, primary-ID metrics, calibration figures, source-variant holdout and Qdrant parity are in evaluation/results/report.json. Per-case predictions are beside it.

Warm CPU model plus exact vector search: p50 115.7 ms; p95 124.1 ms over 12 single-query samples. This excludes HTTP/SQL overhead. Model startup: 6.9 seconds with cached weights; initial download/import is separate.

## Limits and next steps

Small balanced benchmark is not production traffic. Variants are correlated within families and cannot be counted as independent evidence. Primary-ID results can penalize valid alternative answers; semantic labels and equivalence groups are reported separately. The variant holdout retains closely related sibling templates and is optimistic. Corpus statements/fees have not been checked for legal currency. Source answer 499 retains a Cyrillic lookalike by design.

A similarity score is not a probability. Closely related unsupported private lookups/actions can still retrieve a plausible FAQ; rejection rates must be read alongside answer coverage. No automatic typo correction, stemming or transliteration has been added. Numeric/negation token and cosine diagnostics appear in report.json; preservation does not prove semantic correctness. Answer-vector reranking was tested on development data and not selected; see answer-support-experiment.json. Any further tuning needs another fresh held-out set.

## Reranker stage

Model BAAI/bge-reranker-v2-m3 (revision 953dc6f6f85a), local CPU. Uncertain dense matches (ambiguous, or dense score from the floor up to 0.76) are reranked over the top 5 answer groups. Chosen on development families only: floor **0.62**, reranker score **0.75**, reranker margin **0.22**. Adopted for runtime: **yes**.

**The test split is reused, not fresh.** It was measured once before this change and motivated it. No value was tuned on it, but it is not independent evidence; a fresh held-out set is still required.

| Data | Accepted precision | Correct coverage | False acceptance | Correct | Wrong |
|---|---:|---:|---:|---:|---:|
| Development, dense only | 97.2% | 48.6% | 2.8% | 35 | 1 |
| Development, dense + rerank | 98.2% | 77.8% | 2.8% | 56 | 1 |
| Test (reused), dense only | 94.4% | 47.2% | 5.6% | 17 | 1 |
| Test (reused), dense + rerank | 92.9% | 72.2% | 5.6% | 26 | 2 |

Reranked cases: 388 of 625; skipped for length: 0. Warm rerank path (query embedding, exact search, reranking): p50 999.0 ms, p95 2363.5 ms over 12 samples. Reranker startup 251.0 s. Evaluation process memory with both models loaded: 3.31 GiB. Three values on 108 development families can overfit.
