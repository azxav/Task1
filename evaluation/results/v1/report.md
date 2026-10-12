# Uzbek FAQ measured quality

Measured with pinned BGE-M3 on local CPU; no generative model or external inference.

108 independently authored query families: 72 answerable (8/category) and 36 no-match/action requests. Families split before typography/casing/typo/Cyrillic variants. 404 total cases; calibration selects profiles and rejection thresholds. Final test is evaluated afterward; labels are reviewed source-coverage judgments, not an external adjudication.

| Profile | Test top-1 | Accepted precision | Answer coverage | No-match false acceptance |
|---|---:|---:|---:|---:|
| raw | 69.4% | 100.0% | 25.0% | 0.0% |
| uz_surface_v1 | 69.4% | 100.0% | 25.0% | 0.0% |
| uz_surface_casefold_v1 | 66.7% | 100.0% | 25.0% | 0.0% |

Selected on calibration: **uz_surface_casefold_v1**, minimum score **0.82**, ambiguity margin **0.0**.

Full counts, per-slice metrics, primary-ID metrics, calibration figures, source-variant holdout and Qdrant parity are in evaluation/results/report.json. Per-case predictions are beside it.

Warm CPU model plus exact vector search: p50 211.4 ms; p95 221.1 ms over 12 single-query samples. This excludes HTTP/SQL overhead. Model startup: 4.9 seconds with cached weights; initial download/import is separate.

## Limits and next steps

Small balanced benchmark is not production traffic. Variants are correlated within families and cannot be counted as independent evidence. Primary-ID results can penalize valid alternative answers; semantic labels and equivalence groups are reported separately. The variant holdout retains closely related sibling templates and is optimistic. Corpus statements/fees have not been checked for legal currency. Source answer 499 retains a Cyrillic lookalike by design.

A similarity score is not a probability. Closely related unsupported private lookups/actions can still retrieve a plausible FAQ; rejection rates must be read alongside answer coverage. No automatic typo correction, stemming or transliteration has been added. Review calibration failures before adding narrowly tested candidate adapters, then use a fresh held-out set if final-test failures influence a redesign.
