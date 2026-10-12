# Execution ledger — Task 1

Plan: docs/superpowers/plans/2026-10-12-application.md and 2026-10-12-uzbek-normalization.md.

Preflight: Task 1/2 normalizer signature shared; importer and query use embedding_text, never comparison_key. Task 3/4 evaluation depends on reviewed registry; complete registry before calibration. Optional query adapters must be selected before final test. Hardware: 16 GB RAM, RTX 4050; Docker initially stopped. Python 3.12 isolated environment.

Ruling: approved five-stage scope permits implementation and necessary local dependency/model/service setup. Dedicated clean workspace stays on feature branch; no second checkout or repeated approval gates. No remote deployment or messaging others.

Completed: corpus audit; shared normalizer; validated CSV/Excel importer; PostgreSQL/Qdrant publication; search; FastAPI; Telegram adapter; unit/integration checks; CPU Docker image build.

Quality: first benchmark archived in evaluation/results/v1. Review found incomplete acceptable-answer labels and leading BOM idempotence defects. BOM regression fixtures now pass. Reviewed alternatives will be corrected; inspected v1 test becomes development data. Fresh families reserved for final test. Investigate answer-vector support on development data before changing production retrieval.

Completed review fixes: idempotent leading BOM handling; reviewed query-specific acceptable IDs; parsed-row corpus digest allows identical CSV/Excel exports; model/device/profile/corpus/registry/threshold provenance guard. v1 archived without changing original results. Answer-support reranking tested and declined due lower accepted coverage.

Version 2: 108 development + 54 fresh test families, 625 total cases. Selected uz_surface_v1 / score 0.76 / margin 0.04. Fresh base precision 94.4%, correct coverage 47.2%, unsupported false acceptance 1/18. Targets missed on final test; no tuning afterward. Numeric/negation, Cyrillic and mixed-script diagnostics recorded.

Verification: 73 tests passed including isolated real PostgreSQL/Qdrant integration; Ruff clean. Full 3,048-question / 505-answer Excel import succeeds with the same canonical digest as CSV. Real HTTP/model/vector/database flow agrees with all 54 frozen test decisions; verbatim answers, 422 limits and typography equivalence checked. Locked CPU wheel agrees with original calibration within 1e-5. API p50 339 ms, p95 457 ms. Windows localhost IPv6 fallback diagnosed by paired dependency timings and corrected to Compose-bound 127.0.0.1 defaults.

Telegram: token remains local/untracked; getMe verified ragtaskbot_bot. User confirmed replies to /start, a FAQ question and an unrelated question. API and bot running locally. README, walkthrough and measured reports included. CPU Docker image build verified. Current case hash is LF-normalized across platforms; archived v1 and raw CSV bytes retain original hashes. Staged whitespace and secret checks pass. Implementation complete on feat/semantic-faq; local commit follows. No remote push/deployment.
