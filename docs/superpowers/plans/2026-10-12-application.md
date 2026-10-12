# Application execution plan

User approved the five-stage implementation on 2026-10-12. Execute inline, beginning with the detailed Uzbek normalization plan and its spec; no repeated design handoff is needed.

1. Pure normalizer and script/collision lint; tests include actual corpus and typography mutants. Files: src/faq/text, tests/text.
2. Import source models/CSV/Excel validation, PostgreSQL snapshot storage, BGE-M3 encoding/cache, versioned Qdrant index; real import idempotency and generation checks. Files: src/faq/data.py, storage.py, embeddings.py, index.py, cli.py; tests/test_import.py, tests/integration.
3. Reviewed answer groups and shared search/decision layer; threshold boundary tests and generation mismatch tests. Files: src/faq/search, service.py; tests/search.
4. Curated labeled families and normalization ablation, calibration, held-out test, lexical baseline and separate source-variant holdout. Files: evaluation/uzbek, src/faq/evaluation, docs/evaluation. Freeze profile, equivalence registry and thresholds before final evaluation. Optional adapters only with failure evidence.
5. API, Telegram client, Compose, Dockerfile, startup scripts and README. Files: src/faq/api.py, bot.py, config.py. Test API behavior, real database/index/API flow and bot message formatting. Live Telegram requires a supplied token; expose that concrete dependency if unavailable.
6. Whole-branch fresh review, regression fixes with tests, final verification and concise handoff. Do not merge/push/deploy externally without authorization.

Ruling: this is a dedicated clean job-application workspace; keep files in the user's current directory on feat/semantic-faq rather than creating another checkout. No existing product code or concurrent user changes are displaced.
