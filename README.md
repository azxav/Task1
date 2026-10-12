# Uzbek semantic FAQ

Task 1: retrieve a stored answer by meaning using **BGE-M3 → Qdrant → PostgreSQL**, exposed through **FastAPI and Telegram**. Answers are returned verbatim; no generative model or cloud inference. Provided CSVs are included in `data/source`; Excel import uses `questions` and `answers` sheets with the same headers.

## Run locally

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Desktop/Engine, and initial internet access to download dependencies/model weights (~2.3 GB). CPU inference works; local measurements are in [quality report](docs/evaluation/uzbek-normalization.md).

Local dependency URLs use `127.0.0.1`, matching Compose's IPv4 bindings. Windows `localhost` IPv6 fallback caused seconds of delay per connection in live verification.

```powershell
uv sync --frozen --extra dev
docker compose up -d postgres qdrant
uv run faq import data/source
uv run uvicorn faq.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal:

```powershell
uv run faq search "ID-kartamni yangilash uchun arizani qayerga beraman?"
Invoke-RestMethod -Uri http://127.0.0.1:8000/search -Method Post -ContentType 'application/json; charset=utf-8' -Body '{"question":"ID-karta olishga qaysi hujjatlar kerak?"}'
uv run faq bot
```

For the bot, create local `.env` and set `TELEGRAM_BOT_TOKEN`. Keep `.env` untracked. `/start` describes scope; text questions call the API, non-text input requests text. Bot uses long polling and a reused HTTP client; only the API loads the model. End both foreground commands with Ctrl+C. API documentation: [localhost:8000/docs](http://localhost:8000/docs).

Implementation explanation for interview preparation: [walkthrough](docs/walkthrough.md).

Optional database/password/model settings are documented in `.env.example`. Omit `FAQ_MIN_SCORE`, `FAQ_MIN_MARGIN`, and `FAQ_PROFILE` to use the measured frozen configuration in `evaluation/uzbek/manifest.json`. Model, device, profile, corpus, registry, top-k and threshold changes require recalibration; import/readiness reject mismatched provenance. `FAQ_ALLOW_UNCALIBRATED=1` permits deliberate experiments, marked uncalibrated in import metadata. If changing POSTGRES_PASSWORD, set the matching local DATABASE_URL too. Do not change credentials on an existing volume without migrating them.

## All application processes in Docker

```powershell
docker compose --profile app build api
docker compose --profile app run --rm api faq import data/source
docker compose --profile app up -d api
docker compose --profile app --profile bot up -d bot
```

Host `.env` supplies the bot token to Compose; other local values apply only where referenced in compose.yaml. PostgreSQL and Qdrant use persistent named volumes. Docker model weights have their own cache volume and may download again after a host-only run. GPU use is optional: install a driver-compatible PyTorch build and set EMBEDDING_DEVICE; this demo's measured configuration uses CPU. Changing device/precision warrants a new measured evaluation; do not assume identical scores.

## Import and publication

```powershell
uv run faq import path/to/workbook.xlsx
```

Input validation finishes before either database changes: headers, IDs, references, timestamps, booleans, active status and nonempty/control-safe text. Warnings identify mixed-script text and normalized collisions without changing originals. The importer builds a new collection, then uses an exclusive PostgreSQL advisory lock while replacing the relational snapshot transactionally and switching the Qdrant alias. Search holds a shared lock through retrieval and answer fetching. Readiness checks model/profile/registry/generation consistency; a partial publication fails closed. Re-run import to recover; stop API/bot for planned maintenance. Repeated import is logically idempotent but creates a new index generation.

Existing old collections are retained for recovery. They can be removed manually after successful verification; the CLI does not delete published history automatically. PostgreSQL uses a dedicated `faq` schema; integration tests use `faq_test` and a separate vector alias. `docker compose down` stops services while retaining volumes; `down -v` deletes local database/model data.

## Text and retrieval flow

```text
CSV/Excel → validation → original questions/answers in PostgreSQL
                      → shared text profile → local embeddings → versioned Qdrant collection

Telegram or POST /search → same profile → query embedding → top 30 questions
                         → distinct reviewed answer groups → calibrated score/margin checks
                         → stored answer, no_match, or service error
```

`uz_surface_v1` preserves case and meaning: NFC, contextual Uzbek letter/tutuq markers mapped to corpus ASCII apostrophes, word-final g markers, contextual backticks, letter hyphens and whitespace. Separate comparison keys casefold for lint/lexical search. Preserve numbers, negation, suffixes, domains, acronym text and outer quotation marks. Missing apostrophes, typos and Cyrillic are evaluated, not globally rewritten. See [normalization design](docs/superpowers/specs/2026-10-12-uzbek-normalization-design.md) and [actual data findings](docs/research/task1-data-findings.md).

Answer equivalence registry currently groups 181/279 and 380/494 after source-text review; 22/144 remain distinct globally because coverage differs. No match means similarity is low or distinct groups are too close. Scores are cosine similarity, not probabilities. Invalid requests return 422; dependency/configuration failures and overload return 503. `/health` checks liveness; `/ready` checks the published snapshot. Search inference runs outside the async event loop, with bounded admission and a model lock. Keep one API worker to avoid duplicating model memory. Input limit is 2,000 characters **and** 512 tokenizer tokens; oversized inputs are rejected instead of silently truncated.

## Quality and tests

```powershell
uv run python scripts/build_evaluation_cases.py
uv run faq evaluate
# Evaluation may select another profile. Publish its matching index afterward:
uv run faq import data/source
uv run pytest -q
$env:RUN_INTEGRATION='1'
uv run pytest -q
```

Evaluation validates family-level splits and expected-answer coverage, compares raw/conservative/casefold profiles, calibrates score/margin rules on independent calibration families only, freezes selection, then measures fresh held-out test. Version 1 had incomplete acceptable-ID labels; original artifacts remain in `evaluation/results/v1` and inspected families now form development data. Version 2 adds 54 fresh test families; generated variants stay with their parents and do not influence threshold selection. It includes a character-TFIDF baseline, no-match cases, Cyrillic/typo/keyboard/mixed-script slices, numeric/negation token diagnostics, equivalent-answer handling, and a separate optimistic supplied-variant holdout. Answer-vector reranking improved development top-1 but reduced accepted coverage, so was not added to runtime. Predictions and counts are in `evaluation/results`; report is in `docs/evaluation`. Re-running calibration overwrites current manifest/results; preserve results before a new experiment. Changes inspired by final-test failures require another fresh held-out set.

Source data: 3,048 questions, 505 answers, nine category IDs. Most variants follow repeated templates; source IDs/timestamps are preserved. Answer 499 has a Cyrillic lookalike in Latin text and remains verbatim. Curated facts/fees have not been checked for legal currency. Scope is stored FAQ guidance, not private account lookup, submission of applications, personalized professional advice, or current external-data verification.

Known limitations: small curated benchmark, correlated synthetic variants, incomplete coverage of possible acceptable alternative answers, uncertain rejection of near-topic unsupported requests, dense-model sensitivity to script/wording, CPU latency and model download size. Review actual failures before adding hybrid retrieval, reranking or a targeted transliteration candidate. Production work would add access control, rate limits, monitoring, a maintained knowledge source, and operational recovery tooling.

Local verification: 73 tests passed including PostgreSQL/Qdrant integration. All 54 fresh base/no-match HTTP decisions agree with frozen offline predictions; stored answers are unchanged. API median 339 ms / p95 457 ms on this host, recorded in `evaluation/results/live-api.json`. User confirmed Telegram replies for `/start`, one FAQ and one unrelated question. Final base-test precision 94.4%, correct answer coverage 47.2%; this prototype still rejects many answerable questions.
