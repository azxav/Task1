# Project guide: Uzbek semantic FAQ (Task 1)

This guide explains the whole project in plain words: what the task asks, how the system is built,
what every file does, how a question travels through the code, how quality was measured, and how
to run everything.

1. [The task](#1-the-task)
2. [The idea in one picture](#2-the-idea-in-one-picture)
3. [Technology and why each piece is used](#3-technology-and-why-each-piece-is-used)
4. [Folder map](#4-folder-map)
5. [The data](#5-the-data)
6. [Import: from CSV/Excel to a searchable index](#6-import-from-csvexcel-to-a-searchable-index)
7. [Search: from a user question to an answer](#7-search-from-a-user-question-to-an-answer)
8. [Uzbek text normalization](#8-uzbek-text-normalization)
9. [The answer-or-refuse decision](#9-the-answer-or-refuse-decision)
10. [FastAPI service](#10-fastapi-service)
11. [Telegram bot](#11-telegram-bot)
12. [Configuration and safety checks](#12-configuration-and-safety-checks)
13. [How quality was measured](#13-how-quality-was-measured)
14. [Results](#14-results)
15. [Tests](#15-tests)
16. [How to run it](#16-how-to-run-it)
17. [Limits, known problems, next steps](#17-limits-known-problems-next-steps)
18. [Short answers for the interview](#18-short-answers-for-the-interview)

---

## 1. The task

From `Tasklar.docx`, Task 1 ("Semantik qidiruvga asoslangan savol-javob tizimi"):

- Questions and answers are given as a spreadsheet (the archive actually contains two CSV files).
- A user writes a question **in their own words**; the system must find the best stored answer.
- Required stack: **bge-m3** (embeddings), **Qdrant** (vector DB), **PostgreSQL** (Q&A storage),
  **FastAPI** (API), **Telegram bot**.
- The system must:
  1. load the questions/answers into the database;
  2. search by meaning and return the answer;
  3. tell the user honestly when no good answer exists;
  4. work through both the API and the Telegram bot.
- Deliver: the code, a short README, and **how search quality was checked, with the result**.

What this project does **not** do: it does not generate text with an LLM. Every reply is a stored
answer, returned word for word. It also does not check whether the facts in the answers are still
legally correct.

## 2. The idea in one picture

```text
                      IMPORT (once, or when data changes)
 questions.csv ─┐
 answers.csv  ──┼─► validate ─► normalize text ─► BGE-M3 ─► vectors ─► Qdrant (new collection)
 or .xlsx     ──┘        │                                                   │
                         └──────────────► PostgreSQL (original text) ◄───────┘ publish together

                      SEARCH (every user question)
 Telegram ─► POST /search ─► normalize ─► BGE-M3 ─► Qdrant: top 30 similar stored questions
                                                    │
                                   group by answer ─┤
                                                    ▼
                    s1 ≥ 0.76 AND lead over the next answer ≥ 0.04  → matched (stage "dense")
                    s1 < 0.62 (floor)                               → no match (low_similarity)
                    otherwise (ambiguous, or 0.62 ≤ s1 < 0.76)      → rerank top 5 answer groups
                         r1 ≥ 0.75 AND r1 − r2 ≥ 0.22 → matched (stage "rerank")
                         else                         → no match (rerank_low_score / rerank_ambiguous)
                                                    ▼
                    matched  ─► fetch answer from PostgreSQL ─► reply word for word
                    no match ─► polite "no match" message
```

Key idea: we search over the **3,048 stored question variants**, not over the answers. Each stored
question points to one of the **505 answers**. If the user's question is close in meaning to a
stored question, that question's answer is the reply.

## 3. Technology and why each piece is used

| Piece | What it does here | Why |
|---|---|---|
| Python 3.12 + `uv` | Runs the app; `uv.lock` pins every dependency | Same install on every machine |
| **BGE-M3** (`BAAI/bge-m3`, pinned revision) | Turns text into 1,024 numbers (a vector); similar meaning → close vectors | Required; multilingual, works for Uzbek; runs locally on CPU, no cloud |
| **bge-reranker-v2-m3** (pinned revision) | Re-scores the top 5 answer groups for uncertain queries | Reads query and stored text together; local, multilingual |
| **Qdrant** | Stores the 3,048 question vectors; finds the nearest by cosine similarity | Required; vector search plus aliases for safe switching |
| **PostgreSQL** | Stores original questions, answers and publication metadata | Required; the source of truth for answer text |
| **FastAPI** | `POST /search`, `/health`, `/ready` | Required; one shared service for the bot and any HTTP client |
| **python-telegram-bot** | Long-polling bot that calls the API | Required; the bot does not load the model itself |
| Docker Compose | Starts PostgreSQL and Qdrant (and optionally API and bot) | Repeatable setup, data kept in volumes |
| pytest, ruff | Tests and lint | Quality checks |

Only BGE-M3's **dense** vectors are used. Its sparse/multi-vector modes and LLMs were left out on
purpose. The reranker was added later, only after it was measured on development data (§13–14).

## 4. Folder map

```text
.
├── README.md                 Short "how to run" page
├── pyproject.toml / uv.lock  Dependencies; defines the `faq` command
├── compose.yaml, Dockerfile  Docker services (postgres, qdrant, api, bot)
├── .env.example              All settings with defaults (copy to .env)
│
├── data/source/              The supplied data: questions.csv, answers.csv
│
├── src/faq/                     Application code
│   ├── cli.py                   `faq import | search | evaluate | bot`; the import pipeline
│   ├── config.py                Settings from env + frozen manifest; calibration guard
│   ├── data.py                  Read + validate CSV/Excel → Dataset object
│   ├── text/normalization.py    Uzbek text clean-up (apostrophes, spaces, BOM, …)
│   ├── text/scripts.py          Diagnostics: mixed Latin/Cyrillic words, duplicate questions
│   ├── embeddings.py            Loads BGE-M3 once; text → vectors; disk cache for import
│   ├── index.py                 Qdrant: create collection, switch alias, search
│   ├── storage.py               PostgreSQL: tables, snapshot replace, locks, fetch answer
│   ├── search/answer_groups.py  Group hits by answer; routing; the answer-or-refuse decision
│   ├── rerank.py                Loads the pinned reranker; scores (query, stored text) pairs
│   ├── service.py               SearchService: runs one question through everything
│   ├── api.py                   FastAPI app
│   ├── bot.py                   Telegram bot
│   └── evaluation/              Quality measurement (runner, metrics, labels, text profiles,
│                                rerank_calibration.py = reranker grid search + adoption rule)
│
├── evaluation/
│   ├── uzbek/                Benchmark inputs + frozen configuration
│   │   ├── base_queries.tsv, fresh_test.tsv, …   Hand-written test questions
│   │   ├── queries.jsonl                         All 625 generated test cases
│   │   ├── answer_equivalence.json               Answers reviewed as "the same"
│   │   ├── label_review.json                     Extra acceptable answers per question
│   │   └── manifest.json                         FROZEN profile + thresholds used at runtime
│   └── results/              Measured outputs (report.json, per-case predictions incl.
│                             rerank-predictions.jsonl, live API check)
│       └── v1/               First benchmark, kept unchanged on purpose (had label errors)
│
├── scripts/                          One-off tools
│   ├── audit_task1_data.py           Read-only audit of the original archive
│   ├── build_evaluation_cases.py     Builds evaluation/uzbek/queries.jsonl
│   ├── experiment_answer_support.py  Tried (and rejected) an extra re-ranking idea
│   └── verify_live_api.py            Checks real HTTP results match offline predictions
│
├── tests/                    Unit tests + one integration test
└── docs/
    ├── GUIDE.md                           This file
    ├── evaluation/uzbek-normalization.md  Quality report (written by `faq evaluate`)
    ├── research/                          Data audit findings, Uzbek spelling sources
    └── design/                            Original design documents
```

Not in git (ignored): `.venv/` (installed packages), `.cache/` (model weights ≈4.5 GB,
embedding cache and reranker score cache), `.env` (your bot token).

## 5. The data

- `data/source/questions.csv` — 3,048 rows: `id, answer_id, question, is_active, created_at, updated_at`.
- `data/source/answers.csv` — 505 rows: `id, category_id, topic_id, answer, status, created_at, updated_at`.

Facts from the audit (`docs/research/task1-data-findings.md`):

- All rows are active; every question points to an existing answer; 9 categories.
- Most answers have exactly 6 question variants (3 answers have 12). Variants follow templates
  such as "… haqida ma'lumot bering", so they are not independent real user questions.
- All apostrophes in the data are plain ASCII `'`. Real users type `ʻ`, `ʼ`, `‘`, `’` or `` ` ``.
  This mismatch is the main reason for the normalization step.
- Answer 499 has one Cyrillic letter inside a Latin word. It is kept as is (only flagged).
- Some answers mean the same thing (181/279, 380/494). Some only partly overlap (22/144).

An Excel file also works: one `.xlsx` with sheets named `questions` and `answers`, same columns.

## 6. Import: from CSV/Excel to a searchable index

Command: `uv run faq import data/source` (or a path to an `.xlsx`). Code: `cli.py → import_data()`.

1. **Validate everything first** (`data.py → load_dataset`). Checks: required columns, unique
   positive integer IDs, every `answer_id` exists, dates parse, `is_active` is true/false, `status`
   is active/inactive, text is not empty and has no control characters. Any error stops the import
   and names the file, sheet and row. **Nothing touches the databases until every row passes.**
   Warnings (not errors) are collected: mixed-script words, questions that become identical after
   normalization.
2. **Prepare search text** (`evaluation/text_profiles.prepare_profile`). Each question gets an
   `embedding_text` using the selected profile `uz_surface_v1` (§8). The original text is kept.
3. **Calibration guard** (`config.validate_calibration`). Data digest, answer-equivalence file,
   model, device, profile and thresholds must match `evaluation/uzbek/manifest.json`. If anything
   changed, the thresholds were not measured for it, so the import refuses (unless
   `FAQ_ALLOW_UNCALIBRATED=1`).
4. **Embed** (`embeddings.Encoder`). BGE-M3 loads once (pinned revision, CPU). A text longer than
   512 tokens is rejected, never silently cut. Vectors are normalized to length 1. Results are
   cached in `.cache/embeddings/`, keyed by exact text + model settings, so a repeat import is fast.
5. **Build a new Qdrant collection** (`index.create_generation`), named `faq_<random>`. Each point:
   id = question id, the vector, payload `{question_id, answer_id, generation}`. Only active
   questions whose answer is active are indexed.
6. **Publish** (`storage.snapshot_lock(publish=True)`):
   - take an exclusive PostgreSQL advisory lock (searches wait);
   - in one transaction, delete and re-insert all answers and questions, and write one `metadata`
     row (generation name, profile, model, digests, device, …);
   - switch the Qdrant alias `faq_questions` to the new collection;
   - release the lock.

Why a "generation" name: PostgreSQL and Qdrant cannot share one transaction. So both record the
same generation name. If they ever disagree (for example a crash in the middle), search refuses to
answer (`/ready` returns 503) instead of mixing old and new data. Running the import again fixes it.

Old Qdrant collections are kept for recovery; you can delete them by hand.

PostgreSQL tables (schema `faq`, created by `storage.py`):

| Table | Columns |
|---|---|
| `answers` | `id` PK, `category_id`, `topic_id`, `answer`, `status`, `created_at`, `updated_at` |
| `questions` | `id` PK, `answer_id` → answers, `question`, `embedding_text`, `is_active`, `created_at`, `updated_at` |
| `metadata` | single row `id=1`, `value` JSON (generation, profile, model, digests, …) |

## 7. Search: from a user question to an answer

Code: `service.py → SearchService.search()`.

1. Reject if longer than 2,000 characters (HTTP 422).
2. Normalize with the **same** profile used at import (§8).
3. Take a **shared** PostgreSQL lock (many searches can run together; an import waits for them).
4. `assert_ready()`: the published metadata must match the running settings (model, profile,
   device, equivalence file, calibration), and Qdrant's alias must point to the same generation.
   Otherwise → 503 "not ready".
5. Encode the question with BGE-M3 (a model lock lets one encode run at a time).
6. Ask Qdrant for the **top 30** most similar stored questions (exact cosine search).
7. **Group hits by answer** (`group_answer_candidates`). Six variants of the same answer must not
   compete with each other. Reviewed equivalent answers (181/279, 380/494) form one group.
   Each group keeps its best score.
8. **Decide** (`decide`, §9): matched or no_match.
   8b. If the dense result is uncertain (ambiguous, or score between the floor 0.62 and 0.76),
   fetch the question + answer text of the top 5 groups and rerank them (§9). A reranker error
   gives 503, never a silent no_match.
9. If matched, read the answer from PostgreSQL (it must exist and be active) and the matched
   question text. Return them unchanged.
10. Log request id, status, reason, stage and time. The user's text is **not** logged.

Response example:

```json
{
  "status": "matched", "reason": null,
  "answer_id": 1, "score": 0.83, "margin": 0.09, "question_id": 4,
  "answer": "…stored answer text…", "matched_question": "…stored question…",
  "message": null, "request_id": "…",
  "stage": "dense", "rerank_score": null, "rerank_margin": null
}
```

For no match: `status: "no_match"`, `reason` is `"low_similarity"`, `"ambiguous"`,
`"rerank_low_score"` or `"rerank_ambiguous"`, and
`message: "Savolingizga ishonchli mos javob topilmadi. Iltimos, savolni aniqroq yozing."`

## 8. Uzbek text normalization

Code: `text/normalization.py`. Profile name: `uz_surface_v1`. Goal: make *equal spellings equal*
without changing *meaning*.

What it changes:

| Input | Becomes | Why |
|---|---|---|
| `Oʻqish`, `O‘qish`, `O’qish`, ``O`qish`` | `O'qish` | Same Uzbek letter typed differently; data uses `'` |
| `maʼlumot` | `ma'lumot` | Same for the tutuq sign (ʼ) |
| `ID‑karta` (special hyphen) | `ID-karta` | Same hyphen, different code point |
| leading BOM, tabs, double spaces, non-breaking spaces | one normal space, trimmed | Invisible differences |
| composed/decomposed Unicode forms | NFC | Same letter, different encoding |

What it deliberately does **not** change: letter case, numbers, negation (`emas`), suffixes,
domains (`my.gov.uz`), acronyms, quotation marks outside letter context, Cyrillic letters, typos,
missing apostrophes. Changing these could change meaning, so they are measured instead (§13).

Apostrophes are replaced only **in letter context**: between two Latin letters, or right after
o/O/g/G (for example word-final `bog‘`). Quotes around a phrase like `'hayotiy vaziyat'` stay.

Each result has three forms:
- the original text — kept for storage and display;
- `embedding_text` — what BGE-M3 sees;
- `comparison_key` — lower-cased form, used only for duplicate checks and the string baseline.

On the supplied data this profile changes **zero** stored questions (they are already clean). It
matters for user input. Two other profiles were compared: `raw` (no change) and
`uz_surface_casefold_v1` (also lower-cases). See §14.

## 9. The answer-or-refuse decision

Code: `search/answer_groups.py → decide()`.

- `s1` = best group score, `s2` = second-best group score.
- **Matched** only if `s1 ≥ 0.76` **and** `s1 − s2 ≥ 0.04`.
- `s1 < 0.76` → `no_match`, reason `low_similarity` (nothing is close enough).
- `s1 − s2 < 0.04` → `no_match`, reason `ambiguous` (two different answers are almost equally close).

Scores are cosine similarities, **not** probabilities. The values 0.76 and 0.04 were not guessed:
they were chosen automatically on development data only (§13) and frozen in
`evaluation/uzbek/manifest.json`. The app reads them from there.

**Second stage (reranker).** Code: `route()` and `decide_reranked()` in the same file.

- 0.76 / 0.04 stay the "accept without reranker" line; the confident path is unchanged.
- `s1 < 0.62` (`rerank_floor`) → `no_match`, `low_similarity`; no reranking.
- Otherwise (ambiguous, or `0.62 ≤ s1 < 0.76`): the top 5 answer groups are scored by
  `bge-reranker-v2-m3`. Each pair is (user question, best stored question + newline + its answer).
  The score is the sigmoid of the cross-encoder output (0 to 1), **not** a probability.
- `r1` = best reranker score, `r2` = second best. **Matched** if `r1 ≥ 0.75`
  (`rerank_min_score`) and `r1 − r2 ≥ 0.22` (`rerank_min_margin`). With one group, only the
  score check applies. Otherwise `no_match` with `rerank_low_score` or `rerank_ambiguous`.
- Equal reranker scores → lowest answer ID wins (deterministic).
- A pair longer than 512 tokens is never cut: reranking is skipped and the dense no_match stays
  (log `rerank_skipped_length`).
- Response fields: `stage` (`"dense"` or `"rerank"`), `rerank_score`, `rerank_margin`. `score` and
  `margin` always hold the dense values.
- The values 0.62 / 0.75 / 0.22 and `rerank_enabled: true` are frozen in `manifest.json`.

Errors are separate from no_match: bad input → 422; database/Qdrant/model/reranker problems → 503.

## 10. FastAPI service

Code: `api.py`. Start: `uv run uvicorn faq.api:app --host 127.0.0.1 --port 8000 --workers 1`.

| Endpoint | Meaning |
|---|---|
| `GET /health` | The process is alive |
| `GET /ready` | Data published, generations match, configuration matches → 200; otherwise 503 |
| `POST /search` `{"question": "..."}` | Search; 200 with result, 422 bad input, 503 busy/unavailable |
| `GET /docs` | Auto-generated Swagger UI to try the API in a browser |

Details:
- The models load once at startup (BGE-M3 ≈5 s with cached weights; the reranker loads only when
  `rerank_enabled` is true). With both models the API process uses about 3.3 GB of RAM.
- Model work runs in a thread pool, so the async event loop stays free.
- At most 2 searches are admitted at once; if no slot frees within 0.2 s → 503 "busy".
- Use **one worker**: each worker would load its own copy of the large model.

## 11. Telegram bot

Code: `bot.py`. Start: `uv run faq bot` (needs `TELEGRAM_BOT_TOKEN` in `.env`, and the API running).

- Long polling, so no public URL is needed.
- `/start` → short Uzbek explanation of what the bot does.
- Text message → `POST /search` to the API → reply with the stored answer or the no-match message.
- Answers longer than Telegram's 4,096-unit limit are split safely (counted in UTF-16 units).
- 422 from the API → asks for a non-empty text under 2,000 characters. API down → "try later".
- Non-text messages (photo, sticker, …) → asks for text.
- The bot never loads the model; all logic is in the API. The token and chat text are never logged.

## 12. Configuration and safety checks

Settings come from environment variables or `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://faq:faq_local@127.0.0.1:55432/faq` | PostgreSQL |
| `QDRANT_URL` | `http://127.0.0.1:56333` | Qdrant |
| `MODEL_NAME` / `MODEL_REVISION` | `BAAI/bge-m3` / pinned commit | Exact model version |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` works but needs a new evaluation |
| `HF_HOME` | `.cache/huggingface` | Where model weights are stored |
| `TELEGRAM_BOT_TOKEN` | empty | Bot token (keep only in `.env`) |
| `FAQ_API_URL` | `http://127.0.0.1:8000` | Where the bot sends questions |
| `FAQ_PROFILE`, `FAQ_MIN_SCORE`, `FAQ_MIN_MARGIN`, `FAQ_TOP_K` | from `manifest.json` | Leave unset |
| `FAQ_RERANK_ENABLED`, `FAQ_RERANK_FLOOR`, `FAQ_RERANK_MIN_SCORE`, `FAQ_RERANK_MIN_MARGIN`, `FAQ_RERANK_GROUPS` | from `manifest.json` | Leave unset |
| `RERANKER_NAME` / `RERANKER_REVISION` | `BAAI/bge-reranker-v2-m3` / pinned commit | Exact reranker version |
| `FAQ_ALLOW_UNCALIBRATED` | `0` | `1` = allow unmeasured settings (experiments only) |

URLs use `127.0.0.1`, not `localhost`: on Windows, `localhost` first tries IPv6, which added
seconds of delay per connection.

**Calibration guard:** thresholds are only valid for the exact model, device, profile, data and
equivalence file they were measured with (and, when enabled, the exact reranker and its three
values). Import and `/ready` compare all of these with
`manifest.json` and refuse a mismatch. This stops "changed the model but kept the old thresholds".

## 13. How quality was measured

Command: `uv run faq evaluate` (code: `evaluation/runner.py`). It uses the real model and exact
cosine search, offline; it does not publish anything.

**Test questions** (`evaluation/uzbek/`), written by hand, not copied from the data:

- *Development set* (used to choose settings): 72 answerable questions (8 per category) + 36 that
  must be refused (18 unrelated, 18 "do it for me / check my private data" requests the FAQ
  cannot do).
- *Fresh test set* (used once, at the end): 36 answerable (4 per category) + 18 must-refuse.
- Each question is a **family**. `scripts/build_evaluation_cases.py` generates variants from it:
  UPPER CASE, odd whitespace/BOM, typographic apostrophes, missing apostrophes, small typos,
  Cyrillic spelling, one Cyrillic letter mixed in. Variants stay in their parent's split, so
  nothing leaks between development and test. Total: **625 cases**.
- Each answerable question lists all acceptable answer IDs, decided by reading the full answers
  (`label_review.json`, `answer_equivalence.json`).
- A test question that exactly equals a stored question (after normalization) is rejected, so the
  test is not trivial.

**Procedure:**

1. Embed all stored questions and all test queries for 3 profiles (`raw`, `uz_surface_v1`,
   `uz_surface_casefold_v1`).
2. For each profile, try every score threshold from 0.35 to 0.96 and every margin from 0 to 0.15,
   **on the development set only** (base and must-refuse questions; variants excluded). Goal:
   precision ≥ 95% and false acceptance ≤ 5%; among those, accept the most correct answers.
3. Choose the profile; write it and its thresholds into `manifest.json` (**frozen**).
4. Only then compute results on the fresh test set. No tuning after that.

**Second stage (reranker)** (`evaluation/rerank_calibration.py`):

1. The dense profile and 0.76 / 0.04 stay frozen.
2. Every case that is not a confident dense match and has `s1 ≥ 0.50` gets reranker scores for its
   top 5 groups once (cached in `.cache/rerank/`).
3. Grid search **on development base and must-refuse families only** (an assertion refuses any
   test row): floor 0.50–0.76, reranker score 0.00–0.99, reranker margin 0.00–0.50, step 0.01.
   Same objective as above, utility `correct − 8 × wrong`; ties prefer fewer reranks.
4. Adoption rule: correct answers must rise, precision ≥ 95%, false acceptance ≤ 5% (development).
   If it fails, `rerank_enabled: false` and runtime stays dense-only.
5. Only then compute results on the test split. That split was already used once, so it is
   labeled **"reused, not fresh"** (`test_status: "reused"` in the manifest).

**Metrics** (`evaluation/metrics.py`):

- *Top-1 accuracy* — is the best answer correct (before any refusal)?
- *Recall@5* — is a correct answer in the top 5?
- *Accepted precision* — of the answers given, how many were correct?
- *Correct answer coverage* — of answerable questions, how many got a correct answer?
- *False acceptance* — of must-refuse questions, how many got an answer anyway?

**Extra checks:** character n-gram TF-IDF baseline (is the neural model better than string
matching?), a "hold out one stored variant" test, number/negation diagnostics, latency, and
offline-vs-real-Qdrant parity. `scripts/verify_live_api.py` sends all 54 test families (reused, not
fresh, since the reranker) through the real HTTP API and confirms the same decisions.

**History, kept on purpose:** the first benchmark (`evaluation/results/v1/`) missed some valid
alternative answers, so it under-scored the system. It was kept unchanged; its test questions moved
to the development set (they had been inspected), and a new fresh test set was written.
`scripts/experiment_answer_support.py` tried re-ranking with answer vectors on development data; it
lowered coverage, so it was not adopted.

## 14. Results

Fresh held-out test, base questions + must-refuse (54 families):

| Profile | Top-1 | Accepted precision | Correct coverage | False acceptance |
|---|---:|---:|---:|---:|
| raw | 75.0% | 94.4% | 47.2% | 5.6% |
| **uz_surface_v1 (selected)** | 75.0% | **94.4%** | **47.2%** | **5.6%** |
| uz_surface_casefold_v1 | 75.0% | 90.5% | 55.6% | 5.6% |

Selected: `uz_surface_v1`, minimum score **0.76**, margin **0.04**.

In counts: 17 correct answers, 1 wrong answer, 19 of 36 answerable questions refused, and 1 of 18
must-refuse questions wrongly answered. Recall@5 = 94.4%.

How to read this:
- When the system answers, it is right about 94% of the time.
- It is cautious: it refuses about half of the answerable questions. The right answer is usually
  in the top 5 (94%), so better ranking could raise coverage later.
- The 95% / 5% targets were narrowly missed on this small test; this is reported, not hidden.
- Character TF-IDF baseline: top-1 75%, recall@5 91.7%. The data is very template-like, so plain
  string matching is a strong baseline here.
- Variants: typographic and missing apostrophes keep about 76% top-1; Cyrillic 67%;
  **UPPER CASE drops top-1 to 28%** — a known weakness of the case-preserving profile.
- Latency on CPU, dense path: model + vector search p50 116 ms / p95 124 ms (latest `faq evaluate`;
  an earlier run measured 188 / 212 ms on the same machine). Full HTTP API before the reranker
  (earlier live run): p50 339 ms / p95 457 ms.

**Reranker stage** (`bge-reranker-v2-m3`, floor 0.62, reranker score 0.75, margin 0.22; adopted).
**The test split is reused, not fresh**: it was measured once before this change and motivated it.
No value was tuned on it, but it is not independent evidence.

| Data | Accepted precision | Correct coverage | False acceptance | Correct | Wrong |
|---|---:|---:|---:|---:|---:|
| Development, dense only | 97.2% | 48.6% | 2.8% | 35 | 1 |
| Development, dense + rerank | 98.2% | 77.8% | 2.8% | 56 | 1 |
| Test (reused), dense only | 94.4% | 47.2% | 5.6% | 17 | 1 |
| Test (reused), dense + rerank | 92.9% | 72.2% | 5.6% | 26 | 2 |

- Correct coverage rises by 29.2 points on development (48.6% → 77.8%) and 25.0 points on the
  reused test (47.2% → 72.2%). On the reused test, precision falls to 92.9% (2 wrong of 28
  answers) and false acceptance stays 5.6% (1 of 18); both miss the 95% / 5% targets. Three values
  chosen on 108 development families can overfit.
- 388 of 625 cases were scored by the reranker for calibration (dense score ≥ 0.50). At the chosen
  floor 0.62, 290 of 625 cases take the rerank path (100 of them in the test split). None was
  skipped for length.
- Latency on CPU: reranked queries over real HTTP p50 1,083 ms / p95 1,280 ms (27 samples); all 54
  live queries p50 593 ms / p95 1,258 ms. Offline warm rerank path p50 999 ms / p95 2,363 ms.
- Memory: about 3.3 GB RAM with both models loaded (evaluation process 3.31 GiB from
  `report.json`; API process 3.26 GiB Windows working set, measured by hand after the live check).
- The live check (`verify_live_api.py`) gave the same decision for all 54 test families, including
  the 27 reranked ones. The recorded reranker start time (251 s) includes the first download.

Full numbers: `docs/evaluation/uzbek-normalization.md`, `evaluation/results/report.json`, and
per-case predictions in `evaluation/results/*-predictions.jsonl`.

## 15. Tests

```powershell
uv run pytest -q                              # unit tests (no services needed)
$env:RUN_INTEGRATION='1'; uv run pytest -q    # also the real PostgreSQL/Qdrant test
```

| Test file | Checks |
|---|---|
| `tests/text/test_normalization.py` | Apostrophe/hyphen/BOM/whitespace rules; repeat-safe; meaning tokens kept |
| `tests/text/test_corpus_lint.py` | Mixed-script and collision reports; real corpus keeps its meaning |
| `tests/test_import.py` | Real data loads; bad rows rejected before DB changes; Excel = CSV |
| `tests/search/test_answer_groups.py` | Equivalent IDs, partial overlap, threshold boundaries, ties; routing and reranked decision |
| `tests/test_service_api.py` | Verbatim answers, 422 / 503 codes, no mixed generations; rerank branch, reranker error → 503, long pair → dense |
| `tests/test_calibration_provenance.py` | IPv4 defaults; changed pipeline, data or reranker values need recalibration |
| `tests/test_evaluation.py` | Threshold selection, split leakage, metric definitions, top groups |
| `tests/test_rerank.py` | Reranker score order, long pair rejected (not cut), non-finite score, cache |
| `tests/test_rerank_calibration.py` | Grid selection, development rows only, adoption rule |
| `tests/test_bot.py` | Telegram message splitting and reply text |
| `tests/integration/test_snapshot.py` | Real publish + search on PostgreSQL/Qdrant |

Current result: 97 passed, 1 skipped (the integration test needs `RUN_INTEGRATION=1`).

## 16. How to run it

Needs: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker. The first run downloads ≈4.5 GB of
model weights (BGE-M3 + reranker).

```powershell
uv sync --frozen --extra dev                 # install packages
docker compose up -d postgres qdrant         # start the databases
uv run faq import data/source                # load data (first time also downloads the model)
uv run uvicorn faq.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal:

```powershell
uv run faq search "ID-kartamni yangilash uchun arizani qayerga beraman?"
Invoke-RestMethod -Uri http://127.0.0.1:8000/search -Method Post -ContentType 'application/json; charset=utf-8' -Body '{"question":"ID-karta olishga qaysi hujjatlar kerak?"}'
```

Telegram: copy `.env.example` to `.env`, set `TELEGRAM_BOT_TOKEN`, then run `uv run faq bot`
(the API must be running).

Everything in Docker:

```powershell
docker compose --profile app build api
docker compose --profile app run --rm api faq import data/source
docker compose --profile app up -d api
docker compose --profile app --profile bot up -d bot
```

Measure quality again:

```powershell
uv run python scripts/build_evaluation_cases.py   # rebuild queries.jsonl
uv run faq evaluate                                # recalibrate + report (overwrites manifest/results)
uv run faq import data/source                      # republish with the selected profile
```

Stop: `docker compose down` (keeps data) or `docker compose down -v` (deletes data).

## 17. Limits, known problems, next steps

Limits:
- Small benchmark (54 test families); generated variants are correlated, not extra evidence.
- Dense alone refuses many answerable questions (47% correct coverage); the reranker raises this to
  72% on the reused test, but with 92.9% precision, and each reranked query takes about 1 s on CPU.
- The reranker values were checked only on a reused test split; a fresh test is still needed.
- UPPER-CASE queries do badly; Cyrillic and typos are weaker than clean Latin.
- Near-topic requests ("check my application status") can still look like an FAQ question.
- Facts and fees in the answers were not checked for current legal accuracy.
- Import needs a short downtime; no authentication, rate limiting or monitoring.

Next steps (each needs a new fresh test set after the change):
1. Collect real user questions and label them independently.
2. Fix upper-case input (for example, restore case for all-caps queries) and measure it.
3. Done: reranker stage added and adopted (§14). Measured coverage 72.2% on the reused test.
   Next: confirm it on a fresh test, and try hybrid dense + sparse search.
4. Optional Cyrillic → Latin transliteration as an extra candidate query.
5. Production: authentication, rate limits, monitoring, zero-downtime publishing.

## 18. Short answers for the interview

- **How did you understand the task?** Retrieval, not generation: find the stored answer whose
  question means the same as the user's, or say honestly that none fits.
- **Why search questions, not answers?** Users ask questions; stored variants look like user
  questions, so question-to-question similarity is the most direct signal.
- **Why both PostgreSQL and Qdrant?** Qdrant finds similar vectors; PostgreSQL holds the original
  text and metadata. The Qdrant index can always be rebuilt from the data.
- **How do you avoid wrong answers?** Two rules: a minimum similarity and a minimum lead over the
  next different answer. Both were tuned on development data and frozen before the final test.
- **Why group by answer?** Six variants of one answer would otherwise make the lead look tiny and
  cause false "ambiguous" refusals.
- **Why not lower-case everything?** Measured: lower-casing answered more but with lower precision
  (90.5% vs 94.4%). The case-preserving profile won on development data.
- **How do you know the API behaves like the offline evaluation?** `verify_live_api.py` sent all
  54 test families through the real HTTP stack (with the reranker on): same decisions, dense score
  differences < 1e-6, reranker score differences < 1e-4.
- **What if the data changes?** Re-import; the calibration guard detects the new data and asks for
  a new evaluation before publishing with old thresholds.
- **Biggest weakness?** Upper-case input, and evidence: the reranker raised coverage from 47% to
  72%, but only on a reused test split (precision 92.9%). A fresh held-out set must confirm it.
- **Why a reranker only for uncertain queries?** Confident dense matches are already ~97% precise
  and fast. The cross-encoder reads question and stored text together, so it separates the gray
  zone better, but it costs about 1 s per query on CPU.
