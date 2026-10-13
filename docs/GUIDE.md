# Project guide: Uzbek semantic FAQ (Task 1)

This guide explains the whole project in plain words: the task, how a question becomes an answer,
what every file does, how quality was measured, what the results are, and how to run everything.

1. [The task](#1-the-task)
2. [The whole system on one page](#2-the-whole-system-on-one-page)
3. [Technology and why](#3-technology-and-why)
4. [Folder map](#4-folder-map)
5. [The data](#5-the-data)
6. [Import: from files to a searchable index](#6-import-from-files-to-a-searchable-index)
7. [Search: from a question to an answer](#7-search-from-a-question-to-an-answer)
8. [Uzbek text normalization](#8-uzbek-text-normalization)
9. [The decision: answer or refuse](#9-the-decision-answer-or-refuse)
10. [FastAPI service](#10-fastapi-service)
11. [Telegram bot](#11-telegram-bot)
12. [Configuration and the calibration guard](#12-configuration-and-the-calibration-guard)
13. [How quality was measured](#13-how-quality-was-measured)
14. [Results](#14-results)
15. [Tests](#15-tests)
16. [How to run it](#16-how-to-run-it)
17. [Limits and next steps](#17-limits-and-next-steps)
18. [Short answers for the interview](#18-short-answers-for-the-interview)

---

## 1. The task

From `Tasklar.docx`, Task 1 ("Semantik qidiruvga asoslangan savol-javob tizimi"):

- Questions and answers are given as a spreadsheet (the archive contains two CSV files).
- A user writes a question **in their own words**; the system must find the best stored answer.
- Required stack: **bge-m3** (embeddings), **Qdrant** (vector database), **PostgreSQL** (question
  and answer storage), **FastAPI** (API), **Telegram bot**.
- The system must load the data, search by meaning, say honestly when no good answer exists, and
  work through both the API and the bot.
- Deliverables: code, a short README, and **how search quality was checked, with the result**.

What the project does **not** do: it does not write text with an LLM. Every reply is a stored
answer, returned word for word. It does not check whether the facts in the answers are still legally
correct.

## 2. The whole system on one page

```text
                      IMPORT (once, or when the data changes)
 questions.csv ─┐
 answers.csv  ──┼─► validate ─► normalize text ─► BGE-M3 ─► vectors ─► Qdrant (new collection)
 or .xlsx     ──┘        │                                                   │
                         └──────────────► PostgreSQL (original text) ◄───────┘ publish together

                      SEARCH (every user question)
 Telegram ─► POST /search ─► normalize ─► BGE-M3 ─► Qdrant: top 30 similar stored questions
                                                          │
                                         group by answer ─┤  s1 = best group score
                                                          ▼  s2 = second best
   s1 ≥ 0.76 and s1 − s2 ≥ 0.04  ───────────────────────────► matched        (stage "dense")
   s1 < 0.62                     ───────────────────────────► no match: low_similarity
   anything else (unsure)        ─► reranker reads the top 5 answers with the question
                                     r1 ≥ 0.75 and r1 − r2 ≥ 0.22 ─► matched (stage "rerank")
                                     otherwise                    ─► no match: rerank_low_score
                                                                       or rerank_ambiguous
   matched  ─► read the answer from PostgreSQL ─► reply word for word
   no match ─► polite "no match" message
```

Three ideas carry the design:

1. **Search questions, not answers.** The data has 3,048 stored question variants that point to
   505 answers. Users ask questions, so the system compares the user's question with stored
   questions. The answer of the closest stored question is the reply.
2. **Two stages.** A fast first stage (BGE-M3 vectors) settles clear cases. Only unsure cases go to
   a slower, more careful second stage (the reranker).
3. **Refuse when unsure.** Every threshold was chosen on development data so that wrong answers
   stay rare. Refusing is better than giving a wrong official answer.

## 3. Technology and why

| Piece | What it does here | Why |
|---|---|---|
| Python 3.12 + `uv` | Runs the app; `uv.lock` pins every dependency | Same install on every machine |
| **BGE-M3** (`BAAI/bge-m3`, pinned revision) | Turns text into 1,024 numbers (a vector); similar meaning → close vectors | Required; multilingual, handles Uzbek; runs on CPU, no cloud |
| **bge-reranker-v2-m3** (pinned revision) | Reads the question and a stored question + answer together and gives one score | Better judge for unsure cases; multilingual; local |
| **Qdrant** | Stores the 3,048 question vectors; finds the nearest by cosine similarity | Required; exact search, aliases for safe switching |
| **PostgreSQL** | Stores original questions, answers and publication metadata | Required; the source of truth for answer text |
| **FastAPI** | `POST /search`, `GET /health`, `GET /ready` | Required; one service for the bot and any HTTP client |
| **python-telegram-bot** | Long-polling bot that calls the API | Required; the bot holds no model |
| Docker Compose | Starts PostgreSQL and Qdrant (optionally API and bot) | Repeatable setup; data kept in volumes |
| pytest, ruff | Tests and lint | Quality checks |

BGE-M3 vs the reranker: BGE-M3 turns each text into a vector **separately**, so all 3,048 stored
questions are prepared once and compared in milliseconds. The reranker reads the user question
**together with** one stored text, which is more accurate but takes about 1 second for 5
candidates on CPU. That is why it only runs for unsure cases.

Left out on purpose: BGE-M3's sparse and multi-vector modes, and any LLM.

## 4. Folder map

```text
.
├── README.md                 Short "how to run" page and the main numbers
├── pyproject.toml / uv.lock  Dependencies; defines the `faq` command
├── compose.yaml, Dockerfile  Docker services (postgres, qdrant, api, bot)
├── .env.example              All settings with defaults (copy to .env)
│
├── data/source/              The supplied data: questions.csv, answers.csv
│
├── src/faq/                     Application code
│   ├── cli.py                   `faq import | search | evaluate | bot`; the import pipeline
│   ├── config.py                Settings from env + frozen manifest; calibration guard
│   ├── data.py                  Read and validate CSV/Excel → Dataset
│   ├── text/normalization.py    Uzbek text clean-up (apostrophes, spaces, BOM, …)
│   ├── text/scripts.py          Diagnostics: mixed Latin/Cyrillic words, duplicate questions
│   ├── embeddings.py            Loads BGE-M3 once; text → vectors; disk cache
│   ├── rerank.py                Loads the reranker once; scores (question, stored text) pairs
│   ├── index.py                 Qdrant: create collection, switch alias, search
│   ├── storage.py               PostgreSQL: tables, snapshot replace, locks, read texts
│   ├── search/answer_groups.py  Group hits by answer; routing; both decision stages
│   ├── service.py               SearchService: runs one question through everything
│   ├── api.py                   FastAPI app
│   ├── bot.py                   Telegram bot
│   └── evaluation/              Quality measurement
│       ├── runner.py            `faq evaluate`: the full measurement run
│       ├── metrics.py           Metric definitions; dense threshold search
│       ├── rerank_calibration.py  Reranker value search + adoption rule
│       ├── labels.py            Checks the test cases are valid
│       └── text_profiles.py     The three text profiles that were compared
│
├── evaluation/
│   ├── uzbek/                Benchmark inputs + frozen configuration
│   │   ├── base_queries.tsv, no_match_queries.txt, cyrillic_variants.tsv   Development questions
│   │   ├── fresh_test.tsv, fresh_no_match.tsv, fresh_cyrillic.tsv          Test questions
│   │   ├── queries.jsonl             All 625 generated cases
│   │   ├── answer_equivalence.json   Answers reviewed as "the same"
│   │   ├── label_review.json         Extra acceptable answers per question
│   │   └── manifest.json             FROZEN settings used at runtime
│   └── results/              Measured outputs
│       ├── report.json                   All numbers from `faq evaluate`
│       ├── *-predictions.jsonl           Per-case results (one file per profile, plus rerank)
│       ├── live-api.json                 Real HTTP check
│       ├── answer-support-experiment.json  A rejected earlier idea
│       └── v1/                           First benchmark, kept unchanged (it had label errors)
│
├── scripts/
│   ├── audit_task1_data.py           Read-only audit of the original archive
│   ├── build_evaluation_cases.py     Builds evaluation/uzbek/queries.jsonl
│   ├── experiment_answer_support.py  Tried and rejected: rescoring with answer vectors
│   └── verify_live_api.py            Checks real HTTP results against offline predictions
│
├── tests/                    Unit tests + one integration test
└── docs/
    ├── GUIDE.md                           This file
    ├── evaluation/uzbek-normalization.md  Quality report (written by `faq evaluate`)
    ├── research/                          Data audit findings, Uzbek spelling sources
    └── design/                            Design documents (task 1, normalization, reranker)
```

Not in git: `.venv/` (packages), `.cache/` (model weights ≈4.5 GB, embedding and reranker score
caches), `.env` (your bot token).

## 5. The data

- `data/source/questions.csv` — 3,048 rows: `id, answer_id, question, is_active, created_at, updated_at`.
- `data/source/answers.csv` — 505 rows: `id, category_id, topic_id, answer, status, created_at, updated_at`.

Facts from the audit (`docs/research/task1-data-findings.md`):

- All rows are active; every question points to an existing answer; 9 categories.
- Most answers have exactly 6 question variants (3 answers have 12). The variants follow templates
  such as "… haqida ma'lumot bering", so they are not independent real user questions.
- All apostrophes in the data are plain ASCII `'`. Real users type `ʻ`, `ʼ`, `‘`, `’` or `` ` ``.
  This mismatch is the main reason for normalization (§8).
- Answer 499 has one Cyrillic letter inside a Latin word. It is kept as is and only flagged.
- Some answers mean the same thing (181/279, 380/494). Some only overlap (22/144).

An Excel file also works: one `.xlsx` with sheets `questions` and `answers`, same columns.

## 6. Import: from files to a searchable index

Command: `uv run faq import data/source` (or a path to an `.xlsx`). Code: `cli.py → import_data()`.

1. **Validate everything first** (`data.py → load_dataset`): required columns, unique positive IDs,
   every `answer_id` exists, dates parse, `is_active` is true/false, `status` is active/inactive,
   text is not empty and has no control characters. Any error stops the import and names the file,
   sheet and row. **Nothing touches the databases until every row passes.** Warnings (mixed-script
   words, questions that become identical after normalization) are collected, not fatal.
2. **Prepare search text** (`evaluation/text_profiles.prepare_profile`). Each question gets an
   `embedding_text` with the selected profile `uz_surface_v1` (§8). The original text is kept.
3. **Calibration guard** (§12). Data, equivalence file, model, device, profile, thresholds and
   reranker settings must match `evaluation/uzbek/manifest.json`. If anything changed, the import
   refuses (unless `FAQ_ALLOW_UNCALIBRATED=1`).
4. **Embed** (`embeddings.Encoder`). BGE-M3 loads once. A text longer than 512 tokens is rejected,
   never cut. Vectors are normalized to length 1 and cached in `.cache/embeddings/`.
5. **Build a new Qdrant collection** named `faq_<random>`. Each point: id = question id, the
   vector, payload `{question_id, answer_id, generation}`. Only active questions with an active
   answer are indexed.
6. **Publish** (`storage.snapshot_lock(publish=True)`):
   - take an exclusive PostgreSQL advisory lock (searches wait);
   - in one transaction, replace all answers and questions and write one `metadata` row
     (generation name, profile, model, digests, device, …);
   - switch the Qdrant alias `faq_questions` to the new collection;
   - release the lock.

Why a "generation" name: PostgreSQL and Qdrant cannot share one transaction, so both record the
same generation name. If they disagree (for example after a crash), search refuses (`/ready` → 503)
instead of mixing old and new data. Running the import again fixes it. Old Qdrant collections stay
for recovery; delete them by hand when you like.

The reranker plays no part in import. It works only at question time.

PostgreSQL tables (schema `faq`, created by `storage.py`):

| Table | Columns |
|---|---|
| `answers` | `id` PK, `category_id`, `topic_id`, `answer`, `status`, `created_at`, `updated_at` |
| `questions` | `id` PK, `answer_id` → answers, `question`, `embedding_text`, `is_active`, `created_at`, `updated_at` |
| `metadata` | single row `id=1`, `value` JSON (generation, profile, model, digests, …) |

## 7. Search: from a question to an answer

Code: `service.py → SearchService.search()`.

1. Reject a question longer than 2,000 characters (HTTP 422).
2. Normalize it with the **same** profile used at import (§8).
3. Take a **shared** PostgreSQL lock: many searches can run together; an import waits for them.
4. `assert_ready()`: the published metadata must match the running settings, and the Qdrant alias
   must point to the same generation. Otherwise → 503 "not ready".
5. Encode the question with BGE-M3 (one encode at a time).
6. Ask Qdrant for the **top 30** most similar stored questions (exact cosine search).
7. **Group the hits by answer** (`group_answer_candidates`). Six variants of one answer must not
   compete with each other. Reviewed equivalent answers (181/279, 380/494) form one group. Each
   group keeps its best score and its best stored question.
8. **Route** (`route`, §9): confident → matched; very low → no match; unsure → step 9.
9. **Rerank** (only for unsure cases): read the best stored question + its answer for the top 5
   groups from PostgreSQL (`group_texts`), score each with the reranker, and decide again
   (`decide_reranked`, §9).
10. If matched, read the answer (it must exist and be active) and the matched question from
    PostgreSQL. Return them unchanged.
11. Log request id, status, reason, stage and time. The user's text is **never** logged.

Response example:

```json
{
  "status": "matched", "reason": null,
  "answer_id": 1, "score": 0.83, "margin": 0.09, "question_id": 4,
  "stage": "dense", "rerank_score": null, "rerank_margin": null,
  "answer": "…stored answer text…", "matched_question": "…stored question…",
  "message": null, "request_id": "…"
}
```

- `score` / `margin` are always the dense (BGE-M3) values of the top dense group.
- `stage` is `"dense"` or `"rerank"`; `rerank_score` / `rerank_margin` are filled only when the
  reranker decided.
- For no match: `status: "no_match"`, a `reason` (§9) and
  `message: "Savolingizga ishonchli mos javob topilmadi. Iltimos, savolni aniqroq yozing."`

## 8. Uzbek text normalization

Code: `text/normalization.py`. Profile name: `uz_surface_v1`. Goal: make *equal spellings equal*
without changing *meaning*.

| Input | Becomes | Why |
|---|---|---|
| `Oʻqish`, `O‘qish`, `O’qish`, ``O`qish`` | `O'qish` | Same Uzbek letter typed differently; the data uses `'` |
| `maʼlumot` | `ma'lumot` | Same for the tutuq sign (ʼ) |
| `ID‑karta` (special hyphen) | `ID-karta` | Same hyphen, different code point |
| BOM, tabs, double or non-breaking spaces | one normal space, trimmed | Invisible differences |
| composed/decomposed Unicode forms | NFC | Same letter, different encoding |

Deliberately **not** changed: letter case, numbers, negation (`emas`), suffixes, domains
(`my.gov.uz`), acronyms, quotes outside letter context, Cyrillic letters, typos, missing
apostrophes. Changing these could change meaning, so they are measured instead (§14).

Apostrophes are replaced only **in letter context**: between two Latin letters, or right after
o/O/g/G (for example word-final `bog‘`). Quotes around a phrase like `'hayotiy vaziyat'` stay.

Each text has three forms: the original (stored and shown), `embedding_text` (what the models see)
and `comparison_key` (lower case; only for duplicate checks and the string baseline).

On the supplied data this profile changes **zero** stored questions; it matters for user input.
Two other profiles were compared: `raw` (no change) and `uz_surface_casefold_v1` (also lower-cases).

## 9. The decision: answer or refuse

Code: `search/answer_groups.py` → `route()`, `decide()`, `decide_reranked()`.

**Stage 1 — dense (BGE-M3).** `s1` = best group score, `s2` = second best (cosine similarity).

| Condition | Result | Reason |
|---|---|---|
| `s1 ≥ 0.76` and `s1 − s2 ≥ 0.04` | matched, stage `dense` | — |
| `s1 < 0.62` | no match | `low_similarity` |
| anything else (ambiguous, or `0.62 ≤ s1 < 0.76`) | go to stage 2 | — |

**Stage 2 — reranker.** The top 5 answer groups are scored. Each pair is (user question, best
stored question + newline + its answer). The score is the sigmoid of the reranker output, from 0
to 1. `r1` = best reranker score, `r2` = second best.

| Condition | Result | Reason |
|---|---|---|
| `r1 ≥ 0.75` and `r1 − r2 ≥ 0.22` | matched, stage `rerank` (the reranker's top answer) | — |
| `r1 < 0.75` | no match | `rerank_low_score` |
| `r1 − r2 < 0.22` | no match | `rerank_ambiguous` |

Details:
- With only one group, only the score check applies (no margin).
- Equal reranker scores → the lowest answer ID wins, so results are repeatable.
- A pair longer than 512 tokens is never cut. Reranking is skipped and the dense no-match stays
  (log line `rerank_skipped_length`).
- Scores are **not** probabilities. The values 0.76, 0.04, 0.62, 0.75 and 0.22 were chosen
  automatically on development data (§13) and are frozen in `evaluation/uzbek/manifest.json`
  together with "top 5 groups". The app reads them from there.
- If the reranker is switched off (`rerank_enabled: false`), stage 1 alone decides: everything that
  is not confident is refused with `low_similarity` or `ambiguous`.

Errors are separate from "no match": bad input → 422; database, Qdrant, model or reranker failure,
or a stored text that disappeared → 503. A failure never becomes a silent "no match".

## 10. FastAPI service

Code: `api.py`. Start: `uv run uvicorn faq.api:app --host 127.0.0.1 --port 8000 --workers 1`.

| Endpoint | Meaning |
|---|---|
| `GET /health` | The process is alive |
| `GET /ready` | Data published, generations match, configuration matches → 200; otherwise 503 |
| `POST /search` `{"question": "..."}` | 200 with the result; 422 bad input; 503 busy or unavailable |
| `GET /docs` | Swagger UI to try the API in a browser |

- Both models load once at startup (BGE-M3 ≈7 s with cached weights; the reranker only when
  enabled). With both models the process uses about 3.3 GB of RAM.
- Model work runs in a thread pool, so the event loop stays free.
- At most 2 searches are admitted at once; if no slot frees within 0.2 s → 503 "busy". Reranked
  questions take about 1 s, so this happens sooner under load.
- Use **one worker**: each worker would load its own copy of both models.

## 11. Telegram bot

Code: `bot.py`. Start: `uv run faq bot` (needs `TELEGRAM_BOT_TOKEN` in `.env`, and the API running).

- Long polling, so no public URL is needed.
- `/start` → short Uzbek explanation of what the bot does.
- Text message → `POST /search` → reply with the stored answer or the no-match message.
- Answers longer than Telegram's 4,096-unit limit are split safely (counted in UTF-16 units).
- 422 → asks for a non-empty text under 2,000 characters. API down → "try later".
- Non-text messages (photo, sticker, …) → asks for text.
- The bot loads no model; all logic is in the API. The token and chat text are never logged.

## 12. Configuration and the calibration guard

Settings come from environment variables or `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://faq:faq_local@127.0.0.1:55432/faq` | PostgreSQL |
| `QDRANT_URL` | `http://127.0.0.1:56333` | Qdrant |
| `MODEL_NAME` / `MODEL_REVISION` | `BAAI/bge-m3` / pinned commit | Exact embedding model |
| `RERANKER_NAME` / `RERANKER_REVISION` | `BAAI/bge-reranker-v2-m3` / pinned commit | Exact reranker |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` works but needs a new evaluation |
| `HF_HOME` | `.cache/huggingface` | Where model weights are stored |
| `TELEGRAM_BOT_TOKEN` | empty | Bot token (keep only in `.env`) |
| `FAQ_API_URL` | `http://127.0.0.1:8000` | Where the bot sends questions |
| `FAQ_PROFILE`, `FAQ_MIN_SCORE`, `FAQ_MIN_MARGIN`, `FAQ_TOP_K` | from `manifest.json` | Leave unset |
| `FAQ_RERANK_ENABLED`, `FAQ_RERANK_FLOOR`, `FAQ_RERANK_MIN_SCORE`, `FAQ_RERANK_MIN_MARGIN`, `FAQ_RERANK_GROUPS` | from `manifest.json` | Leave unset |
| `FAQ_ALLOW_UNCALIBRATED` | `0` | `1` = allow unmeasured settings (experiments only) |

URLs use `127.0.0.1`, not `localhost`: on Windows, `localhost` first tries IPv6, which added
seconds of delay per connection.

**Calibration guard** (`config.validate_calibration`): thresholds are valid only for the exact
setup they were measured with — model, device, profile, data, equivalence file and, when enabled,
the exact reranker and its values. Import and `/ready` compare all of these with `manifest.json`
and refuse a mismatch. This stops "changed the model but kept the old thresholds". To run a
different setup on purpose (for example dense only with `FAQ_RERANK_ENABLED=0`), also set
`FAQ_ALLOW_UNCALIBRATED=1`.

## 13. How quality was measured

Command: `uv run faq evaluate` (code: `evaluation/runner.py`). It uses the real models and exact
cosine search, offline; it publishes nothing.

**Test questions** (`evaluation/uzbek/`), written by hand, not copied from the data:

- *Development set* (used to choose all settings): 72 answerable questions (8 per category) + 36
  that must be refused (18 unrelated, 18 "do it for me / check my private data" requests the FAQ
  cannot answer).
- *Test set*: 36 answerable (4 per category) + 18 must-refuse.
- Each question is a **family**. `scripts/build_evaluation_cases.py` makes variants of it: UPPER
  CASE, odd whitespace/BOM, typographic apostrophes, missing apostrophes, small typos, Cyrillic
  spelling, one Cyrillic letter mixed in. Variants stay in their parent's split, so nothing leaks
  between development and test. Total: **625 cases**.
- Each answerable question lists all acceptable answer IDs, decided by reading the full answers
  (`label_review.json`, `answer_equivalence.json`).
- A test question that exactly equals a stored question (after normalization) is rejected.

**Procedure:**

1. Embed all stored questions and all test questions for 3 profiles (`raw`, `uz_surface_v1`,
   `uz_surface_casefold_v1`).
2. **Dense thresholds.** For each profile, try every score from 0.35 to 0.96 and every margin from
   0 to 0.15 **on the development set only** (base and must-refuse questions; variants excluded).
   Goal: precision ≥ 95% and false acceptance ≤ 5%; among those, the most correct answers.
3. Choose the profile and freeze it with its thresholds in `manifest.json`.
4. **Reranker values.** Dense values stay frozen. Every case that is not a confident dense match
   and has `s1 ≥ 0.50` gets reranker scores for its top 5 groups (cached in `.cache/rerank/`).
   Grid search on the development set only — floor 0.50–0.76, reranker score 0.00–0.99, reranker
   margin 0.00–0.50, step 0.01 — with utility `correct − 8 × wrong`; ties prefer fewer reranks.
   An assertion refuses any test row.
5. **Adoption rule** (development only): correct answers must rise, precision ≥ 95%, false
   acceptance ≤ 5%. If the rule fails, `rerank_enabled: false` and the app stays dense only. It
   passed, so the reranker is on.
6. Only then compute results on the test set. No tuning after that.

**Metrics** (`evaluation/metrics.py`):

- *Top-1 accuracy* — is the best-ranked answer correct (before any refusal)?
- *Recall@5* — is a correct answer in the top 5?
- *Accepted precision* — of the answers given, how many were correct?
- *Correct answer coverage* — of answerable questions, how many got a correct answer?
- *False acceptance* — of must-refuse questions, how many got an answer anyway?

**Extra checks:** a character n-gram TF-IDF baseline (is the model better than string matching?),
a "hold out one stored variant" test, number/negation diagnostics, latency, offline vs real Qdrant
parity, and `scripts/verify_live_api.py`, which sends all 54 test families through the real HTTP
API and confirms the same decisions.

**History, kept on purpose:**
- The first benchmark (`evaluation/results/v1/`) missed some valid alternative answers. It was kept
  unchanged; its test questions moved to the development set, and a new test set was written.
- `scripts/experiment_answer_support.py` tried rescoring with answer vectors; it lowered coverage
  and was not adopted.
- The test set was first used for the dense-only system. Its result (47% coverage) is why the
  reranker was built. So for the reranker, the test set is **reused, not fresh**. No value was
  tuned on it, but it is not fully independent evidence (`test_status: "reused"` in the manifest).

## 14. Results

**Final system** (dense + reranker) on the test set, base questions + must-refuse (54 families;
reused, not fresh):

| Pipeline | Accepted precision | Correct coverage | False acceptance | Correct | Wrong |
|---|---:|---:|---:|---:|---:|
| Dense only | 94.4% | 47.2% | 5.6% (1/18) | 17 | 1 |
| **Dense + rerank (used)** | **92.9%** | **72.2%** | **5.6% (1/18)** | **26** | **2** |

The same on the development set (where values were chosen):

| Pipeline | Accepted precision | Correct coverage | False acceptance | Correct | Wrong |
|---|---:|---:|---:|---:|---:|
| Dense only | 97.2% | 48.6% | 2.8% | 35 | 1 |
| Dense + rerank | 98.2% | 77.8% | 2.8% | 56 | 1 |

How to read this:
- The reranker turns many refusals into correct answers: +29.2 points coverage on development,
  +25.0 on test.
- On the test set it also adds one wrong answer, so precision (92.9%) misses the 95% target, and
  false acceptance (5.6%) misses the 5% target, as dense alone did. The test is small: one answer
  changes precision by about 3 points.
- Three reranker values chosen on 108 development families can overfit. The chosen margin 0.22 is
  very close to a few development cases, which is one sign of this.
- 290 of 625 cases take the rerank path at the chosen floor 0.62. None was skipped for length.

**Profiles** (dense only, test set):

| Profile | Top-1 | Accepted precision | Correct coverage | False acceptance |
|---|---:|---:|---:|---:|
| raw | 75.0% | 94.4% | 47.2% | 5.6% |
| **uz_surface_v1 (selected)** | 75.0% | **94.4%** | **47.2%** | **5.6%** |
| uz_surface_casefold_v1 | 75.0% | 90.5% | 52.8% | 5.6% |

**Variants** (test set, answerable questions):

| Variant | Dense top-1 | Final correct coverage | Final precision |
|---|---:|---:|---:|
| base | 75% | 72% | 96% |
| whitespace / BOM | 75% | 72% | 96% |
| typographic apostrophes | 76% | 72% | 95% |
| missing apostrophes | 76% | 76% | 95% |
| one Cyrillic letter mixed in | 72% | 69% | 96% |
| Cyrillic spelling | 67% | 78% | 88% |
| **UPPER CASE** | **28%** | **3%** | 100% |

UPPER CASE is the clear weak point: the case-preserving profile and both models handle it badly.

**Other numbers:**
- Recall@5 is 94.4%: the right answer is usually among the candidates the reranker sees.
- Character TF-IDF baseline: top-1 75%, recall@5 91.7%. The data is very template-like, so plain
  string matching is a strong baseline.
- Latency on CPU: dense path (model + vector search) p50 116 ms / p95 124 ms. Over real HTTP, all
  54 test questions p50 593 ms / p95 1,258 ms; the 27 reranked ones p50 1,083 ms / p95 1,280 ms.
- Memory: about 3.3 GB RAM with both models loaded.
- Live check: the real API gave the same decision as the offline run for all 54 test families,
  including the 27 reranked ones (dense score difference < 1e-6, reranker < 1e-4).

Full numbers: `docs/evaluation/uzbek-normalization.md`, `evaluation/results/report.json`,
`evaluation/results/live-api.json`, and per-case results in `evaluation/results/*-predictions.jsonl`.

## 15. Tests

```powershell
uv run pytest -q                              # unit tests (no services needed)
$env:RUN_INTEGRATION='1'; uv run pytest -q    # also the real PostgreSQL/Qdrant test
```

| Test file | Checks |
|---|---|
| `tests/text/test_normalization.py` | Apostrophe/hyphen/BOM/whitespace rules; repeat-safe; meaning tokens kept |
| `tests/text/test_corpus_lint.py` | Mixed-script and collision reports; the real data keeps its meaning |
| `tests/test_import.py` | Real data loads; bad rows rejected before DB changes; Excel = CSV |
| `tests/search/test_answer_groups.py` | Equivalent answers, overlaps, thresholds, routing, reranked decision, ties |
| `tests/test_rerank.py` | Reranker keeps input order, refuses long pairs (no cutting), rejects NaN, caches |
| `tests/test_service_api.py` | Verbatim answers, 422/503, generations; rerank path, reranker error → 503, long pair → dense |
| `tests/test_calibration_provenance.py` | Changed model, data or reranker values need recalibration; old manifests still work |
| `tests/test_evaluation.py` | Threshold selection, split leakage, metric definitions |
| `tests/test_rerank_calibration.py` | Reranker value search uses development rows only; adoption rule |
| `tests/test_bot.py` | Telegram message splitting and reply text |
| `tests/integration/test_snapshot.py` | Real publish, search and text fetch on PostgreSQL/Qdrant |

Unit tests use small fake models; the real models are checked by `faq evaluate` and
`scripts/verify_live_api.py`. Current result: 97 passed, 1 skipped (the integration test needs
`RUN_INTEGRATION=1`).

## 16. How to run it

Needs: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker. The first run downloads ≈4.5 GB of
model weights (BGE-M3 + reranker). CPU is enough.

```powershell
uv sync --frozen --extra dev                 # install packages
docker compose up -d postgres qdrant         # start the databases
uv run faq import data/source                # load data (first time also downloads BGE-M3)
uv run uvicorn faq.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal:

```powershell
uv run faq search "ID-kartamni yangilash uchun arizani qayerga beraman?"
Invoke-RestMethod -Uri http://127.0.0.1:8000/search -Method Post -ContentType 'application/json; charset=utf-8' -Body '{"question":"ID-karta olishga qaysi hujjatlar kerak?"}'
```

Telegram: copy `.env.example` to `.env`, set `TELEGRAM_BOT_TOKEN`, then `uv run faq bot` (the API
must be running).

Everything in Docker:

```powershell
docker compose --profile app build api
docker compose --profile app run --rm api faq import data/source
docker compose --profile app up -d api
docker compose --profile app --profile bot up -d bot
```

Measure quality again (slow on CPU the first time, including the reranker download; reranker scores
are cached after that):

```powershell
uv run python scripts/build_evaluation_cases.py   # rebuild queries.jsonl
uv run faq evaluate                                # recalibrate + report (overwrites manifest/results)
uv run faq import data/source                      # republish with the selected settings
uv run python scripts/verify_live_api.py           # with the API running: live check
```

Stop: `docker compose down` (keeps data) or `docker compose down -v` (deletes data).

## 17. Limits and next steps

Limits:
- Small benchmark (54 test families); generated variants are correlated, not extra evidence.
- The reranker result is measured on a **reused** test set; a fresh one must confirm it.
- On that test set, precision is 92.9% and false acceptance 5.6% — both just miss the targets.
- UPPER-CASE questions almost always fail; Cyrillic is weaker than clean Latin.
- Reranked questions take about 1 s on CPU; RAM is about 3.3 GB.
- Near-topic requests ("check my application status") can still look like an FAQ question.
- Facts and fees in the answers were not checked for current legal accuracy.
- Import needs a short downtime; no authentication, rate limiting or monitoring.

Next steps (each needs a fresh test set after the change):
1. Collect real user questions, label them independently, and use them as the new test set.
2. Fix upper-case input (for example, restore normal case for all-caps questions) and measure it.
3. Try hybrid dense + sparse search to improve the candidates.
4. Optional Cyrillic → Latin transliteration as an extra candidate query.
5. Production: authentication, rate limits, monitoring, zero-downtime publishing.

## 18. Short answers for the interview

- **How did you understand the task?** Retrieval, not generation: find the stored answer whose
  question means the same as the user's, or say honestly that none fits.
- **Why search questions, not answers?** Users ask questions; the stored variants look like user
  questions, so question-to-question similarity is the most direct signal.
- **Why both PostgreSQL and Qdrant?** Qdrant finds similar vectors; PostgreSQL holds the original
  text and metadata. The Qdrant index can always be rebuilt from the data.
- **Why two stages?** BGE-M3 is fast and settles clear cases. The reranker reads question and
  stored text together, so it judges unsure cases better, but it is slow (about 1 s). Running it
  only for unsure cases keeps most answers fast.
- **How do you avoid wrong answers?** Every threshold is a minimum score plus a minimum lead over
  the next different answer, chosen on development data only and frozen before the test.
- **Why group by answer?** Six variants of one answer would otherwise make the lead look tiny and
  cause false "ambiguous" refusals.
- **Why not lower-case everything?** Measured: lower-casing answered more but with lower precision
  (90.5% vs 94.4%). The case-preserving profile won on development data.
- **How do you know the API behaves like the offline evaluation?** `verify_live_api.py` sent all
  54 test families through the real HTTP stack: same decisions, including every reranked case.
- **What if the data changes?** Re-import; the calibration guard sees the new data and asks for a
  new evaluation before publishing with old thresholds.
- **Biggest weakness?** Upper-case input, and evidence: coverage rose from 47% to 72% with the
  reranker, but only on a reused test set, with 92.9% precision. A fresh test set must confirm it.
