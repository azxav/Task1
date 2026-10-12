# Task 1 semantic FAQ system

## Purpose and requirements

Build the first job application task and make every decision explainable to the candidate. The supplied brief requires BGE-M3, Qdrant, PostgreSQL, FastAPI, a Telegram bot, Excel import, no-match handling, a startup README, and measured search quality. The archive actually contains questions.csv and answers.csv: 3,048 active question variants linked to 505 active answers, with no orphan references. Support these CSVs and an Excel workbook with questions and answers sheets using the same column names. Do not implement Task 2.

The system retrieves curated answers verbatim. It does not generate answers, verify their legal accuracy, or infer missing facts. The dataset includes dated statements; search quality and source correctness are separate concerns.

## Architecture and alternatives

Use one Python application package with separate import, retrieval, API, bot, and evaluation entry points. PostgreSQL is the source of truth; Qdrant is a rebuildable question index. Load BGE-M3 once in the API process using sentence-transformers; use normalized dense embeddings with cosine similarity. Telegram uses HTTP to call the API and never loads a second model.

Start with dense retrieval across question variants. Alternatives are dense-plus-sparse hybrid retrieval and a reranker after retrieval. Both add complexity and latency; introduce them only if measured errors show the baseline needs improvement. Do not add an orchestration framework, generative model, frontend, or authentication subsystem to this demo.

Docker Compose runs PostgreSQL, Qdrant, API, and an optional bot. Import and evaluation run as application commands. Pin dependencies and document model downloads, persistent volumes, CPU operation, and optional GPU setup. Secrets come from environment variables; commit only .env.example. Telegram polling requires a user-supplied bot token.

## Storage

PostgreSQL answers: id primary key, category_id, topic_id, answer text, status, created_at, updated_at.

PostgreSQL questions: id primary key, answer_id foreign key, original question text, normalized question text, is_active, created_at, updated_at. Preserve source IDs and timestamps. Index answer_id.

Qdrant point ID equals question ID. Each point stores a 1,024-dimensional dense vector and payload question_id, answer_id, question text, and index generation. Only active questions whose answers are active are indexed. The model identifier and normalization version form index metadata; changing either requires a rebuild.

## Import and index publication

Uzbek preparation and quality rules are refined by `2026-10-12-uzbek-normalization-design.md`, which takes precedence for text profiles, diagnostics, equivalence-group ambiguity, and evaluation splits. The default embedding profile preserves case; casefold is initially a comparison key and a separately measured experiment.

Accept the two supplied UTF-8 CSV files or one .xlsx workbook with questions and answers sheets. Check required columns, unique IDs, nonempty content, booleans, timestamps, answer status, and references before making changes. Errors include file/sheet and row information. Normalize whitespace, Unicode, and Uzbek apostrophe variants consistently for indexing and queries; retain original text for display. Do not transliterate or remove meaningful punctuation blindly.

For this small dataset, import replaces the complete snapshot rather than merging unspecified rows. Repeated imports are idempotent. Create a new Qdrant collection and batch-embed active questions before publishing it. Take the API offline for publication, replace PostgreSQL rows transactionally, then switch the Qdrant alias to the new generation. Record the generation in PostgreSQL. Start the API only when database and index generations match. A publication failure leaves readiness false and requires rerunning import; never serve mixed generations. Live imports are outside the demo scope. Keep the previous collection until successful publication, then document cleanup.

## Search and decisions

POST /search accepts a nonempty question of at most 2,000 characters. Normalize it, embed it, then request the top 30 question candidates from the published Qdrant collection. Group by answer_id, keeping the maximum score for each answer. Duplicate paraphrases must not appear as separate competing answers.

Let s1 be the best answer-group score and s2 the next distinct group score, if present. Groups use only reviewed globally equivalent answer IDs; otherwise each answer ID is its own group. Return matched only when s1 reaches the calibrated minimum score and, when a second group exists, s1 - s2 reaches the calibrated ambiguity margin. Otherwise return no_match with a neutral Uzbek message. This margin is a heuristic to evaluate, not a probability or correctness guarantee. Candidate count and thresholds are configurable and recorded with evaluation results. Fetch the accepted group's best-scoring stored answer from PostgreSQL and confirm active status before returning it. Missing or inconsistent data is a service error, not a semantic no-match.

Responses contain status, answer_id and answer when matched, matched question, similarity score, and a request ID. Declined responses include a reason such as low_similarity or ambiguous. Do not present scores to Telegram users as confidence percentages. API errors distinguish invalid input from unavailable dependencies. /health checks process liveness; /ready checks database connectivity, published index, generation consistency, and loaded model.

Embedding is CPU-bound. Use bounded admission/concurrency and run model inference outside the async event loop. Start with one API worker to avoid multiplying model memory. Add timeouts for dependency calls. Log timing and decision metadata without recording full user messages by default.

## Telegram flow

Use long polling with a Python Telegram library. /start explains the FAQ scope in Uzbek. Text messages call /search through a reused HTTP client. Display matched answer verbatim, splitting messages at Telegram's text length limit. Display a no-match message for declined queries and a retry-later message for service failures. Non-text messages receive a request to send text. Telegram/API share the same retrieval logic because the bot only calls the API.

## Evaluation

Measure answer-level retrieval and the decision to answer or decline. Create a manually reviewable labeled set with expected answer IDs or expected no-match, including original paraphrases, spelling errors, apostrophe variation, closely related intents, ambiguous questions, and unrelated queries. Accept multiple answer IDs only where human review establishes equivalent answers. Do not assume differently numbered answers are semantically distinct.

Keep calibration and final test cases separate. Tune thresholds only on calibration; report final results without retuning. Author test paraphrases independently and flag exact normalized overlaps with indexed questions. Add a separate held-out variant experiment by excluding selected supplied variants from the evaluation index; keep all production variants in the production index. This experiment measures paraphrase retrieval, not generalization to unseen FAQ answers.

Compare a simple lexical baseline with dense retrieval. Report top-1 answer accuracy and recall at 5 before rejection; accepted-answer precision, answer coverage, false acceptance on no-match cases, and rejection on answerable cases after rejection. Report counts alongside percentages, plus warm query latency p50/p95 and separate cold model startup time. Save per-case predictions, scores, reason, expected labels, and failures. No quality claims or thresholds are valid until actual measurements run.

## Verification and delivery

Use focused tests for validation failures, normalization, grouping by distinct answer, score/margin boundaries, service errors, and API response behavior. Test import idempotency and actual PostgreSQL/Qdrant retrieval with the supplied files. Verify /search and bot formatting; live Telegram verification depends on the bot token. Run evaluation against the real model and index, preserving outputs and hardware context.

Deliver repository, concise README with exact commands, source data provenance, architecture and limitations, evaluation inputs and measured report. Document startup ordering, import failures/recovery, threshold settings, model cache, rebuilding the index, and bot configuration. Do not commit tokens, downloaded model files, database volumes, or fabricated results.

## Implementation sequence

1. Python package, configuration, Compose, relational schema.
2. Validated CSV/Excel import, embeddings, versioned Qdrant index.
3. Shared search logic, rejection rules, API readiness and errors.
4. Labeled evaluation, baseline, threshold calibration, failure analysis.
5. Telegram integration.
6. Real-service verification, final measured report, README.

## Remaining constraints

Hardware availability, model-download access, and a Telegram token are not yet verified. Measure on available hardware before promising latency. Dense multilingual support alone does not establish Uzbek quality. Thresholds cannot guarantee perfect rejection. The current snapshot publication intentionally requires downtime; a production system would need coordinated versioned database and index publication, access control, rate limits, and monitored data updates.
