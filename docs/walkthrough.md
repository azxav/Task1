# Implementation walkthrough

Use this beside the code to explain the submission. Run README commands first.

## Stack responsibilities

| Component | Responsibility | Why needed |
|---|---|---|
| Python 3.12 + uv | Application and locked dependencies | Reproducible local/Docker setup |
| BGE-M3 | Text → normalized 1,024-dimensional dense vector | Retrieves by meaning; never writes answers |
| Qdrant | Cosine search over 3,048 question vectors | Finds likely stored FAQ intents |
| PostgreSQL | Original questions, 505 answers, publication metadata | Authoritative verbatim answer and integrity checks |
| FastAPI | Validation, readiness, search response | One shared service for bot and HTTP clients |
| python-telegram-bot | Telegram polling → HTTP request → plain-text reply | Bot adapter without a second model instance |
| Docker Compose | PostgreSQL, Qdrant, optional application processes | Repeatable service setup and persistent volumes |

Only BGE-M3's dense vector output is used. Its other retrieval modes are not implemented. No LLM generates an answer. The bot token stays in local `.env`.

## Import flow

1. `faq import data/source` reads CSVs. Excel uses named `questions` and `answers` sheets with identical headers.
2. `data.py` validates the entire dataset: unique positive IDs, answer references, active flags, dates and nonempty text. Nothing reaches a database before validation completes.
3. `normalization.py` retains originals, prepares search text and reports diagnostics. This corpus already uses ASCII apostrophes, so the selected surface profile changes zero indexed questions.
4. `embeddings.py` loads the pinned local BGE-M3 revision, checks 512-token limits and embeds active questions. Exact prepared texts/model settings key the cache.
5. `index.py` builds a new Qdrant collection. Points retain question ID, answer ID and generation.
6. `storage.py` takes an exclusive advisory lock, replaces PostgreSQL rows in a transaction, then switches the Qdrant alias. Readers hold a shared lock through search and answer fetch. Two databases cannot share one atomic transaction; generation checks fail closed after a partial publication. Reimport repairs it.

Raw-file hashes track source provenance. A canonical sorted parsed-row digest identifies the same corpus across CSV and Excel. Calibration also checks model revision, text profile, CPU device, reviewed answer groups, top-k and thresholds. Deliberate unmeasured overrides require `FAQ_ALLOW_UNCALIBRATED=1`.

## Query flow

1. Telegram text becomes `POST /search {"question": "..."}`. FastAPI accepts at most 2,000 characters; the model independently enforces 512 tokens.
2. The same selected profile used at import prepares the query. `Oʻqish` and `O'qish` become identical search text; numbers, negation and suffixes remain. Originals and displayed answers are preserved. No global transliteration, stemming or spelling correction.
3. One BGE instance encodes the query outside the async event loop. A model lock and bounded admission limit concurrent work.
4. Qdrant returns the top 30 question candidates. Exact cosine search suits this small dataset; approximate-index tuning is unnecessary here.
5. Candidates form distinct answer groups. Only reviewed globally equivalent IDs merge: 181/279 and 380/494. Multiple question variants pointing to one answer do not create artificial ambiguity.
6. Select the best group when score ≥ 0.76 and the next distinct group's gap ≥ 0.04. Otherwise return `no_match` with low-similarity or ambiguity reason. Scores are not probabilities.
7. Fetch the active PostgreSQL answer by original ID and return it unchanged. Telegram uses plain text, splitting long messages safely. Invalid input is 422; dependency/configuration failure is 503, distinct from ordinary no-match.

`/health` means API alive. `/ready` additionally checks the published dataset, index generation and configuration. Run one API worker because each worker would load another large model.

## Quality decisions and limits

The original benchmark omitted valid alternative IDs. Its artifacts remain in `evaluation/results/v1`; inspected test families became development data. Version 2 calibrates on 108 independent development families and measures 54 fresh test families. Synthetic variants stay with their parent and do not influence threshold selection.

Selected configuration preserves case. Casefold and answer-vector reranking were measured; neither improved correct accepted coverage enough to justify changing runtime. Missing apostrophes, typography, casing, Cyrillic and mixed-script variants have separate results. Numeric/negation diagnostics verify distinct token sequences; they do not prove semantic understanding.

Fresh base test: 17 correct accepted answers, one falsely accepted unsupported request, 19 rejected answerable questions. Accepted precision 94.4%; correct answer coverage 47.2%. Targets were missed slightly and coverage remains limited. Corpus templates, close neighboring intents and incomplete label alternatives constrain the benchmark. Source facts and fees have not been checked for current legal accuracy.

A defensible next iteration would collect realistic user questions, independently adjudicate labels, then compare broader candidate retrieval or a dedicated reranker on development data. It needs another fresh held-out test after changes. See `docs/evaluation/uzbek-normalization.md` for measured counts and latency.
