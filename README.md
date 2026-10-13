# Uzbek semantic FAQ

Task 1: a user asks a question in their own words; the system finds the stored FAQ answer with the
same meaning, or says honestly that it has none. Stack: **BGE-M3 → Qdrant → PostgreSQL**, served by
**FastAPI** and a **Telegram bot**. Answers are returned word for word; no LLM, no cloud inference.

**Full explanation of how everything works: [docs/GUIDE.md](docs/GUIDE.md).**

## Run locally

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker. The first run downloads ≈4.5 GB of
model weights (BGE-M3 + reranker). CPU is enough.

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
```

API docs: <http://127.0.0.1:8000/docs>. Endpoints: `POST /search`, `GET /health`, `GET /ready`.

Telegram bot: copy `.env.example` to `.env`, set `TELEGRAM_BOT_TOKEN`, keep the API running, then:

```powershell
uv run faq bot
```

Excel import: `uv run faq import path/to/workbook.xlsx` (sheets `questions` and `answers`, same
columns as the CSVs).

## Everything in Docker

```powershell
docker compose --profile app build api
docker compose --profile app run --rm api faq import data/source
docker compose --profile app up -d api
docker compose --profile app --profile bot up -d bot
```

`docker compose down` stops services and keeps data; `down -v` also deletes it.

## Search quality

Measured on a hand-written, held-out test set (36 answerable + 18 must-refuse questions), with
thresholds chosen beforehand on a separate development set:

| Accepted precision | Correct answer coverage | False acceptance (must-refuse) | Recall@5 |
|---:|---:|---:|---:|
| 94.4% | 47.2% | 5.6% (1/18) | 94.4% |

The dense stage alone rarely gives a wrong answer but refuses about half of answerable questions.

**Reranker stage (enabled at runtime).** Uncertain dense matches go to a local cross-encoder,
`BAAI/bge-reranker-v2-m3`. It accepts only on its own calibrated score and margin. Same 54 families:

| Pipeline | Accepted precision | Correct answer coverage | False acceptance (must-refuse) |
|---|---:|---:|---:|
| Dense only | 94.4% | 47.2% | 5.6% (1/18) |
| Dense + rerank | 92.9% | 72.2% | 5.6% (1/18) |

The test split is reused, not fresh: it was measured once before the reranker and motivated it.
Reranker values were chosen on development data only. On development data the reranker raised
correct answers from 35 to 56 of 72 at 98.2% precision; on the reused test it gave 26 correct and
2 wrong answers, so precision is below the 95% target; false acceptance stays 5.6%, above
the 5% target, as for dense only. Reranked queries take about 1.1 s over HTTP
on CPU. Details, method and limits: [GUIDE §13–17](docs/GUIDE.md#13-how-quality-was-measured) and
[docs/evaluation/uzbek-normalization.md](docs/evaluation/uzbek-normalization.md).

## Tests

```powershell
uv run pytest -q
$env:RUN_INTEGRATION='1'; uv run pytest -q   # needs the Docker databases running
```

## Repository layout

| Path | Contents |
|---|---|
| `src/faq/` | Application code (import, search, API, bot, evaluation) |
| `data/source/` | Supplied questions/answers CSVs |
| `evaluation/uzbek/` | Test questions, answer-equivalence review, frozen thresholds (`manifest.json`) |
| `evaluation/results/` | Measured results; `v1/` keeps the first, flawed benchmark for transparency |
| `scripts/` | Data audit, test-case builder, live API check, rejected experiment |
| `tests/` | Unit and integration tests |
| `docs/` | Guide, quality report, data research, design documents |
