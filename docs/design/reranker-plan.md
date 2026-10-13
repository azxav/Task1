# Reranker Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send uncertain dense matches (ambiguous, or dense score in a calibrated gray band below 0.76) to a local cross-encoder reranker, and accept only on the reranker's own calibrated score and margin.

**Architecture:** The dense path (BGE-M3 → Qdrant top 30 → answer groups → 0.76 / 0.04) is unchanged. A pure routing function decides matched / rerank / no_match. The service reranks the top 5 groups with `BAAI/bge-reranker-v2-m3`. The evaluation runner calibrates the three new values on development rows only, applies the adoption rule, and writes them to `evaluation/uzbek/manifest.json`, which the runtime and the calibration guard read.

**Tech Stack:** Python 3.12, sentence-transformers `CrossEncoder`, numpy, FastAPI, psycopg 3, pytest, uv.

**Spec:** `docs/design/reranker-design.md`

## Global Constraints

- Reranker model `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, device from `EMBEDDING_DEVICE` (CPU measured), `max_length=512`, sigmoid scores.
- Dense values stay fixed: profile `uz_surface_v1`, `min_score` 0.76, `min_margin` 0.04, `top_k` 30.
- `rerank_groups` = 5. Pair text per group: `stored question + "\n" + its answer`; query side uses the same prepared profile text as dense search.
- Grids: X 0.50–0.76, R_score 0.00–0.99, R_margin 0.00–0.50, step 0.01.
- Selection objective: feasible (precision ≥ 0.95 and false acceptance ≤ 0.05) → utility `correct − 8 × wrong` → correct → higher X → higher R_score → higher R_margin.
- Adoption rule (development only): correct accepted increases vs dense-only, precision ≥ 95%, false acceptance ≤ 5%. Otherwise `rerank_enabled: false`.
- No test-split row may enter selection (assertion). Test split is reported as **"reused, not fresh"**; manifest `test_status: "reused"`.
- Answers are returned verbatim. No truncation of reranker input. Reranker failure → 503, never a silent no_match.
- `bot.py`, import, the Qdrant index, `compose.yaml`, `Dockerfile` do not change. No new runtime dependency; `psutil` is dev-only.
- Never log user text. Ruff line length 110.

Two refinements of the spec, both narrower:
- `Storage.group_texts(question_ids)` takes question IDs only: a group's best question always belongs to the group's best answer, so one JOIN returns both texts.
- `api.py` needs no edit: `SearchService` is built in the API lifespan, and it loads the reranker in its constructor only when enabled.

## Review Focus

- Older manifest with no reranker keys → system must behave exactly as before (Task 1 test).
- Candidate answer missing/inactive between vector search and text fetch → 503, never a wrong answer (Task 4 test).
- Long user query that passes the dense 512-token check but makes a pair longer than 512 tokens → dense no_match, not 422/503 (Task 3 and Task 4 tests).
- Equal reranker scores → deterministic answer by lowest answer ID (Task 2 test).
- Reviewed equivalent answers (380/494) inside the reranked set → the group's best answer is returned, no false ambiguity between equivalents (Task 2 test).

---

### Task 1: Reranker settings and calibration guard

**Files:**
- Modify: `src/faq/config.py` (whole file)
- Test: `tests/test_calibration_provenance.py`

**Interfaces:**
- Produces: `Settings` fields `rerank_enabled: bool`, `reranker_name: str`, `reranker_revision: str`, `rerank_groups: int`, `rerank_floor: float`, `rerank_min_score: float`, `rerank_min_margin: float`; `Settings.calibration_values() -> dict`; constant `RERANKER_REVISION`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_calibration_provenance.py`, change `calibrated()` so its base settings always have the reranker off, and add three tests at the end of the file:

```python
def calibrated():
    settings = replace(Settings.load(), rerank_enabled=False)
    frozen = {
        "profile": settings.profile,
        "model_name": settings.model_name,
        "model_revision": settings.model_revision,
        "device": settings.device,
        "top_k": settings.top_k,
        "min_score": settings.min_score,
        "min_margin": settings.min_margin,
        "registry_digest": "reviewed",
        "corpus_digest": "original",
    }
    return replace(settings, calibration=frozen, allow_uncalibrated=False), {
        "registry_digest": "reviewed",
        "corpus_digest": "original",
    }


RERANK = {
    "rerank_enabled": True,
    "reranker_name": "BAAI/bge-reranker-v2-m3",
    "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "rerank_groups": 5,
    "rerank_floor": 0.7,
    "rerank_min_score": 0.6,
    "rerank_min_margin": 0.1,
}


def test_manifest_without_reranker_keys_keeps_dense_pipeline_calibrated():
    settings, meta = calibrated()
    assert "rerank_enabled" not in settings.calibration
    assert settings.validate_calibration(meta) == "calibrated"
    with pytest.raises(ValueError, match="rerank_enabled"):
        replace(settings, rerank_enabled=True).validate_calibration(meta)


@pytest.mark.parametrize(
    "change",
    [
        {"reranker_revision": "different"},
        {"rerank_floor": 0.65},
        {"rerank_min_score": 0.7},
        {"rerank_min_margin": 0.2},
        {"rerank_groups": 3},
    ],
)
def test_changed_reranker_cannot_reuse_calibrated_values(change):
    settings, meta = calibrated()
    settings = replace(settings, calibration={**settings.calibration, **RERANK}, **RERANK)
    assert settings.validate_calibration(meta) == "calibrated"
    with pytest.raises(ValueError, match="calibration"):
        replace(settings, **change).validate_calibration(meta)


def test_reranker_switch_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("FAQ_RERANK_ENABLED", "0")
    assert Settings.load().rerank_enabled is False
    monkeypatch.setenv("FAQ_RERANK_ENABLED", "1")
    assert Settings.load().rerank_enabled is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_calibration_provenance.py -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'rerank_enabled'`.

- [ ] **Step 3: Implement**

Replace `src/faq/config.py` with:

```python
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# Keys missing from older manifests mean these values.
CALIBRATION_DEFAULTS = {"rerank_enabled": False}


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    qdrant_url: str
    model_name: str
    model_revision: str
    device: str
    profile: str
    min_score: float
    min_margin: float
    top_k: int
    registry_path: str
    token: str = field(repr=False)
    api_url: str
    batch_size: int = 16
    calibration: dict = field(default_factory=dict, repr=False)
    allow_uncalibrated: bool = False
    rerank_enabled: bool = False
    reranker_name: str = RERANKER_NAME
    reranker_revision: str = RERANKER_REVISION
    rerank_groups: int = 5
    rerank_floor: float = 0.76
    rerank_min_score: float = 0.5
    rerank_min_margin: float = 0.1

    def calibration_values(self):
        values = {
            "profile": self.profile,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "rerank_enabled": self.rerank_enabled,
        }
        if self.rerank_enabled:
            values.update(
                reranker_name=self.reranker_name,
                reranker_revision=self.reranker_revision,
                rerank_groups=self.rerank_groups,
                rerank_floor=self.rerank_floor,
                rerank_min_score=self.rerank_min_score,
                rerank_min_margin=self.rerank_min_margin,
            )
        return values

    def validate_calibration(self, metadata):
        expected = {
            **self.calibration_values(),
            "registry_digest": metadata.get("registry_digest"),
            "corpus_digest": metadata.get("corpus_digest"),
        }
        mismatches = [
            key
            for key, value in expected.items()
            if self.calibration.get(key, CALIBRATION_DEFAULTS.get(key)) != value
        ]
        if mismatches and not self.allow_uncalibrated:
            raise ValueError(
                "Frozen calibration mismatch: "
                + ", ".join(mismatches)
                + "; evaluate this pipeline or set FAQ_ALLOW_UNCALIBRATED=1 for an explicit experiment"
            )
        return "uncalibrated" if mismatches else "calibrated"

    @classmethod
    def load(cls):
        load_dotenv(override=False)
        manifest_path = Path(os.getenv("FAQ_MANIFEST", "evaluation/uzbek/manifest.json"))
        frozen = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        min_score = float(os.getenv("FAQ_MIN_SCORE", frozen.get("min_score", 0.75)))
        rerank_switch = os.getenv("FAQ_RERANK_ENABLED")
        return cls(
            os.getenv("DATABASE_URL", "postgresql://faq:faq_local@127.0.0.1:55432/faq"),
            os.getenv("QDRANT_URL", "http://127.0.0.1:56333"),
            os.getenv("MODEL_NAME", "BAAI/bge-m3"),
            os.getenv("MODEL_REVISION", MODEL_REVISION),
            os.getenv("EMBEDDING_DEVICE", "cpu"),
            os.getenv("FAQ_PROFILE", frozen.get("profile", "uz_surface_v1")),
            min_score,
            float(os.getenv("FAQ_MIN_MARGIN", frozen.get("min_margin", 0.03))),
            int(os.getenv("FAQ_TOP_K", frozen.get("top_k", 30))),
            os.getenv("FAQ_REGISTRY", "evaluation/uzbek/answer_equivalence.json"),
            os.getenv("TELEGRAM_BOT_TOKEN", ""),
            os.getenv("FAQ_API_URL", "http://127.0.0.1:8000"),
            int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            frozen,
            os.getenv("FAQ_ALLOW_UNCALIBRATED", "0") == "1",
            rerank_enabled=(
                frozen.get("rerank_enabled", False) if rerank_switch is None else rerank_switch == "1"
            ),
            reranker_name=os.getenv("RERANKER_NAME", RERANKER_NAME),
            reranker_revision=os.getenv("RERANKER_REVISION", RERANKER_REVISION),
            rerank_groups=int(os.getenv("FAQ_RERANK_GROUPS", frozen.get("rerank_groups", 5))),
            rerank_floor=float(os.getenv("FAQ_RERANK_FLOOR", frozen.get("rerank_floor", min_score))),
            rerank_min_score=float(os.getenv("FAQ_RERANK_MIN_SCORE", frozen.get("rerank_min_score", 0.5))),
            rerank_min_margin=float(
                os.getenv("FAQ_RERANK_MIN_MARGIN", frozen.get("rerank_min_margin", 0.1))
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass (72 old + 7 new), 1 skipped.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/faq/config.py tests/test_calibration_provenance.py
git commit -m "feat: add reranker settings to calibration guard"
```

---

### Task 2: Routing and reranked decision

**Files:**
- Modify: `src/faq/search/answer_groups.py` (`Decision` dataclass; append three functions)
- Test: `tests/search/test_answer_groups.py`

**Interfaces:**
- Consumes: `AnswerGroup(group_id, member_answer_ids, best_answer_id, best_score, question_id)` (existing).
- Produces:
  - `Decision` gains `stage: str = "dense"`, `rerank_score: float | None = None`, `rerank_margin: float | None = None`.
  - `route_scores(score, margin, *, min_score, min_margin, floor) -> str` returning `"matched" | "rerank" | "no_match"`.
  - `route(groups, *, min_score, min_margin, floor) -> str`.
  - `decide_reranked(groups, scores, *, min_score, min_margin) -> Decision`. `score`/`margin` keep the dense values of the top dense group; `rerank_score`/`rerank_margin` hold reranker values.

- [ ] **Step 1: Write the failing tests**

In `tests/search/test_answer_groups.py`, change the import line to:

```python
from faq.search.answer_groups import (
    AnswerGroup,
    EquivalenceRegistry,
    ScoredQuestion,
    decide,
    decide_reranked,
    group_answer_candidates,
    route,
)
```

Append:

```python
def groups_of(*items):
    """(answer_id, question_id, dense_score) in dense order."""
    return [AnswerGroup(a, (a,), a, s, q) for a, q, s in items]


ROUTE = {"min_score": 0.76, "min_margin": 0.04, "floor": 0.62}


def test_route_boundaries():
    assert route(groups_of((1, 10, 0.76), (2, 20, 0.72)), **ROUTE) == "matched"
    assert route(groups_of((1, 10, 0.80), (2, 20, 0.79)), **ROUTE) == "rerank"
    assert route(groups_of((1, 10, 0.62)), **ROUTE) == "rerank"
    assert route(groups_of((1, 10, 0.6199)), **ROUTE) == "no_match"
    assert route([], **ROUTE) == "no_match"
    with pytest.raises(ValueError):
        route(groups_of((1, 10, 0.7)), min_score=0.76, min_margin=0.04, floor=0.8)


def test_rerank_decision_boundaries_keep_dense_context():
    groups = groups_of((1, 10, 0.74), (2, 20, 0.73), (3, 30, 0.70))
    decision = decide_reranked(groups, [0.2, 0.9, 0.5], min_score=0.9, min_margin=0.39)
    assert (decision.status, decision.answer_id, decision.question_id, decision.stage) == (
        "matched",
        2,
        20,
        "rerank",
    )
    assert decision.score == 0.74 and decision.margin == pytest.approx(0.01)
    assert decision.rerank_score == 0.9 and decision.rerank_margin == pytest.approx(0.4)
    low = decide_reranked(groups, [0.2, 0.9, 0.5], min_score=0.91, min_margin=0)
    assert (low.status, low.reason, low.answer_id) == ("no_match", "rerank_low_score", None)
    close = decide_reranked(groups, [0.2, 0.9, 0.5], min_score=0.5, min_margin=0.41)
    assert (close.status, close.reason) == ("no_match", "rerank_ambiguous")


def test_rerank_single_group_ties_equivalents_and_invalid_scores():
    single = decide_reranked(groups_of((7, 70, 0.7)), [0.6], min_score=0.5, min_margin=0.9)
    assert single.status == "matched" and single.rerank_margin is None
    tie = decide_reranked(groups_of((9, 90, 0.7), (4, 40, 0.69)), [0.8, 0.8], min_score=0.5, min_margin=0)
    assert tie.answer_id == 4 and tie.reason is None
    equivalent = [AnswerGroup(380, (380, 494), 494, 0.71, 5), AnswerGroup(22, (22,), 22, 0.70, 6)]
    assert decide_reranked(equivalent, [0.9, 0.2], min_score=0.5, min_margin=0.1).answer_id == 494
    with pytest.raises(ValueError):
        decide_reranked(groups_of((1, 10, 0.7)), [float("nan")], min_score=0.5, min_margin=0)
    with pytest.raises(ValueError):
        decide_reranked(groups_of((1, 10, 0.7)), [0.5, 0.4], min_score=0.5, min_margin=0)
    with pytest.raises(ValueError):
        decide_reranked([], [], min_score=0.5, min_margin=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/search/test_answer_groups.py -q`
Expected: FAIL with `ImportError: cannot import name 'decide_reranked'`.

- [ ] **Step 3: Implement**

In `src/faq/search/answer_groups.py`, replace the `Decision` dataclass with:

```python
@dataclass(frozen=True)
class Decision:
    status: str
    reason: str | None
    answer_id: int | None = None
    score: float | None = None
    margin: float | None = None
    question_id: int | None = None
    stage: str = "dense"
    rerank_score: float | None = None
    rerank_margin: float | None = None
```

Append at the end of the file:

```python
def route_scores(score, margin, *, min_score: float, min_margin: float, floor: float) -> str:
    """Dense routing: confident accept, reject below the floor, otherwise send to the reranker."""
    if not floor <= min_score:
        raise ValueError("Rerank floor must not exceed the dense threshold")
    if score is None or score < floor:
        return "no_match"
    if score >= min_score and (margin is None or margin >= min_margin):
        return "matched"
    return "rerank"


def route(groups, *, min_score: float, min_margin: float, floor: float) -> str:
    if not groups:
        return "no_match"
    margin = groups[0].best_score - groups[1].best_score if len(groups) > 1 else None
    return route_scores(groups[0].best_score, margin, min_score=min_score, min_margin=min_margin, floor=floor)


def decide_reranked(groups, scores, *, min_score: float, min_margin: float) -> Decision:
    """Second-stage decision over dense-ordered groups; reranker scores are aligned with `groups`."""
    if not (0 <= min_score <= 1 and 0 <= min_margin <= 1):
        raise ValueError("Invalid reranker threshold or margin")
    if not groups or len(groups) != len(scores):
        raise ValueError("Each reranked group needs exactly one score")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Non-finite rerank score")
    order = sorted(
        range(len(groups)), key=lambda i: (-scores[i], groups[i].best_answer_id, groups[i].question_id)
    )
    best, top = groups[order[0]], scores[order[0]]
    gap = top - scores[order[1]] if len(order) > 1 else None
    context = {
        "score": groups[0].best_score,
        "margin": groups[0].best_score - groups[1].best_score if len(groups) > 1 else None,
        "stage": "rerank",
        "rerank_score": top,
        "rerank_margin": gap,
    }
    if top < min_score:
        return Decision("no_match", "rerank_low_score", **context)
    if gap is not None and gap < min_margin:
        return Decision("no_match", "rerank_ambiguous", **context)
    return Decision("matched", None, best.best_answer_id, question_id=best.question_id, **context)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass, 1 skipped.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/faq/search/answer_groups.py tests/search/test_answer_groups.py
git commit -m "feat: route uncertain dense matches and decide on reranker scores"
```

---

### Task 3: Reranker model wrapper

**Files:**
- Create: `src/faq/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `Settings.reranker_name`, `reranker_revision`, `device` (Task 1).
- Produces: `class PairTooLong(ValueError)`; `class Reranker` with `__init__(settings)`, `startup_seconds: float`, `score(query: str, texts: list[str], *, cache: bool = False) -> list[float]` (sigmoid, aligned with `texts`). Cache directory `.cache/rerank/` (already git-ignored via `.cache/`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rerank.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest

from faq.rerank import PairTooLong, Reranker


class FakeTokenizer:
    def __call__(self, query, text, add_special_tokens=True):
        return {"input_ids": list(range(len(query.split()) + len(text.split()) + 3))}


class FakeModel:
    tokenizer = FakeTokenizer()

    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def predict(self, pairs, **kwargs):
        self.calls.append(pairs)
        return np.array(self.scores[: len(pairs)], dtype=np.float32)


def make(scores):
    reranker = Reranker.__new__(Reranker)
    reranker.model = FakeModel(scores)
    reranker.settings = SimpleNamespace(reranker_name="m", reranker_revision="r", device="cpu")
    return reranker


def test_scores_follow_input_order():
    reranker = make([0.25, 0.75])
    assert reranker.score("savol", ["a", "b"]) == pytest.approx([0.25, 0.75])
    assert reranker.model.calls == [[("savol", "a"), ("savol", "b")]]
    assert reranker.score("savol", []) == []


def test_long_pair_is_rejected_not_truncated():
    reranker = make([0.5])
    with pytest.raises(PairTooLong):
        reranker.score("so'z " * 300, ["javob " * 300])
    assert reranker.model.calls == []


def test_non_finite_score_is_an_error():
    with pytest.raises(ValueError, match="Non-finite"):
        make([float("nan")]).score("savol", ["a"])


def test_evaluation_cache_reuses_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reranker = make([0.4])
    assert reranker.score("savol", ["a"], cache=True) == pytest.approx([0.4])
    assert reranker.score("savol", ["a"], cache=True) == pytest.approx([0.4])
    assert len(reranker.model.calls) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rerank.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'faq.rerank'`.

- [ ] **Step 3: Implement**

Create `src/faq/rerank.py`:

```python
"""Local cross-encoder for uncertain dense matches; pinned model, never truncates input."""

import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter


class PairTooLong(ValueError):
    """Query plus stored text exceeds the reranker's 512-token input."""


class Reranker:
    max_length = 512

    def __init__(self, settings):
        os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
        import torch
        from sentence_transformers import CrossEncoder

        started = perf_counter()
        self.model = CrossEncoder(
            settings.reranker_name,
            revision=settings.reranker_revision,
            device=settings.device,
            cache_folder=os.environ["HF_HOME"],
            max_length=self.max_length,
            activation_fn=torch.nn.Sigmoid(),
        )
        self.settings = settings
        self.startup_seconds = perf_counter() - started

    def score(self, query: str, texts: list[str], *, cache: bool = False) -> list[float]:
        for text in texts:
            if len(self.model.tokenizer(query, text, add_special_tokens=True)["input_ids"]) > self.max_length:
                raise PairTooLong("Query and stored text exceed the reranker's 512-token limit")
        if not texts:
            return []
        key = hashlib.sha256(
            json.dumps(
                [
                    self.settings.reranker_name,
                    self.settings.reranker_revision,
                    self.settings.device,
                    self.max_length,
                    query,
                    texts,
                ],
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        path = Path(".cache/rerank") / f"{key}.json"
        if cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        scores = [
            float(value)
            for value in self.model.predict(
                [(query, text) for text in texts], batch_size=8, show_progress_bar=False, convert_to_numpy=True
            )
        ]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("Non-finite rerank score")
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(scores), encoding="utf-8")
            temporary.replace(path)
        return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass, 1 skipped.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests
git add src/faq/rerank.py tests/test_rerank.py
git commit -m "feat: add pinned local cross-encoder reranker"
```

---

### Task 4: Candidate texts and service rerank branch

**Files:**
- Modify: `src/faq/storage.py` (append method `group_texts`)
- Modify: `src/faq/service.py` (imports, `__init__`, new `_decide`, `search`)
- Test: `tests/test_service_api.py`, `tests/integration/test_snapshot.py`

**Interfaces:**
- Consumes: `route`, `decide_reranked`, `Decision.stage` (Task 2); `Reranker`, `PairTooLong` (Task 3); reranker settings (Task 1).
- Produces: `Storage.group_texts(question_ids: list[int]) -> dict[int, str]` (value `question + "\n" + answer`; raises `RuntimeError("Rerank candidate missing or inactive")`); `SearchService(settings, *, storage=None, index=None, encoder=None, registry=None, reranker=None)`; `/search` response gains `stage`, `rerank_score`, `rerank_margin`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_service_api.py`:

1. Add `from faq.rerank import PairTooLong` to the imports.
2. Add this method to `MemoryStore`:

```python
    def group_texts(self, question_ids):
        if getattr(self, "missing", False):
            raise RuntimeError("Rerank candidate missing or inactive")
        return {ident: f"Savol {ident}\nJavob {ident}" for ident in question_ids}
```

3. Replace `make_service` with:

```python
def make_service(score=0.9, *, index=None, reranker=None, **overrides):
    settings = replace(
        Settings.load(),
        profile="uz_surface_v1",
        min_score=0.8,
        min_margin=0.02,
        allow_uncalibrated=True,
        rerank_enabled=False,
        **overrides,
    )
    registry = EquivalenceRegistry("v1", ())
    meta = {
        "generation": "generation1",
        "profile": settings.profile,
        "model_name": settings.model_name,
        "model_revision": settings.model_revision,
        "registry_digest": registry.digest,
        "embedding_device": settings.device,
        "max_tokens": 512,
    }
    return SearchService(
        settings,
        storage=MemoryStore(meta),
        index=index or FixedIndex(score),
        encoder=RecordingEncoder(),
        registry=registry,
        reranker=reranker,
    )
```

4. Append:

```python
class BandIndex(FixedIndex):
    def search(self, vector, **kwargs):
        return [ScoredQuestion(1, 2, 0.70), ScoredQuestion(3, 4, 0.69)]


class FakeReranker:
    def __init__(self, scores=None, error=None):
        self.scores, self.error, self.calls = scores, error, []

    def score(self, query, texts):
        self.calls.append((query, texts))
        if self.error:
            raise self.error
        return self.scores


RERANK = {"rerank_enabled": True, "rerank_floor": 0.6, "rerank_min_score": 0.5, "rerank_min_margin": 0.1}


def test_uncertain_match_is_reranked_and_answer_stays_verbatim():
    reranker = FakeReranker([0.2, 0.9])
    service = make_service(index=BandIndex(), reranker=reranker, **RERANK)
    with TestClient(create_app(service)) as client:
        body = client.post("/search", json={"question": "Oʻqish qanday?"}).json()
    assert (body["status"], body["stage"], body["answer_id"]) == ("matched", "rerank", 4)
    assert body["answer"] == "Stored answer, unchanged."
    assert body["score"] == pytest.approx(0.70) and body["rerank_score"] == 0.9
    assert reranker.calls == [("O'qish qanday?", ["Savol 1\nJavob 1", "Savol 3\nJavob 3"])]


def test_reranker_failure_is_unavailable_not_no_match():
    service = make_service(index=BandIndex(), reranker=FakeReranker(error=RuntimeError("boom")), **RERANK)
    with TestClient(create_app(service)) as client:
        assert client.post("/search", json={"question": "savol"}).status_code == 503


def test_missing_rerank_candidate_is_unavailable():
    service = make_service(index=BandIndex(), reranker=FakeReranker([0.9, 0.1]), **RERANK)
    service.storage.missing = True
    with pytest.raises(ServiceUnavailable):
        service.search("savol")


def test_too_long_pair_keeps_dense_no_match():
    service = make_service(index=BandIndex(), reranker=FakeReranker(error=PairTooLong("long")), **RERANK)
    result = service.search("savol")
    assert (result["status"], result["reason"], result["stage"]) == ("no_match", "low_similarity", "dense")


def test_below_floor_and_confident_matches_skip_reranker():
    reranker = FakeReranker([0.9])
    assert make_service(0.5, reranker=reranker, **RERANK).search("savol")["reason"] == "low_similarity"
    assert make_service(0.9, reranker=reranker, **RERANK).search("savol")["stage"] == "dense"
    assert reranker.calls == []


def test_disabled_reranker_keeps_dense_behavior():
    service = make_service(index=BandIndex())
    result = service.search("savol")
    assert (result["status"], result["reason"], result["stage"]) == ("no_match", "low_similarity", "dense")
    assert result["rerank_score"] is None and service.reranker is None
```

In `tests/integration/test_snapshot.py`, inside `test_real_snapshot_and_index_publication`, directly after the line `assert store.answer(499).answer == dataset.answers[499].answer`, insert:

```python
    first = dataset.indexed[0]
    expected = first.question + "\n" + dataset.answers[first.answer_id].answer
    assert store.group_texts([first.id]) == {first.id: expected}
    with pytest.raises(RuntimeError, match="Rerank"):
        store.group_texts([10**9])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_service_api.py -q`
Expected: FAIL with `TypeError: SearchService.__init__() got an unexpected keyword argument 'reranker'`.

- [ ] **Step 3: Implement storage**

Append to class `Storage` in `src/faq/storage.py`:

```python
    def group_texts(self, question_ids):
        """Question text plus its answer per stored question, for second-stage reranking."""
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT q.id, q.question, a.answer, a.status FROM questions q "
                "JOIN answers a ON a.id = q.answer_id WHERE q.id = ANY(%s)",
                (list(question_ids),),
            ).fetchall()
        texts = {row["id"]: row["question"] + "\n" + row["answer"] for row in rows if row["status"] == "active"}
        if any(ident not in texts for ident in question_ids):
            raise RuntimeError("Rerank candidate missing or inactive")
        return texts
```

- [ ] **Step 4: Implement the service branch**

In `src/faq/service.py`, replace the line `from .search.answer_groups import EquivalenceRegistry, decide, group_answer_candidates` with:

```python
from .rerank import PairTooLong, Reranker
from .search.answer_groups import (
    EquivalenceRegistry,
    decide,
    decide_reranked,
    group_answer_candidates,
    route,
)
```

Replace `__init__` with:

```python
    def __init__(self, settings, *, storage=None, index=None, encoder=None, registry=None, reranker=None):
        self.settings = settings
        self.storage = storage if storage is not None else Storage(settings.database_url)
        self.index = index if index is not None else VectorIndex(settings.qdrant_url)
        self.registry = registry if registry is not None else EquivalenceRegistry.load(settings.registry_path)
        self.encoder = encoder if encoder is not None else Encoder(settings)
        if reranker is None and settings.rerank_enabled:
            reranker = Reranker(settings)
        self.reranker = reranker
        self.model_lock = threading.Lock()
        self.rerank_lock = threading.Lock()
```

Add this method directly after `search_prepared`:

```python
    def _decide(self, text, groups):
        s = self.settings
        dense = decide(groups, min_score=s.min_score, min_margin=s.min_margin)
        if not s.rerank_enabled:
            return dense
        if route(groups, min_score=s.min_score, min_margin=s.min_margin, floor=s.rerank_floor) != "rerank":
            return dense
        top = groups[: s.rerank_groups]
        texts = self.storage.group_texts([group.question_id for group in top])
        try:
            with self.rerank_lock:
                scores = self.reranker.score(text, [texts[group.question_id] for group in top])
        except PairTooLong:
            logger.info("rerank_skipped_length")
            return dense
        return decide_reranked(top, scores, min_score=s.rerank_min_score, min_margin=s.rerank_min_margin)
```

In `search`, replace

```python
                decision = decide(
                    group_answer_candidates(candidates, self.registry),
                    min_score=self.settings.min_score,
                    min_margin=self.settings.min_margin,
                )
```

with

```python
                decision = self._decide(text, group_answer_candidates(candidates, self.registry))
```

and replace the `logger.info(...)` call after the `with` block with:

```python
            logger.info(
                "search request_id=%s status=%s reason=%s stage=%s elapsed_ms=%.1f",
                request_id,
                decision.status,
                decision.reason,
                decision.stage,
                1000 * (perf_counter() - started),
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass, 1 skipped.

If Docker is running (`docker compose up -d postgres qdrant`), also run:
`$env:RUN_INTEGRATION='1'; uv run pytest tests/integration -q` → 1 passed. Then `Remove-Item Env:RUN_INTEGRATION`.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/faq/storage.py src/faq/service.py tests/test_service_api.py tests/integration/test_snapshot.py
git commit -m "feat: rerank uncertain matches in search service"
```

---

### Task 5: Development-only reranker calibration

**Files:**
- Create: `src/faq/evaluation/rerank_calibration.py`
- Modify: `src/faq/evaluation/runner.py` (`rank_queries` only)
- Test: `tests/test_rerank_calibration.py`, `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `AnswerGroup`, `decide_reranked`, `route_scores` (Task 2); `metrics` (existing).
- Produces:
  - `rank_queries(query_vectors, document_vectors, questions, registry, *, limit=30, groups_limit=5)`; each row gains `top_groups: list[{"answer_id": int, "question_id": int, "score": float}]` (dense order).
  - `LOWEST_FLOOR = 0.50`, `FLOORS`, `SCORES`, `MARGINS`.
  - `needs_rerank(row, *, min_score, min_margin) -> bool`.
  - `apply_rerank(row, *, min_score, min_margin, floor, r_score, r_margin) -> dict` (adds `stage`, `rerank_score`, `rerank_margin`; may replace `predicted`, `reason`).
  - `select_rerank(rows, *, min_score, min_margin) -> {"rerank_floor", "rerank_min_score", "rerank_min_margin", "calibration", "quality_target_met"}`; raises `ValueError` on any non-development row.
  - `adopt(dense: dict, combined: dict) -> bool`.
  - Row contract: `split`, `slice`, `expected`, `ranked`, `score`, `margin`, `predicted`, `reason`, `top_groups`, optional `rerank_scores` (aligned with `top_groups`, or `None` when skipped for length).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rerank_calibration.py`:

```python
import pytest

from faq.evaluation.rerank_calibration import adopt, apply_rerank, select_rerank

DENSE = {"min_score": 0.76, "min_margin": 0.04}


def dev_row(expected, *, score, margin, predicted=None, groups=(), rerank=None, split="calibration"):
    return {
        "split": split,
        "slice": "base",
        "expected": list(expected),
        "ranked": [a for a, _, _ in groups],
        "score": score,
        "margin": margin,
        "predicted": predicted,
        "reason": None if predicted else "low_similarity",
        "top_groups": [{"answer_id": a, "question_id": q, "score": s} for a, q, s in groups],
        "rerank_scores": rerank,
    }


def rows():
    return [
        dev_row([1], score=0.90, margin=0.10, predicted=1, groups=[(1, 10, 0.90)]),
        dev_row([2], score=0.70, margin=0.01, groups=[(3, 30, 0.70), (2, 20, 0.69)], rerank=[0.1, 0.9]),
        dev_row([], score=0.72, margin=0.02, groups=[(4, 40, 0.72), (5, 50, 0.70)], rerank=[0.3, 0.2]),
    ]


def test_selection_recovers_band_answer_and_rejects_unsupported_request():
    choice = select_rerank(rows(), **DENSE)
    assert (choice["rerank_floor"], choice["rerank_min_score"], choice["rerank_min_margin"]) == (0.7, 0.9, 0.5)
    assert choice["quality_target_met"] is True
    assert choice["calibration"]["correct_accepted"] == 2
    assert choice["calibration"]["false_accepted"] == 0


def test_selection_refuses_test_rows():
    with pytest.raises(ValueError, match="development"):
        select_rerank([*rows(), dev_row([1], score=0.9, margin=0.1, predicted=1, split="test")], **DENSE)


def test_apply_rerank_keeps_dense_outside_band_and_for_skipped_pairs():
    confident, band, _ = rows()
    values = {**DENSE, "floor": 0.7, "r_score": 0.9, "r_margin": 0.5}
    assert apply_rerank(confident, **values)["stage"] == "dense"
    reranked = apply_rerank(band, **values)
    assert (reranked["stage"], reranked["predicted"], reranked["rerank_score"]) == ("rerank", 2, 0.9)
    skipped = apply_rerank({**band, "rerank_scores": None}, **values)
    assert (skipped["stage"], skipped["predicted"], skipped["reason"]) == ("dense", None, "low_similarity")


def test_adoption_requires_more_correct_answers_within_quality_targets():
    dense = {"correct_accepted": 2, "accepted_precision": 1.0, "false_acceptance": 0.0}
    assert not adopt(dense, dense)
    assert not adopt(dense, {"correct_accepted": 3, "accepted_precision": 0.9, "false_acceptance": 0.0})
    assert not adopt(dense, {"correct_accepted": 3, "accepted_precision": 1.0, "false_acceptance": 0.1})
    assert adopt(dense, {"correct_accepted": 3, "accepted_precision": 0.96, "false_acceptance": 0.05})
```

Append to `tests/test_evaluation.py`:

```python
def test_rank_queries_keeps_top_groups_for_reranking():
    from types import SimpleNamespace

    import numpy as np

    from faq.evaluation.runner import rank_queries
    from faq.search.answer_groups import EquivalenceRegistry

    questions = [SimpleNamespace(id=10, answer_id=1), SimpleNamespace(id=20, answer_id=2)]
    documents = np.eye(2, dtype=np.float32)
    query = np.array([[0.1, 0.9]], dtype=np.float32)
    row = rank_queries(query, documents, questions, EquivalenceRegistry("v1", ()), limit=2, groups_limit=1)[0]
    assert row["ranked"] == [2, 1]
    assert row["top_groups"] == [{"answer_id": 2, "question_id": 20, "score": pytest.approx(0.9)}]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rerank_calibration.py tests/test_evaluation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'faq.evaluation.rerank_calibration'` and `TypeError: ... unexpected keyword argument 'groups_limit'`.

- [ ] **Step 3: Implement the `rank_queries` change**

In `src/faq/evaluation/runner.py`, replace `rank_queries` with:

```python
def rank_queries(query_vectors, document_vectors, questions, registry, *, limit=30, groups_limit=5):
    scores = query_vectors @ document_vectors.T
    output = []
    for row in scores:
        order = np.argsort(-row, kind="stable")[:limit]
        candidates = [
            ScoredQuestion(questions[int(i)].id, questions[int(i)].answer_id, float(row[i])) for i in order
        ]
        groups = group_answer_candidates(candidates, registry)
        output.append(
            {
                "ranked": [g.best_answer_id for g in groups],
                "score": groups[0].best_score if groups else -1,
                "margin": groups[0].best_score - groups[1].best_score if len(groups) > 1 else None,
                "top_groups": [
                    {"answer_id": g.best_answer_id, "question_id": g.question_id, "score": g.best_score}
                    for g in groups[:groups_limit]
                ],
            }
        )
    return output
```

- [ ] **Step 4: Implement the calibration module**

Create `src/faq/evaluation/rerank_calibration.py`:

```python
"""Second stage: dense thresholds stay frozen; floor, reranker score and margin use development rows only."""

import numpy as np

from ..search.answer_groups import AnswerGroup, decide_reranked, route_scores
from .metrics import metrics

LOWEST_FLOOR = 0.50
FLOORS = [round(0.50 + 0.01 * i, 2) for i in range(27)]  # 0.50 .. 0.76
SCORES = [round(0.01 * i, 2) for i in range(100)]  # 0.00 .. 0.99
MARGINS = [round(0.01 * i, 2) for i in range(51)]  # 0.00 .. 0.50


def needs_rerank(row, *, min_score, min_margin):
    route = route_scores(
        row["score"], row["margin"], min_score=min_score, min_margin=min_margin, floor=LOWEST_FLOOR
    )
    return route == "rerank"


def _groups(row):
    return [
        AnswerGroup(g["answer_id"], (g["answer_id"],), g["answer_id"], g["score"], g["question_id"])
        for g in row["top_groups"]
    ]


def apply_rerank(row, *, min_score, min_margin, floor, r_score, r_margin):
    """Combined runtime decision for one evaluated row; `row` already holds the dense decision."""
    route = route_scores(row["score"], row["margin"], min_score=min_score, min_margin=min_margin, floor=floor)
    if route != "rerank" or row.get("rerank_scores") is None:
        return {**row, "stage": "dense", "rerank_score": None, "rerank_margin": None}
    decision = decide_reranked(_groups(row), row["rerank_scores"], min_score=r_score, min_margin=r_margin)
    return {
        **row,
        "predicted": decision.answer_id,
        "reason": decision.reason,
        "stage": "rerank",
        "rerank_score": decision.rerank_score,
        "rerank_margin": decision.rerank_margin,
    }


def select_rerank(rows, *, min_score, min_margin):
    if any(row["split"] != "calibration" for row in rows):
        raise ValueError("Reranker selection must use development rows only")
    size = len(rows)
    positive = np.array([bool(row["expected"]) for row in rows])
    negatives = int((~positive).sum())
    confident = np.array([row["predicted"] is not None for row in rows])
    confident_ok = np.array(
        [row["predicted"] is not None and row["predicted"] in row["expected"] for row in rows]
    )
    dense = np.array([row["score"] for row in rows], dtype=float)
    has = np.zeros(size, dtype=bool)
    top = np.zeros(size)
    gap = np.full(size, np.inf)
    top_ok = np.zeros(size, dtype=bool)
    for i, row in enumerate(rows):
        if confident[i] or row.get("rerank_scores") is None:
            continue
        best = decide_reranked(_groups(row), row["rerank_scores"], min_score=0, min_margin=0)
        has[i], top[i], top_ok[i] = True, best.rerank_score, best.answer_id in row["expected"]
        gap[i] = np.inf if best.rerank_margin is None else best.rerank_margin
    scores = np.array(SCORES)[None, :, None]
    margins = np.array(MARGINS)[None, None, :]
    best_key = None
    for floor in FLOORS:
        band = (has & (dense >= floor))[:, None, None]
        reranked = band & (top[:, None, None] >= scores) & (gap[:, None, None] >= margins)
        accepted = confident[:, None, None] | reranked
        correct = (confident_ok[:, None, None] | (reranked & top_ok[:, None, None])).sum(0)
        count = accepted.sum(0)
        false_accepted = (accepted & ~positive[:, None, None]).sum(0)
        for i, j in zip(*np.nonzero(count), strict=True):
            good, total = int(correct[i, j]), int(count[i, j])
            false_rate = false_accepted[i, j] / negatives if negatives else 0
            feasible = good / total >= 0.95 and false_rate <= 0.05
            key = (feasible, good - 8 * (total - good), good, floor, SCORES[i], MARGINS[j])
            if best_key is None or key > best_key:
                best_key = key
    if best_key is None:
        raise ValueError("Reranker calibration cannot accept any answer")
    feasible, _, _, floor, r_score, r_margin = best_key
    chosen = [
        apply_rerank(
            row, min_score=min_score, min_margin=min_margin, floor=floor, r_score=r_score, r_margin=r_margin
        )
        for row in rows
    ]
    return {
        "rerank_floor": floor,
        "rerank_min_score": r_score,
        "rerank_min_margin": r_margin,
        "calibration": metrics(chosen),
        "quality_target_met": bool(feasible),
    }


def adopt(dense, combined):
    """Design rule: more correct answers, precision >= 95%, false acceptance <= 5% (development only)."""
    return (
        combined["correct_accepted"] > dense["correct_accepted"]
        and (combined["accepted_precision"] or 0) >= 0.95
        and (combined["false_acceptance"] or 0) <= 0.05
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass, 1 skipped.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests
git add src/faq/evaluation/rerank_calibration.py src/faq/evaluation/runner.py tests/test_rerank_calibration.py tests/test_evaluation.py
git commit -m "feat: calibrate reranker floor, score and margin on development rows"
```

---

### Task 6: Evaluation runner integration and real measurement

**Files:**
- Modify: `src/faq/evaluation/runner.py` (imports, new `_rerank_stage`, `run_evaluation`, `_write_report`)
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Generated: `evaluation/uzbek/manifest.json`, `evaluation/results/report.json`, `evaluation/results/*-predictions.jsonl`, `evaluation/results/rerank-predictions.jsonl`, `docs/evaluation/uzbek-normalization.md`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: manifest keys `rerank_enabled`, `reranker_name`, `reranker_revision`, `rerank_groups`, `rerank_floor`, `rerank_min_score`, `rerank_min_margin`, `test_status`, `rerank_selection_basis`; `report.json` key `rerank` with `model`, `revision`, `adopted`, `test_status`, `selection`, `dense_development`, `development`, `test`, `reranked_rows`, `skipped_length`, `warm_rerank_path_ms`, `reranker_startup_seconds`, `process_rss_gib`; `rerank-predictions.jsonl` (combined rows with `stage`).

- [ ] **Step 1: Add psutil as a dev dependency**

Run: `uv add --optional dev psutil`
Expected: the `dev` list in `pyproject.toml` gains `psutil`; `uv.lock` updated.

- [ ] **Step 2: Add the import to `runner.py`**

After `from .metrics import apply_threshold, calibration_families, metrics, select_thresholds` add:

```python
from .rerank_calibration import adopt, apply_rerank, needs_rerank, select_rerank
```

- [ ] **Step 3: Add `_rerank_stage` to `runner.py`** (directly before `def run_evaluation`)

```python
def _rerank_stage(settings, selected, rows, cases, frozen, *, encoder, document_vectors, questions, source, registry):
    """Second stage for the selected profile; dense values in `frozen` are never changed here."""
    from ..rerank import PairTooLong, Reranker

    reranker = Reranker(settings)
    texts = {q.id: q.question + "\n" + source.answers[q.answer_id].answer for q in questions}
    dense = {"min_score": frozen["min_score"], "min_margin": frozen["min_margin"]}
    reranked = skipped = 0
    for row, case in zip(rows, cases, strict=True):
        if not needs_rerank(row, **dense):
            continue
        query = prepare_profile(case["query"], selected)
        try:
            row["rerank_scores"] = reranker.score(
                query, [texts[g["question_id"]] for g in row["top_groups"]], cache=True
            )
            reranked += 1
        except PairTooLong:
            row["rerank_scores"] = None
            skipped += 1
    development = calibration_families(rows)
    choice = select_rerank(development, **dense)
    dense_development = metrics(development)
    adopted = adopt(dense_development, choice["calibration"])
    values = {key: choice[key] for key in ("rerank_floor", "rerank_min_score", "rerank_min_margin")}
    combined = [
        apply_rerank(
            row,
            **dense,
            floor=values["rerank_floor"],
            r_score=values["rerank_min_score"],
            r_margin=values["rerank_min_margin"],
        )
        for row in rows
    ]
    latencies = []
    for row, case in [(r, c) for r, c in zip(rows, cases, strict=True) if r.get("rerank_scores")][:12]:
        started = perf_counter()
        query = prepare_profile(case["query"], selected)
        ranked = rank_queries(
            encoder.encode([query]),
            document_vectors,
            questions,
            registry,
            limit=settings.top_k,
            groups_limit=settings.rerank_groups,
        )[0]
        reranker.score(query, [texts[g["question_id"]] for g in ranked["top_groups"]])
        latencies.append(1000 * (perf_counter() - started))
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 2**30
    except ImportError:
        rss = None
    return {
        "rows": combined,
        "manifest": {
            "rerank_enabled": adopted,
            "reranker_name": settings.reranker_name,
            "reranker_revision": settings.reranker_revision,
            "rerank_groups": settings.rerank_groups,
            **values,
            "test_status": "reused",
            "rerank_selection_basis": (
                "Dense 0.76/0.04 frozen. Floor, reranker score and margin chosen on 108 development "
                "families only; adopt only if correct accepted rises with >=95% precision and <=5% "
                "false acceptance. Test split reused, not fresh."
            ),
        },
        "report": {
            "model": settings.reranker_name,
            "revision": settings.reranker_revision,
            "adopted": adopted,
            "test_status": "reused, not fresh",
            "selection": choice,
            "dense_development": dense_development,
            "reranked_rows": reranked,
            "skipped_length": skipped,
            "warm_rerank_path_ms": {
                "samples": len(latencies),
                "p50": float(np.percentile(latencies, 50)) if latencies else None,
                "p95": float(np.percentile(latencies, 95)) if latencies else None,
            },
            "reranker_startup_seconds": reranker.startup_seconds,
            "process_rss_gib": rss,
        },
    }
```

- [ ] **Step 4: Wire it into `run_evaluation`**

1. In the profile loop, replace the `rows = _labeled(...)` line with:

```python
        rows = _labeled(
            rank_queries(
                queries, indexed, questions, registry, limit=settings.top_k, groups_limit=settings.rerank_groups
            ),
            cases,
        )
```

2. Directly before `(root / "manifest.json").write_text(...)`, insert:

```python
    rerank = _rerank_stage(
        settings,
        selected,
        profiles[selected]["rows"],
        cases,
        frozen,
        encoder=encoder,
        document_vectors=prepared_indexes[selected],
        questions=questions,
        source=source,
        registry=registry,
    )
    frozen.update(rerank["manifest"])
```

3. Directly after the `for profile, data in profiles.items():` loop (the one that writes `{profile}-predictions.jsonl`), insert:

```python
    (out / "rerank-predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rerank["rows"]), encoding="utf-8"
    )
    report["rerank"] = {
        **rerank["report"],
        "development": _summary([r for r in rerank["rows"] if r["split"] == "calibration"]),
        "test": _summary([r for r in rerank["rows"] if r["split"] == "test"]),
    }
```

4. In the dict returned by `run_evaluation`, add `"rerank_adopted": report["rerank"]["adopted"],`.

- [ ] **Step 5: Extend `_write_report`**

Directly before `destination = Path("docs/evaluation/uzbek-normalization.md")`, insert:

```python
    rr = report["rerank"]

    def line(name, m):
        return (
            f"| {name} | {percent(m['accepted_precision'])} | {percent(m['correct_answer_coverage'])} | "
            f"{percent(m['false_acceptance'])} | {m['correct_accepted']} | {m['wrong_accepted']} |"
        )

    sel, warm = rr["selection"], rr["warm_rerank_path_ms"]
    rss = "not measured" if rr["process_rss_gib"] is None else f"{rr['process_rss_gib']:.2f} GiB"
    latency = "n/a" if warm["p50"] is None else f"p50 {warm['p50']:.1f} ms, p95 {warm['p95']:.1f} ms"
    lines += [
        "## Reranker stage",
        "",
        (
            f"Model {rr['model']} (revision {rr['revision'][:12]}), local CPU. Uncertain dense matches "
            f"(ambiguous, or dense score from the floor up to {selected['min_score']}) are reranked over the top "
            f"{selected['rerank_groups']} answer groups. Chosen on development families only: floor "
            f"**{sel['rerank_floor']}**, reranker score **{sel['rerank_min_score']}**, reranker margin "
            f"**{sel['rerank_min_margin']}**. Adopted for runtime: **{'yes' if rr['adopted'] else 'no'}**."
        ),
        "",
        (
            "**The test split is reused, not fresh.** It was measured once before this change and motivated it. "
            "No value was tuned on it, but it is not independent evidence; a fresh held-out set is still required."
        ),
        "",
        "| Data | Accepted precision | Correct coverage | False acceptance | Correct | Wrong |",
        "|---|---:|---:|---:|---:|---:|",
        line("Development, dense only", rr["dense_development"]),
        line("Development, dense + rerank", rr["development"]["base_and_no_match"]),
        line("Test (reused), dense only", final),
        line("Test (reused), dense + rerank", rr["test"]["base_and_no_match"]),
        "",
        (
            f"Reranked cases: {rr['reranked_rows']} of {report['cases']}; skipped for length: "
            f"{rr['skipped_length']}. Warm rerank path (query embedding, exact search, reranking): {latency} over "
            f"{warm['samples']} samples. Reranker startup {rr['reranker_startup_seconds']:.1f} s. Evaluation "
            f"process memory with both models loaded: {rss}. Three values on 108 development families can overfit."
        ),
        "",
    ]
```

- [ ] **Step 6: Run unit tests and lint**

Run: `uv run pytest -q` → all pass, 1 skipped. `uv run ruff check src tests scripts` → clean.

- [ ] **Step 7: Run the real evaluation**

Docker Desktop must be running (the runner checks Qdrant parity). The first run downloads the reranker (≈2.2 GB).

```powershell
docker compose up -d postgres qdrant
uv run faq evaluate
```

Expected output: JSON with `"profile": "uz_surface_v1"`, `"thresholds": {"min_score": 0.76, "min_margin": 0.04}` and a `"rerank_adopted"` boolean.

- [ ] **Step 8: Check that the dense stage did not move**

Run: `git diff evaluation/uzbek/manifest.json`
Expected: only added keys (`rerank_*`, `reranker_*`, `test_status`, `rerank_selection_basis`); `profile`, `min_score`, `min_margin`, `top_k`, `corpus_digest`, `registry_digest`, `cases_sha256` unchanged. If any of those changed, STOP and report — do not continue.

Then check that dense decisions are unchanged:

```powershell
uv run python -c "import json,subprocess; old=[json.loads(l) for l in subprocess.run(['git','show','HEAD:evaluation/results/uz_surface_v1-predictions.jsonl'],capture_output=True,text=True,encoding='utf-8').stdout.splitlines()]; new=[json.loads(l) for l in open('evaluation/results/uz_surface_v1-predictions.jsonl',encoding='utf-8')]; print(all((a['case_id'],a['predicted'],a['reason'])==(b['case_id'],b['predicted'],b['reason']) for a,b in zip(old,new,strict=True)))"
```

Expected: `True`.

- [ ] **Step 9: Confirm the reported numbers agree**

```powershell
uv run python -c "import json; r=json.load(open('evaluation/results/report.json',encoding='utf-8'))['rerank']; print(r['adopted'], r['selection']['rerank_floor'], r['selection']['rerank_min_score'], r['selection']['rerank_min_margin']); print(r['dense_development']['correct_accepted'], r['selection']['calibration']['correct_accepted'], r['development']['base_and_no_match']['correct_accepted'])"
```

Expected: the second and third numbers on the last line are equal (selection metrics = recomputed development metrics).

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock src/faq/evaluation/runner.py evaluation/uzbek/manifest.json evaluation/results docs/evaluation/uzbek-normalization.md
git commit -m "feat: measure reranker stage on development data and record adoption"
```

---

### Task 7: Live API verification

**Files:**
- Modify: `scripts/verify_live_api.py`
- Generated: `evaluation/results/live-api.json`

**Interfaces:**
- Consumes: `settings.rerank_enabled` (Task 1), response fields `stage`, `rerank_score` (Task 4), `evaluation/results/rerank-predictions.jsonl` (Task 6).

- [ ] **Step 1: Choose the frozen predictions by runtime mode**

Replace the `predictions = {...}` block with:

```python
predictions_name = "rerank" if settings.rerank_enabled else settings.profile
predictions = {
    r["case_id"]: r
    for r in (
        json.loads(line)
        for line in Path(f"evaluation/results/{predictions_name}-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
}
```

- [ ] **Step 2: Check the rerank stage per case and time it**

Change `checks, latencies = [], []` to `checks, latencies, rerank_latencies = [], [], []`.

Replace `latencies.append(1000 * (perf_counter() - started))` with:

```python
        elapsed = 1000 * (perf_counter() - started)
        latencies.append(elapsed)
```

Directly after `assert delta < 1e-5, (case["case_id"], delta)`, insert:

```python
        assert actual["stage"] == frozen.get("stage", "dense"), (case["case_id"], actual["stage"])
        if actual["stage"] == "rerank":
            rerank_latencies.append(elapsed)
            assert abs(actual["rerank_score"] - frozen["rerank_score"]) < 1e-4, case["case_id"]
```

- [ ] **Step 3: Report the new values**

In the `report = {...}` dict, set `"scope"` to
`"Real FastAPI -> BGE-M3 -> Qdrant -> PostgreSQL (+ reranker when enabled), all 54 test base/no-match families (reused, not fresh)"`, and add:

```python
    "rerank_enabled": settings.rerank_enabled,
    "rerank_http_ms": {
        "samples": len(rerank_latencies),
        "p50": float(np.percentile(rerank_latencies, 50)) if rerank_latencies else None,
        "p95": float(np.percentile(rerank_latencies, 95)) if rerank_latencies else None,
    },
```

- [ ] **Step 4: Run against the real stack**

```powershell
docker compose up -d postgres qdrant
uv run uvicorn faq.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal, after `Invoke-RestMethod http://127.0.0.1:8000/ready` returns `ready`:

```powershell
uv run python scripts/verify_live_api.py
```

Expected: no assertion error; the printed JSON shows `"rerank_enabled"` equal to the manifest value and `rerank_http_ms` filled when enabled. Stop the API with Ctrl+C.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check scripts
git add scripts/verify_live_api.py evaluation/results/live-api.json
git commit -m "test: verify live API decisions including reranker stage"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md`, `docs/GUIDE.md`, `.env.example`, `docs/design/reranker-design.md` (status line)

All numbers come from `evaluation/results/report.json` (`rerank` section), `evaluation/results/live-api.json` and `docs/evaluation/uzbek-normalization.md`. Copy them exactly as rounded in the generated report.

- [ ] **Step 1: `.env.example`**

Append:

```text
# Reranker values come from evaluation/uzbek/manifest.json; set only for explicit experiments.
# FAQ_RERANK_ENABLED=0
```

- [ ] **Step 2: README**

Under "Search quality", add a second table with rows "Dense only" and "Dense + rerank" for the reused test split, columns Accepted precision, Correct answer coverage, False acceptance — from `report.json` → `profiles.uz_surface_v1.test.base_and_no_match` and `rerank.test.base_and_no_match`. Add the sentence: "The test split is reused, not fresh: it was measured once before the reranker and motivated it. Reranker values were chosen on development data only." State whether the reranker is enabled at runtime (`rerank.adopted`). If adopted, change "≈2.3 GB of model weights" in "Run locally" to "≈4.5 GB of model weights (BGE-M3 + reranker)".

- [ ] **Step 3: GUIDE**

- §2 picture: insert the routing block from the spec's "Decision flow" between "group by answer" and the final reply.
- §3 table: add row `| **bge-reranker-v2-m3** (pinned revision) | Re-scores the top 5 answer groups for uncertain queries | Reads query and stored text together; local, multilingual |`.
- §4 folder map: add `rerank.py` and `evaluation/rerank_calibration.py`.
- §7: after step 8 add "8b. If the dense result is uncertain (ambiguous, or score between the floor and 0.76), fetch the question + answer text of the top 5 groups and rerank them (§9)."
- §9: add the routing rules, the reasons `rerank_low_score` / `rerank_ambiguous`, the response fields `stage` / `rerank_score` / `rerank_margin`, and the frozen values from the manifest (`rerank_floor`, `rerank_min_score`, `rerank_min_margin`).
- §12: add the `FAQ_RERANK_*` / `RERANKER_*` variables with default "from manifest.json".
- §13: add a "Second stage" paragraph: dense values frozen; grid over floor / reranker score / margin on development only; adoption rule.
- §14: add the development and reused-test table from `docs/evaluation/uzbek-normalization.md` → "Reranker stage", the latency (`rerank.warm_rerank_path_ms`, `live-api.json` → `rerank_http_ms`) and memory (`rerank.process_rss_gib`), with the "reused, not fresh" warning.
- §15: add `tests/test_rerank.py` and `tests/test_rerank_calibration.py` rows; update the test count from `uv run pytest -q`.
- §17: replace next step 3 with the measured outcome; keep "collect a fresh held-out set" as step 1.
- §18: update "Biggest weakness?" with the new coverage number.

- [ ] **Step 4: Spec status**

In `docs/design/reranker-design.md`, change the status line to `Status: implemented 2026-10-13; adopted: yes.` or `... adopted: no.` per `evaluation/uzbek/manifest.json` → `rerank_enabled`.

- [ ] **Step 5: Final check and commit**

```bash
uv run pytest -q
uv run ruff check src tests scripts
git add README.md docs/GUIDE.md .env.example docs/design/reranker-design.md
git commit -m "docs: document reranker stage and its measured effect"
```
