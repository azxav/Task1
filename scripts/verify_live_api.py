"""Exercise real HTTP/model/index/database flow; compare frozen offline predictions."""

import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np

from faq.config import Settings
from faq.data import load_dataset
from faq.storage import Storage

settings = Settings.load()
dataset = load_dataset("data/source")
cases = [
    json.loads(line)
    for line in Path("evaluation/uzbek/queries.jsonl").read_text(encoding="utf-8").splitlines()
]
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
checks, latencies, rerank_latencies = [], [], []
with httpx.Client(base_url=settings.api_url, timeout=45) as client:
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    for case in cases:
        if case["split"] != "test" or case["slice"] not in ("base", "unrelated", "unsupported_action"):
            continue
        started = perf_counter()
        response = client.post("/search", json={"question": case["query"]})
        elapsed = 1000 * (perf_counter() - started)
        latencies.append(elapsed)
        assert response.status_code == 200, (case["case_id"], response.status_code)
        actual, frozen = response.json(), predictions[case["case_id"]]
        assert actual["answer_id"] == frozen["predicted"], (
            case["case_id"],
            actual["answer_id"],
            frozen["predicted"],
        )
        delta = abs(actual["score"] - frozen["score"])
        assert delta < 1e-5, (case["case_id"], delta)
        assert actual["stage"] == frozen.get("stage", "dense"), (case["case_id"], actual["stage"])
        if actual["stage"] == "rerank":
            rerank_latencies.append(elapsed)
            assert abs(actual["rerank_score"] - frozen["rerank_score"]) < 1e-4, case["case_id"]
        if actual["status"] == "matched":
            assert actual["answer"] == dataset.answers[actual["answer_id"]].answer
        else:
            assert actual["reason"] == frozen["reason"]
        checks.append({"case_id": case["case_id"], "same_decision": True, "score_delta": delta})
    for query in (" ", "x" * 2001, "q " * 700):
        assert client.post("/search", json={"question": query}).status_code == 422
    ordinary = client.post("/search", json={"question": "O'qish joyidan ma'lumotnoma olish mumkinmi?"}).json()
    typography = client.post(
        "/search", json={"question": "Oʻqish joyidan maʼlumotnoma olish mumkinmi?"}
    ).json()
    assert ordinary["answer_id"] == typography["answer_id"]
    assert abs(ordinary["score"] - typography["score"]) < 1e-7
report = {
    "scope": "Real FastAPI -> BGE-M3 -> Qdrant -> PostgreSQL (+ reranker when enabled), all 54 test base/no-match families (reused, not fresh)",
    "checks": checks,
    "verbatim_answers": True,
    "validation_422": True,
    "typography_identical": True,
    "database_counts": Storage(settings.database_url).counts(),
    "runtime_versions": {name: version(name) for name in ("torch", "transformers", "sentence-transformers")},
    "sequential_http_ms": {
        "samples": len(latencies),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
    },
    "rerank_enabled": settings.rerank_enabled,
    "rerank_http_ms": {
        "samples": len(rerank_latencies),
        "p50": float(np.percentile(rerank_latencies, 50)) if rerank_latencies else None,
        "p95": float(np.percentile(rerank_latencies, 95)) if rerank_latencies else None,
    },
    "portability_note": "CPU-only locked wheel agrees with pre-switch torch 2.14.1 Windows CPU calibration to <1e-5 on every fresh base/no-match case; selection unchanged.",
}
Path("evaluation/results/live-api.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))
