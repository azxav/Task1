"""Development-only experiment; inspected v1 cases are NOT a final test."""

import json
from pathlib import Path

import numpy as np

from faq.config import Settings
from faq.data import load_dataset
from faq.embeddings import Encoder
from faq.evaluation.metrics import select_thresholds
from faq.evaluation.runner import _labeled, rank_queries
from faq.evaluation.text_profiles import prepare_profile
from faq.search.answer_groups import EquivalenceRegistry

data = load_dataset("data/source")
settings = Settings.load()
encoder = Encoder(settings)
registry = EquivalenceRegistry.load(settings.registry_path, data.answers)
cases = [
    json.loads(s)
    for s in Path("evaluation/results/v1/queries.jsonl").read_text(encoding="utf-8").splitlines()
]
alternatives = {4: 382, 23: 112, 31: 483, 125: 204, 136: 164, 145: 78, 72: 488, 56: 223}
for case in cases:
    case["split"] = "calibration"
    expected = case["acceptable_answer_ids"]
    for primary, alternative in alternatives.items():
        if primary in expected and alternative not in expected:
            expected.append(alternative)
answers = list(data.answers.values())
answer_pos = {a.id: i for i, a in enumerate(answers)}
questions = data.indexed
support_pos = [answer_pos[q.answer_id] for q in questions]
results = []
for profile in ("uz_surface_v1", "uz_surface_casefold_v1"):
    question_vectors = encoder.encode([prepare_profile(q.question, profile) for q in questions], cache=True)
    answer_vectors = encoder.encode([prepare_profile(a.answer, profile) for a in answers], cache=True)
    queries = encoder.encode([prepare_profile(c["query"], profile) for c in cases], cache=True)
    for weight in (0, 0.25, 0.5):
        # Restrict support to the same top-30 question candidate pool.
        scores = queries @ question_vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, : settings.top_k]
        support = queries @ answer_vectors[support_pos].T
        blended = (1 - weight) * scores + weight * support
        mask = np.full(blended.shape, -1.0, dtype=np.float32)
        np.put_along_axis(mask, order, np.take_along_axis(blended, order, axis=1), axis=1)
        # Reuse grouping via the ranker's matrix product interface.
        ranks = rank_queries(
            mask, np.eye(len(questions), dtype=np.float32), questions, registry, limit=settings.top_k
        )
        rows = _labeled(ranks, cases)
        base = [r for r in rows if r["slice"] in ("base", "unrelated", "unsupported_action")]
        thresholds = select_thresholds(base)
        result = {"profile": profile, "answer_weight": weight, **thresholds}
        results.append(result)
        print(json.dumps(result), flush=True)
Path("evaluation/results/answer-support-experiment.json").write_text(
    json.dumps(
        {
            "scope": "Development only: inspected v1 cases with reviewed coverage labels. No fresh test used.",
            "candidate_policy": "Top-30 question candidates; convex blend of question cosine and full stored answer cosine, same pinned BGE-M3.",
            "decision": "Keep question-only retrieval: answer support improves top-1 but reduces correct accepted coverage at the precision/no-match target. No production reranking dependency added.",
            "results": results,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
