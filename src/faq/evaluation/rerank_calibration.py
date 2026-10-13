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
