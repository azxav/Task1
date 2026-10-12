import numpy as np


def calibration_families(rows):
    return [
        r
        for r in rows
        if r["split"] == "calibration" and r["slice"] in ("base", "unrelated", "unsupported_action")
    ]


def metrics(rows):
    positives = [row for row in rows if row["expected"]]
    negatives = [row for row in rows if not row["expected"]]
    accepted = [row for row in rows if row.get("predicted") is not None]
    correct = sum(row["predicted"] in row["expected"] for row in accepted)
    wrong = len(accepted) - correct
    false_accepted = sum(row.get("predicted") is not None for row in negatives)
    true_accepted = sum(row.get("predicted") is not None for row in positives)
    top1 = sum(bool(row["ranked"]) and row["ranked"][0] in row["expected"] for row in positives)
    recall5 = sum(bool(set(row["ranked"][:5]).intersection(row["expected"])) for row in positives)
    return {
        "cases": len(rows),
        "answerable": len(positives),
        "no_match": len(negatives),
        "accepted": len(accepted),
        "correct_accepted": correct,
        "wrong_accepted": wrong,
        "false_accepted": false_accepted,
        "top1_accuracy": top1 / len(positives) if positives else None,
        "recall_at_5": recall5 / len(positives) if positives else None,
        "accepted_precision": correct / len(accepted) if accepted else None,
        "answer_coverage": true_accepted / len(positives) if positives else None,
        "correct_answer_coverage": correct / len(positives) if positives else None,
        "false_acceptance": false_accepted / len(negatives) if negatives else None,
        "false_rejection": 1 - true_accepted / len(positives) if positives else None,
    }


def apply_threshold(row, score, margin):
    accepted = row["score"] >= score and (row["margin"] is None or row["margin"] >= margin)
    reason = (
        None
        if accepted and row["ranked"]
        else ("low_similarity" if not row["ranked"] or row["score"] < score else "ambiguous")
    )
    return {**row, "predicted": row["ranked"][0] if accepted and row["ranked"] else None, "reason": reason}


def select_thresholds(rows):
    candidates = []
    for score in np.arange(0.35, 0.961, 0.01):
        for margin in np.arange(0, 0.151, 0.005):
            result = metrics([apply_threshold(row, score, margin) for row in rows])
            if not result["accepted"]:
                continue
            feasible = result["accepted_precision"] >= 0.95 and (result["false_acceptance"] or 0) <= 0.05
            utility = result["correct_accepted"] - 8 * result["wrong_accepted"]
            candidates.append(
                (
                    feasible,
                    utility,
                    result["correct_accepted"],
                    -float(score),
                    -float(margin),
                    {
                        "min_score": round(float(score), 3),
                        "min_margin": round(float(margin), 3),
                        "calibration": result,
                        "quality_target_met": feasible,
                    },
                )
            )
    if not candidates:
        raise ValueError("Calibration cannot accept any answer; do not publish thresholds")
    return max(candidates, key=lambda c: c[:5])[-1]
