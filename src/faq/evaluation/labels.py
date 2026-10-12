from ..text.normalization import normalize


def validate_cases(cases, answer_ids, index_questions):
    indexed = {normalize(q).comparison_key for q in index_questions}
    ids, splits = set(), {}
    for case in cases:
        if case["case_id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(case["case_id"])
        if case["split"] not in ("calibration", "test"):
            raise ValueError("Invalid split")
        previous = splits.setdefault(case["family_id"], case["split"])
        if previous != case["split"]:
            raise ValueError("Query family leaks across splits")
        if any(ident not in answer_ids for ident in case["acceptable_answer_ids"]):
            raise ValueError("Unknown/inactive expected answer")
        if bool(case["acceptable_answer_ids"]) != (case["expected_status"] == "matched"):
            raise ValueError("Status/answer labels disagree")
        if not case.get("coverage_rationale"):
            raise ValueError("Coverage rationale is required")
        if case["slice"] == "base" and normalize(case["query"]).comparison_key in indexed:
            raise ValueError("Independent query has exact normalized index overlap")
        if case.get("parent_case_id") and case["parent_case_id"] not in ids:
            raise ValueError("Unknown parent case")
    return {"cases": len(cases), "families": len(splits), "index_overlap": 0}
