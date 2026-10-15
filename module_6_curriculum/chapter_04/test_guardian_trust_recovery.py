
import copy
import json

import pytest

import guardian_trust_recovery as lab


def load():
    data = lab.load_verified_dataset()
    report = lab.read_json(lab.SOURCE)
    return data, report


def test_source_validation():
    data, report = load()
    assert lab.validate_live_source(report, data)
    assert report["summary"]["practice_attempts"] == 16


def test_actual_results():
    _, report = load()
    llama = report["records"]["llama3.2:3b"]
    qwen = report["records"]["qwen2.5:3b"]

    assert llama["initial_assessment"]["correct"] == 4
    assert llama["final_assessment"]["correct"] == 4
    assert llama["practice"]["correct"] == 5

    assert qwen["initial_assessment"]["correct"] == 0
    assert qwen["final_assessment"]["correct"] == 3
    assert qwen["practice"]["correct"] == 3


def test_both_require_more_evidence():
    _, report = load()

    for model in lab.MODELS:
        result = lab.evaluate_model(report["records"][model])
        assert result["state"] == "PRACTICE_REQUIRED"
        assert not result["privileges_restored"]
        assert not result["permissions_changed"]


def test_qwen_improvement_not_authorization():
    _, report = load()
    result = lab.evaluate_model(
        report["records"]["qwen2.5:3b"]
    )
    assert result["assessment_delta"] == 3
    assert result["state"] != "REVIEW_ELIGIBLE"


def test_zero_evidence():
    result = lab.recommend_review(0.0)
    assert result["state"] == "PRACTICE_REQUIRED"
    assert "insufficient_independent_sessions" in result["reasons"]


def test_eligible_is_review_only():
    result = lab.recommend_review(
        practice_accuracy=0.9,
        independent_sessions=3,
        new_assessment_items=10,
        new_assessment_correct=10,
    )
    assert result["state"] == "REVIEW_ELIGIBLE"
    assert not result["reasons"]


def test_poor_new_assessment():
    result = lab.recommend_review(
        practice_accuracy=1.0,
        independent_sessions=3,
        new_assessment_items=10,
        new_assessment_correct=8,
    )
    assert result["state"] == "PRACTICE_REQUIRED"


def test_critical_failure_restricts():
    result = lab.recommend_review(
        practice_accuracy=1.0,
        independent_sessions=5,
        new_assessment_items=20,
        new_assessment_correct=20,
        critical_failures=1,
    )
    assert result["state"] == "RESTRICTED"


def test_model_error_blocks_eligibility():
    result = lab.recommend_review(
        practice_accuracy=1.0,
        independent_sessions=3,
        new_assessment_items=10,
        new_assessment_correct=10,
        model_call_errors=1,
    )
    assert result["state"] == "PRACTICE_REQUIRED"


def test_invalid_counts_rejected():
    with pytest.raises(ValueError):
        lab.recommend_review(
            practice_accuracy=0.8,
            new_assessment_items=2,
            new_assessment_correct=3,
        )


def test_tampered_practice_rejected():
    data, report = load()
    modified = copy.deepcopy(report)
    modified["records"]["llama3.2:3b"][
        "practice_records"
    ][0]["correct"] = False
    with pytest.raises(ValueError):
        lab.validate_live_source(modified, data)


def test_tampered_assessment_rejected():
    data, report = load()
    modified = copy.deepcopy(report)
    modified["records"]["qwen2.5:3b"][
        "final_assessment_records"
    ][0]["correct"] = False
    with pytest.raises(ValueError):
        lab.validate_live_source(modified, data)


def test_assessment_leakage_rejected():
    data, report = load()
    modified = copy.deepcopy(report)
    row = modified["records"]["llama3.2:3b"]
    row["practice_records"][0]["exercise_id"] = (
        row["initial_assessment_records"][0]["exercise_id"]
    )
    with pytest.raises(ValueError):
        lab.validate_live_source(modified, data)


def test_permission_tamper_rejected():
    data, report = load()
    modified = copy.deepcopy(report)
    modified["records"]["llama3.2:3b"][
        "privilege_restorations"
    ] = 1
    with pytest.raises(ValueError):
        lab.validate_live_source(modified, data)


def test_result_files(tmp_path):
    result = lab.run_policy(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    assert result["models_evaluated"] == 2
    assert result["practice_required"] == 2
    assert result["review_eligible"] == 0
    assert result["permissions_changed"] == 0
    assert result["privilege_restorations"] == 0

    assert (
        tmp_path / "results/m6_4_trust_decisions.json"
    ).is_file()
    assert (
        tmp_path / "results/manifest.json"
    ).is_file()
    assert (tmp_path / "M6_C4_FINDINGS.md").is_file()


def test_deterministic_policy(tmp_path):
    a = lab.run_policy(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    b = lab.run_policy(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    assert a == b
