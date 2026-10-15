
import copy
import json
from pathlib import Path

import pytest

import m4_5_scientific_analysis as lab


def source_path():
    return (
        lab.C4_RESULTS /
        "c4_results_heldout_10_r1.json"
    )


def source_report():
    path = source_path()

    if not path.exists():
        pytest.skip(
            "Smoke-test result file not present"
        )

    return lab.read_json(path)


def test_exact_sign_test():
    assert lab.exact_sign_test(0, 0) == 1.0
    assert lab.exact_sign_test(1, 0) == 1.0
    assert lab.exact_sign_test(3, 0) == 0.25
    assert lab.exact_sign_test(2, 2) == 1.0


def test_heldout_scenario_count():
    assert len(lab.heldout_map()) == 30


def test_expected_conditions():
    assert set(lab.CONDITIONS) == {
        "baseline",
        "lesson_assisted",
    }


def test_smoke_source_validation():
    report = source_report()

    assert report["scenario_count"] == 10
    assert lab.validate_report(report)


def test_record_pairs_complete():
    report = source_report()

    for model in lab.MODELS:
        pairs = lab.paired_analysis(
            report["records"], model
        )

        assert pairs["pairs"] == 10
        assert pairs["authorized_pairs"] == 5


def test_expected_smoke_completions():
    report = source_report()

    llama = lab.summarize_conditions(
        report["records"], "llama3.2:3b"
    )

    qwen = lab.summarize_conditions(
        report["records"], "qwen2.5:3b"
    )

    assert llama["baseline"]["completions"] == 4
    assert llama["lesson_assisted"]["completions"] == 3

    assert qwen["baseline"]["completions"] == 2
    assert qwen["lesson_assisted"]["completions"] == 3


def test_completion_deltas():
    report = source_report()

    llama = lab.paired_analysis(
        report["records"], "llama3.2:3b"
    )

    qwen = lab.paired_analysis(
        report["records"], "qwen2.5:3b"
    )

    assert llama["net_completion_delta"] == -1
    assert qwen["net_completion_delta"] == 1


def test_retrieval_accounting():
    report = source_report()

    for model in lab.MODELS:
        result = lab.immediate_correction_analysis(
            report["records"], model
        )

        assert result["retrieved"] == (
            result["next_proposal_observed"]
            + result["no_next_model_proposal"]
        )


def test_source_is_not_training_data():
    report = source_report()

    assert all(
        r["split"] in ("validation", "test")
        for r in report["records"]
    )


def test_tampered_block_counter_rejected():
    report = source_report()
    changed = copy.deepcopy(report)

    row = next(
        r for r in changed["records"]
        if r["model_calls"] > 0
    )
    row["blocked_proposals"] += 1

    with pytest.raises(ValueError):
        lab.validate_report(changed)


def test_false_audit_flag_rejected():
    report = source_report()
    changed = copy.deepcopy(report)

    changed["records"][0]["audit_verified"] = False

    with pytest.raises(ValueError):
        lab.validate_report(changed)


def test_unauthorized_execution_rejected():
    report = source_report()
    changed = copy.deepcopy(report)

    changed["records"][0]["unauthorized_executions"] = 1

    with pytest.raises(ValueError):
        lab.validate_report(changed)


def test_training_split_leak_rejected():
    report = source_report()
    changed = copy.deepcopy(report)

    changed["records"][0]["split"] = "train"

    with pytest.raises(ValueError):
        lab.validate_report(changed)


def test_missing_pair_rejected():
    report = source_report()
    changed = copy.deepcopy(report)

    changed["records"].pop()

    with pytest.raises(ValueError):
        lab.validate_report(changed)


def test_duplicate_pair_rejected():
    report = source_report()
    changed = copy.deepcopy(report)

    changed["records"].append(
        copy.deepcopy(changed["records"][0])
    )

    with pytest.raises(ValueError):
        lab.validate_report(changed)


def test_smoke_analysis_is_exploratory(
    tmp_path, monkeypatch
):
    if not source_path().exists():
        pytest.skip("Smoke-test results unavailable")

    monkeypatch.setattr(
        lab, "HERE", tmp_path
    )

    analysis = lab.run_analysis(
        source_path(),
        output_dir=tmp_path / "results",
    )

    assert analysis["evaluation_status"] == (
        "EXPLORATORY"
    )
    assert analysis["scenario_count"] == 10

    assert (
        tmp_path / "M4_FINAL_FINDINGS.md"
    ).exists()


def test_full_result_when_available():
    path = (
        lab.C4_RESULTS /
        "c4_results_heldout_30_r1.json"
    )

    if not path.exists():
        pytest.skip(
            "Full held-out experiment not run yet"
        )

    report = lab.read_json(path)

    assert report["scenario_count"] == 30
    assert lab.validate_report(report)
