
import copy
import json

import pytest

import m5_4_negative_transfer as lab


def source():
    return lab.read_json(lab.SOURCE)


def test_source_exists():
    assert lab.SOURCE.is_file()


def test_full_source_counts():
    report = source()

    assert report["scenario_count"] == 30
    assert len(report["records"]) == 180
    assert lab.validate_report(report)


def test_all_models_and_conditions():
    report = source()

    for model in lab.MODELS:
        for condition in lab.CONDITIONS:
            rows = [
                row for row in report["records"]
                if row["recipient_model"] == model
                and row["condition"] == condition
            ]
            assert len(rows) == 30


def test_exact_sign_test():
    assert lab.exact_sign_test(0, 0) == 1.0
    assert lab.exact_sign_test(1, 0) == 1.0
    assert lab.exact_sign_test(2, 0) == 0.5
    assert lab.exact_sign_test(3, 0) == 0.25
    assert lab.exact_sign_test(2, 2) == 1.0


def test_no_unauthorized_executions():
    report = source()

    assert all(
        row["unauthorized_executions"] == 0
        for row in report["records"]
    )


def test_all_audits_verified():
    report = source()

    assert all(
        row["audit_verified"]
        for row in report["records"]
    )


def test_donor_recipient_mapping():
    report = source()

    for row in report["records"]:
        expected = lab.c3.donor_for(
            row["recipient_model"],
            row["condition"],
        )
        assert row["donor_model"] == expected


def test_larger_llama_donor_coverage():
    report = source()

    qwen_same = lab.retrieval_coverage(
        report["records"],
        "qwen2.5:3b",
        "same_model",
    )
    qwen_cross = lab.retrieval_coverage(
        report["records"],
        "qwen2.5:3b",
        "cross_model",
    )

    assert qwen_same["hits"] == 39
    assert qwen_same["misses"] == 17

    assert qwen_cross["hits"] == 47
    assert qwen_cross["misses"] == 5


def test_llama_completion_net_deltas():
    report = source()

    for condition in ("same_model", "cross_model"):
        result = lab.compare_pairs(
            report["records"],
            "llama3.2:3b",
            condition,
        )

        assert result["summary"][
            "net_completion_delta"
        ] == 0

        assert result["summary"][
            "authorized_pairs"
        ] == 20


def test_qwen_completion_net_deltas():
    report = source()

    for condition in ("same_model", "cross_model"):
        result = lab.compare_pairs(
            report["records"],
            "qwen2.5:3b",
            condition,
        )

        assert result["summary"][
            "net_completion_delta"
        ] == -1

        assert result["summary"][
            "authorized_pairs"
        ] == 20


def test_no_source_training_split():
    report = source()

    assert all(
        row["split"] in ("validation", "test")
        for row in report["records"]
    )


def test_trace_and_counter_validation():
    report = source()
    scenarios = {
        row["scenario_id"]: row
        for row in lab.c3.scenarios_for_evaluation()
    }

    for row in report["records"]:
        assert lab.validate_episode(
            row,
            scenarios[row["scenario_id"]],
            report["max_steps"],
        )


def test_tampered_block_counter_rejected():
    report = copy.deepcopy(source())

    row = next(
        item for item in report["records"]
        if item["model_calls"] > 0
    )
    row["blocked_proposals"] += 1

    with pytest.raises(ValueError):
        lab.validate_report(report)


def test_tampered_donor_rejected():
    report = copy.deepcopy(source())

    row = next(
        item for item in report["records"]
        if item["condition"] == "cross_model"
    )
    row["donor_model"] = row["recipient_model"]

    with pytest.raises(ValueError):
        lab.validate_report(report)


def test_tampered_audit_flag_rejected():
    report = copy.deepcopy(source())
    report["records"][0]["audit_verified"] = False

    with pytest.raises(ValueError):
        lab.validate_report(report)


def test_tampered_effect_count_rejected():
    report = copy.deepcopy(source())

    row = next(
        item for item in report["records"]
        if item["tool_effects"] > 0
    )
    row["tool_effects"] += 1

    with pytest.raises(ValueError):
        lab.validate_report(report)


def test_tampered_source_fingerprint_rejected():
    report = copy.deepcopy(source())
    report["scenario_fingerprint"] = "0" * 64

    with pytest.raises(ValueError):
        lab.validate_report(report)


def test_missing_condition_rejected():
    report = copy.deepcopy(source())
    report["records"].pop()

    with pytest.raises(ValueError):
        lab.validate_report(report)


def test_duplicate_condition_rejected():
    report = copy.deepcopy(source())
    report["records"].append(
        copy.deepcopy(report["records"][0])
    )

    with pytest.raises(ValueError):
        lab.validate_report(report)


def test_regressions_do_not_export_raw_text():
    report = source()

    result = lab.compare_pairs(
        report["records"],
        "llama3.2:3b",
        "cross_model",
    )

    for item in result["regressions"]:
        assert "raw" not in item
        assert "untrusted_inputs" not in item
        assert "argument" not in item


def test_immediate_feedback_accounting():
    report = source()

    for model in lab.MODELS:
        for condition in (
            "same_model", "cross_model"
        ):
            result = lab.immediate_feedback(
                report["records"],
                model,
                condition,
            )

            assert result["retrieved"] == (
                result["next_proposals_observed"]
                + result["no_next_proposal"]
            )

            assert result["next_proposals_observed"] == (
                result["next_allowed"]
                + result["next_denied"]
            )


def test_full_analysis_artifacts(tmp_path):
    report = lab.run_analysis(
        source=lab.SOURCE,
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert report["evaluation_status"] == "FULL"
    assert report["scenario_count"] == 30
    assert report["episodes_validated"] == 180
    assert report["unauthorized_executions"] == 0
    assert report["invalid_audit_flags"] == 0

    assert (
        tmp_path / "results"
        / "m5_4_analysis.json"
    ).exists()

    assert (
        tmp_path / "results"
        / "scenario_regressions.json"
    ).exists()

    assert (
        tmp_path / "results"
        / "manifest.json"
    ).exists()

    assert (
        tmp_path / "M5_C4_FINDINGS.md"
    ).exists()
