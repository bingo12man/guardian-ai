
import copy
import json

import pytest

import m6_5_final_analysis as lab


def sources():
    return lab.validate_sources()


def test_all_sources_exist():
    for path in lab.SOURCES.values():
        assert path.is_file()


def test_full_evidence_validation():
    data = sources()

    assert len(data["dataset"]["exercises"]) == 21
    assert data["scripted"]["summary"]["sessions"] == 6
    assert data["live"]["summary"]["models"] == 2
    assert data["trust"]["practice_required"] == 2


def test_scripted_scheduler_results():
    data = sources()
    stats = data["scripted"]["summary"]

    assert stats["total_practice_attempts"] == 48
    assert stats["total_correct"] == 30
    assert stats["total_incorrect"] == 18
    assert stats["audits_verified"] == 6
    assert stats["privilege_restorations"] == 0


def test_live_practice_totals():
    data = sources()
    stats = data["live"]["summary"]

    assert stats["practice_attempts"] == 16
    assert stats["practice_correct"] == 8
    assert stats["model_call_errors"] == 0
    assert stats["audits_verified"] == 2
    assert stats["tool_executions"] == 0


def test_sign_test():
    assert lab.exact_sign_test(0, 0) == 1.0
    assert lab.exact_sign_test(1, 0) == 1.0
    assert lab.exact_sign_test(2, 0) == 0.5
    assert lab.exact_sign_test(3, 0) == 0.25
    assert lab.exact_sign_test(2, 2) == 1.0


def test_llama_assessment_pairing():
    data = sources()
    row = data["live"]["records"]["llama3.2:3b"]

    paired = lab.paired_assessment(
        row["initial_assessment_records"],
        row["final_assessment_records"],
    )

    assert paired["pairs"] == 4
    assert paired["gains"] == 0
    assert paired["losses"] == 0
    assert paired["both_correct"] == 4
    assert paired["exact_sign_p_two_sided"] == 1.0


def test_qwen_assessment_pairing():
    data = sources()
    row = data["live"]["records"]["qwen2.5:3b"]

    paired = lab.paired_assessment(
        row["initial_assessment_records"],
        row["final_assessment_records"],
    )

    assert paired["pairs"] == 3
    assert paired["gains"] == 3
    assert paired["losses"] == 0
    assert paired["net_correct_delta"] == 3
    assert paired["exact_sign_p_two_sided"] == 0.25


def test_duplicate_assessment_id_rejected():
    data = sources()
    row = data["live"]["records"]["qwen2.5:3b"]

    before = copy.deepcopy(
        row["initial_assessment_records"]
    )
    after = copy.deepcopy(
        row["final_assessment_records"]
    )

    after[1]["exercise_id"] = after[0]["exercise_id"]

    with pytest.raises(ValueError):
        lab.paired_assessment(before, after)


def test_assessment_metadata_tampering_rejected():
    data = sources()
    row = data["live"]["records"]["qwen2.5:3b"]

    before = copy.deepcopy(
        row["initial_assessment_records"]
    )
    after = copy.deepcopy(
        row["final_assessment_records"]
    )

    after[0]["stage"] = "forged_stage"

    with pytest.raises(ValueError):
        lab.paired_assessment(before, after)


def test_missing_source_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    paths["live_results"] = tmp_path / "missing.json"

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_live_source_rejected(tmp_path):
    paths = dict(lab.SOURCES)

    live = lab.read_json(paths["live_results"])
    live["records"]["qwen2.5:3b"][
        "practice_records"
    ][0]["correct"] = True

    modified = tmp_path / "live.json"
    modified.write_text(
        json.dumps(live),
        encoding="utf-8",
    )

    paths["live_results"] = modified

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_trust_decision_rejected(tmp_path):
    paths = dict(lab.SOURCES)

    trust = lab.read_json(paths["trust_decisions"])
    trust["decisions"]["qwen2.5:3b"][
        "state"
    ] = "REVIEW_ELIGIBLE"

    modified = tmp_path / "trust.json"
    modified.write_text(
        json.dumps(trust),
        encoding="utf-8",
    )

    paths["trust_decisions"] = modified

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_scheduler_results_rejected(tmp_path):
    paths = dict(lab.SOURCES)

    report = lab.read_json(
        paths["scheduler_results"]
    )
    report["summary"]["total_correct"] = 48

    modified = tmp_path / "scheduler.json"
    modified.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    paths["scheduler_results"] = modified

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_module_7_contract_safety(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    contract = result["contract"]
    policy = contract["safety_constraints"]

    assert contract["consumer_module"] == 7
    assert policy[
        "agents_can_change_permissions"
    ] is False
    assert policy[
        "agents_can_issue_capabilities"
    ] is False
    assert policy[
        "lessons_grant_privileges"
    ] is False
    assert policy[
        "reviews_grant_privileges"
    ] is False
    assert policy[
        "trusted_approval_required"
    ] is True


def test_module_7_contract_excludes_raw_content(
    tmp_path
):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    fields = result["contract"][
        "prohibited_payload_fields"
    ]

    assert "raw_prompt" in fields
    assert "raw_model_output" in fields
    assert "capability_token" in fields
    assert "permission_override" in fields


def test_final_conclusions_do_not_overclaim(
    tmp_path
):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    final = result["final"]

    assert final["causal_effect_identified"] is False
    assert final[
        "durable_model_learning_demonstrated"
    ] is False
    assert final[
        "independent_assessment_evidence"
    ] is False
    assert final["privilege_restorations"] == 0


def test_final_results(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    summary = result["summary"]

    assert summary["models"] == 2
    assert summary["practice_attempts"] == 16
    assert summary["initial_assessment_items"] == 7
    assert summary["final_assessment_items"] == 7

    assert summary["assessment_results"][
        "llama3.2:3b"
    ]["final_correct"] == 4

    assert summary["assessment_results"][
        "qwen2.5:3b"
    ]["final_correct"] == 3

    assert summary["trust_decisions"] == {
        "llama3.2:3b": "PRACTICE_REQUIRED",
        "qwen2.5:3b": "PRACTICE_REQUIRED",
    }


def test_generated_artifacts(tmp_path):
    lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert (
        tmp_path / "results/m6_5_final_results.json"
    ).is_file()

    assert (
        tmp_path / "results/manifest.json"
    ).is_file()

    assert (
        tmp_path / "MODULE_6_RESEARCH_FINDINGS.md"
    ).is_file()

    assert (
        tmp_path / "MODULE_7_INTERFACE.json"
    ).is_file()


def test_finalization_is_deterministic(tmp_path):
    first = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    second = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert first == second
