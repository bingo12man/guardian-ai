
import copy
import json

import pytest

import m9_5_final_benchmark as lab


def validated():
    return lab.validate_all()


def test_all_sources_exist():
    assert all(
        path.is_file()
        for path in lab.SOURCES.values()
    )


def test_baseline_registry():
    source = validated()["baseline"]

    assert len(source["records"]) == 2
    assert all(
        row["status"] == "FROZEN_BASELINE"
        for row in source["records"]
    )
    assert len(source["head"]) == 64


def test_candidate_registry():
    source = validated()["candidate"]

    assert len(source["records"]) == 3
    assert all(
        row["status"] == "CANDIDATE_PENDING"
        for row in source["records"]
    )
    assert len(source["head"]) == 64


def test_dataset_hashes_are_distinct():
    source = validated()["candidate"]

    hashes = {
        row["dataset_sha256"]
        for row in source["records"]
    }

    assert len(hashes) == 3


def test_original_lessons_still_pending():
    lessons = validated()["lessons"]
    pending = lessons["pending"]

    assert pending["lessons"] == 8
    assert pending["pending_review"] == 8
    assert pending["approved"] == 0
    assert pending["active"] == 0
    assert pending["rolled_back"] == 0


def test_fixture_was_rolled_back():
    fixture = validated()["lessons"]["fixture"]

    assert fixture["lessons"] == 1
    assert fixture["rolled_back"] == 1
    assert fixture["active"] == 0
    assert fixture["audit_events"] == 4


def test_statistical_gates():
    gates = validated()["statistics"]["gates"]

    assert gates["release_decision"] == "HOLD_FOR_REVIEW"
    assert gates["passed"] == 3
    assert gates["failed"] == 6
    assert gates["candidate_promoted"] is False


def test_original_regression():
    paired = validated()["statistics"]["paired"]
    original = paired["original"]

    assert original["v1"]["correct"] == 12
    assert original["v2"]["correct"] == 9
    assert original["completion_gains"] == 0
    assert original["completion_losses"] == 3
    assert original["accuracy_delta_pp"] == -25.0


def test_development_improvement():
    paired = validated()["statistics"]["paired"]
    development = paired["development"]

    assert development["v1"]["correct"] == 5
    assert development["v2"]["correct"] == 18
    assert development["completion_gains"] == 13
    assert development["completion_losses"] == 0


def test_fresh_synthetic_improvement():
    paired = validated()["statistics"]["paired"]
    fresh = paired["fresh"]

    assert fresh["v1"]["correct"] == 5
    assert fresh["v2"]["correct"] == 11
    assert fresh["completion_gains"] == 6
    assert fresh["completion_losses"] == 0


def test_final_result_has_no_promotion():
    final = lab.make_final(validated())

    assert final["candidate_promoted"] is False
    assert final["baseline_replaced"] is False
    assert final["live_guardian_lessons_activated"] == 0
    assert final["authorization_changes"] == 0
    assert final["real_world_generalization_established"] is False


def test_original_lesson_tamper_rejected(tmp_path):
    paths = dict(lab.SOURCES)

    altered = lab.read_json(paths["lesson_summary"])
    altered["active_module_8_lessons"] = 1

    path = tmp_path / "lesson_summary.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["lesson_summary"] = path

    with pytest.raises(ValueError):
        lab.validate_all(paths)


def test_release_gate_tamper_rejected(tmp_path):
    paths = dict(lab.SOURCES)

    altered = lab.read_json(paths["statistical_analysis"])
    altered["release_gates"]["release_decision"] = (
        "ELIGIBLE_FOR_SEPARATE_REVIEW"
    )

    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["statistical_analysis"] = path

    with pytest.raises(ValueError):
        lab.validate_all(paths)


def test_paired_outcome_tamper_rejected(tmp_path):
    paths = dict(lab.SOURCES)

    altered = lab.read_json(paths["paired_outcomes"])
    altered["fresh"][0]["v2_correct"] = False

    path = tmp_path / "paired.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["paired_outcomes"] = path

    with pytest.raises(ValueError):
        lab.validate_all(paths)


def test_candidate_manifest_tamper_rejected(tmp_path):
    paths = dict(lab.SOURCES)

    altered = lab.read_json(paths["candidate_manifest"])
    altered["audit_head"] = "0" * 64

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["candidate_manifest"] = path

    with pytest.raises(ValueError):
        lab.validate_all(paths)


def test_missing_evidence_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    paths["baseline_summary"] = (
        tmp_path / "missing.json"
    )

    with pytest.raises(ValueError):
        lab.validate_all(paths)


def test_module_10_contract():
    final = lab.make_final(validated())
    contract = lab.make_module_10_contract(final)

    assert contract["producer_module"] == 9
    assert contract["consumer_module"] == 10
    assert contract["integration_mode"] == (
        "read_only_research"
    )
    assert contract["module_9_final_sha256"] == (
        lab.digest(final)
    )


def test_module_10_cannot_change_permissions():
    contract = lab.make_module_10_contract(
        lab.make_final(validated())
    )

    policy = contract["authorization_constraints"]

    assert all(value is False for value in policy.values())
    assert policy["automatic_detector_promotion"] is False
    assert policy["automatic_lesson_approval"] is False
    assert policy["agents_change_permissions"] is False


def test_finalization(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    summary = result["summary"]

    assert summary["evaluation_status"] == (
        "FULL_RECORD_ANALYSIS"
    )
    assert summary["baseline_experiments"] == 2
    assert summary["candidate_experiments"] == 3
    assert summary["lessons_pending"] == 8
    assert summary["lessons_active"] == 0
    assert summary["release_decision"] == (
        "HOLD_FOR_REVIEW"
    )
    assert summary["release_gates_failed"] == 6
    assert summary["module_10_contract_valid"]


def test_generated_files(tmp_path):
    lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for relative in (
        "results/m9_5_final_results.json",
        "results/manifest.json",
        "MODULE_9_RESEARCH_FINDINGS.md",
        "MODULE_10_INTERFACE.json",
    ):
        assert (tmp_path / relative).is_file()


def test_repeatable_finalization(tmp_path):
    first = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert first == second
