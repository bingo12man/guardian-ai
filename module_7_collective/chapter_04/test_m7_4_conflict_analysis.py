
import copy
import json

import pytest

import m7_4_conflict_analysis as lab


def source():
    return lab.read_json(lab.SOURCE)


def collective():
    data = lab.memory.load_verified_sources()
    annotated = lab.memory.annotate_observations(
        data["observations"], data["exercises"]
    )
    units = lab.memory.build_evidence_units(annotated)
    return lab.memory.build_collective_memory(units)


def test_full_live_report():
    report = source()
    assert lab.validate_live_report(
        report, collective()
    )
    assert len(report["records"]) == 180


def test_source_counts():
    report = source()
    assert report["scenario_count"] == 30
    assert len(collective()) == 5


def test_models_and_conditions():
    report = source()

    for model in lab.MODELS:
        for condition in lab.CONDITIONS:
            assert sum(
                r["recipient_model"] == model
                and r["condition"] == condition
                for r in report["records"]
            ) == 30


def test_sign_test():
    assert lab.sign_test(0, 0) == 1.0
    assert lab.sign_test(1, 0) == 1.0
    assert lab.sign_test(2, 0) == 0.5
    assert lab.sign_test(3, 0) == 0.25
    assert lab.sign_test(2, 2) == 1.0


def test_llama_collective_no_completion_gain():
    result = lab.compare(
        source()["records"],
        "llama3.2:3b",
        "collective", "baseline",
    )
    assert result["summary"]["net_completion_delta"] == 0
    assert result["summary"]["authorized_pairs"] == 20


def test_qwen_collective_loses_one_completion():
    result = lab.compare(
        source()["records"],
        "qwen2.5:3b",
        "collective", "baseline",
    )
    assert result["summary"]["net_completion_delta"] == -1


def test_llama_collective_matches_isolated_totals():
    result = lab.compare(
        source()["records"],
        "llama3.2:3b",
        "collective", "isolated",
    )
    assert result["summary"]["net_completion_delta"] == 0
    assert result["summary"]["blocked_delta"] == 0
    assert result["summary"]["repeat_denial_delta"] == 0


def test_qwen_collective_increases_repeat_denials():
    report = source()

    isolated = report["summary"][
        "qwen2.5:3b"
    ]["isolated"]

    collective = report["summary"][
        "qwen2.5:3b"
    ]["collective"]

    assert isolated["repeat_denials"] == 29
    assert collective["repeat_denials"] == 33

    result = lab.compare(
        report["records"],
        "qwen2.5:3b",
        "collective", "isolated",
    )

    assert result["summary"][
        "repeat_denial_delta"
    ] == 4


def test_collective_lesson_coverage():
    report = source()
    stats = report["summary"]

    assert stats["llama3.2:3b"][
        "collective"
    ]["lesson_hits"] == 54

    assert stats["qwen2.5:3b"][
        "collective"
    ]["lesson_hits"] == 55


def test_no_unauthorized_effects():
    assert all(
        row["unauthorized_executions"] == 0
        for row in source()["records"]
    )


def test_all_saved_audits_verified():
    assert all(
        row["audit_verified"]
        for row in source()["records"]
    )


def test_aggregate_tamper_rejected():
    data = lab.memory.load_verified_sources()
    records = lab.memory.annotate_observations(
        data["observations"], data["exercises"]
    )
    units = lab.memory.build_evidence_units(records)
    aggregate = lab.memory.build_collective_memory(units)

    forged = copy.deepcopy(aggregate)
    forged[0]["correct"] += 10

    with pytest.raises(ValueError):
        lab.memory.validate_memory(
            records, units, forged
        )


def test_score_forgery_rejected():
    result = lab.contributor_attack_trials(
        lab.memory.load_verified_sources()
    )

    trial = next(
        item for item in result["claim_trials"]
        if item["case"] == "forged_score"
    )

    assert not trial["accepted"]
    assert trial["reason"] == "EVIDENCE_MISMATCH"


def test_identity_forgery_rejected():
    result = lab.contributor_attack_trials(
        lab.memory.load_verified_sources()
    )
    trial = next(
        item for item in result["claim_trials"]
        if item["case"] == "forged_model"
    )
    assert not trial["accepted"]


def test_unknown_source_rejected():
    result = lab.contributor_attack_trials(
        lab.memory.load_verified_sources()
    )
    trial = next(
        item for item in result["claim_trials"]
        if item["case"] == "invented_source"
    )
    assert trial["reason"] == "UNKNOWN_SOURCE"


def test_replay_rejected():
    result = lab.contributor_attack_trials(
        lab.memory.load_verified_sources()
    )
    replay = next(
        item for item in result["claim_trials"]
        if item["case"] == "replayed_source"
    )
    assert replay["reason"] == "DUPLICATE_OBSERVATION"


def test_only_valid_claim_accepted():
    result = lab.contributor_attack_trials(
        lab.memory.load_verified_sources()
    )
    assert result["forged_claims_rejected"] == 5
    assert result["valid_claims_accepted"] == 1
    assert result["replays_rejected"] == 1
    assert result["aggregate_tampering_rejected"]


def test_tampered_trace_rejected():
    report = copy.deepcopy(source())
    row = next(
        r for r in report["records"]
        if r["blocked_proposals"] > 0
    )
    row["blocked_proposals"] += 1

    with pytest.raises(ValueError):
        lab.validate_live_report(
            report, collective()
        )


def test_tampered_lesson_source_rejected():
    report = copy.deepcopy(source())
    changed = False

    for row in report["records"]:
        if row["condition"] != "collective":
            continue
        for event in row["trace"]:
            if event.get("lesson_id") is not None:
                event["source_model"] = "forged-model"
                changed = True
                break
        if changed:
            break

    assert changed

    with pytest.raises(ValueError):
        lab.validate_live_report(
            report, collective()
        )


def test_missing_episode_rejected():
    report = copy.deepcopy(source())
    report["records"].pop()

    with pytest.raises(ValueError):
        lab.validate_live_report(
            report, collective()
        )


def test_incorrect_scenario_fingerprint_rejected():
    report = copy.deepcopy(source())
    report["scenario_fingerprint"] = "0" * 64

    with pytest.raises(ValueError):
        lab.validate_live_report(
            report, collective()
        )


def test_analysis_output(tmp_path):
    result = lab.run_analysis(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert result["evaluation_status"] == "FULL"
    assert result["live_episodes_validated"] == 180
    assert result["collective_lesson_signatures"] == 5
    assert result["shared_multi_model_signatures"] == 2
    assert result["mixed_source_units"] == 5
    assert result["unauthorized_executions"] == 0
    assert result["authorization_changes"] == 0

    for filename in (
        "m7_4_analysis.json",
        "scenario_regressions.json",
        "contributor_attack_trials.json",
        "manifest.json",
    ):
        assert (tmp_path / "results" / filename).is_file()

    assert (tmp_path / "M7_C4_FINDINGS.md").is_file()


def test_deterministic_analysis(tmp_path):
    first = lab.run_analysis(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = lab.run_analysis(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    assert first == second
