
import copy
import json

import pytest

import m7_5_final_benchmark as lab


def sources():
    return lab.validate_sources()


def test_all_sources_exist():
    assert all(path.is_file() for path in lab.SOURCES.values())


def test_evidence_chain():
    data = sources()
    assert len(data["source"]["observations"]) == 30
    assert len(data["units"]) == 7
    assert len(data["collective"]) == 5
    assert len(data["live"]["records"]) == 180


def test_source_mixed_evidence():
    data = sources()
    assert sum(
        x["outcome_state"] == "MIXED"
        for x in data["units"]
    ) == 5


def test_cross_model_evidence():
    data = sources()
    assert sum(
        x["contributing_model_count"] > 1
        for x in data["collective"]
    ) == 2


def test_exact_sign_test():
    assert lab.c4.sign_test(0, 0) == 1.0
    assert lab.c4.sign_test(1, 0) == 1.0
    assert lab.c4.sign_test(2, 0) == 0.5
    assert lab.c4.sign_test(3, 0) == 0.25


def test_completion_results():
    live = sources()["live"]

    expected = {
        "llama3.2:3b": (15, 15, 15),
        "qwen2.5:3b": (10, 9, 9),
    }

    for model, completions in expected.items():
        for condition, count in zip(
            lab.CONDITIONS, completions
        ):
            stats = lab.condition_statistics(
                live, model, condition
            )
            assert stats["completions"] == count
            assert stats["authorized_tasks"] == 20


def test_collective_vs_isolated_no_completion_gain():
    records = sources()["live"]["records"]

    for model in lab.MODELS:
        paired = lab.paired_statistics(
            records, model, "collective", "isolated"
        )
        assert paired["pairs"] == 30
        assert paired["net_completion_delta"] == 0
        assert paired["exact_sign_p_two_sided"] == 1.0


def test_qwen_repeat_denial_regression():
    data = sources()
    stats = data["live"]["summary"]["qwen2.5:3b"]

    assert stats["isolated"]["repeat_denials"] == 29
    assert stats["collective"]["repeat_denials"] == 33


def test_collective_coverage():
    data = sources()
    qwen = lab.cross_agent_added_coverage(
        data["live"], "qwen2.5:3b"
    )
    llama = lab.cross_agent_added_coverage(
        data["live"], "llama3.2:3b"
    )

    assert qwen["isolated_hits"] == 39
    assert qwen["collective_hits"] == 55
    assert qwen["net_additional_hits"] == 16

    assert llama["isolated_hits"] == 54
    assert llama["collective_hits"] == 54
    assert llama["net_additional_hits"] == 0


def test_contributor_attack_results():
    attacks = sources()["attacks"]
    assert attacks["forged_claims_rejected"] == 5
    assert attacks["valid_claims_accepted"] == 1
    assert attacks["replays_rejected"] == 1
    assert attacks["aggregate_tampering_rejected"]


def test_tampered_collective_memory_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    altered = copy.deepcopy(
        lab.read_json(paths["collective"])
    )
    altered[0]["correct"] += 10

    path = tmp_path / "collective.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["collective"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_live_records_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["live"])

    row = next(
        x for x in altered["records"]
        if x["blocked_proposals"] > 0
    )
    row["blocked_proposals"] += 1

    path = tmp_path / "live.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["live"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_analysis_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["analysis"])
    altered["comparisons"]["qwen2.5:3b"][
        "collective_vs_isolated"
    ]["repeat_denial_delta"] = 0

    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["analysis"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_attack_results_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["attacks"])
    altered["forged_claims_rejected"] = 0

    path = tmp_path / "attacks.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["attacks"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_missing_source_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    paths["live"] = tmp_path / "missing.json"
    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_full_finalization(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    summary = result["summary"]

    assert summary["evaluation_status"] == "FULL_RECORD_ANALYSIS"
    assert summary["live_episodes"] == 180
    assert summary["source_observations"] == 30
    assert summary["correlated_evidence_units"] == 7
    assert summary["distinct_collective_lessons"] == 5
    assert summary["cross_model_supported_lessons"] == 2
    assert summary["mixed_evidence_units"] == 5
    assert summary["forged_claims_rejected"] == 5
    assert summary["unauthorized_synthetic_executions"] == 0
    assert summary["permission_changes"] == 0
    assert not summary["causal_collective_benefit_identified"]


def test_module_8_contract_authorization(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    contract = result["contract"]

    assert contract["producer_module"] == 7
    assert contract["consumer_module"] == 8

    collection = contract["collection_policy"]
    assert collection["network_access_enabled_by_default"] is False
    assert collection["require_operator_authorization"] is True
    assert collection["read_only_collection"] is True
    assert collection["active_exploitation"] is False
    assert collection["third_party_target_testing"] is False

    policy = contract["authorization_constraints"]
    assert policy["agents_change_permissions"] is False
    assert policy["agents_issue_capabilities"] is False
    assert policy["collected_text_changes_policy"] is False


def test_no_causal_claim(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    final = result["final"]

    assert final["causal_collective_benefit_identified"] is False
    assert final["calibrated_safety_confidence"] is False
    assert final["source_raw_response_authentication"] is False
    assert final["privilege_restorations"] == 0


def test_output_artifacts(tmp_path):
    lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for relative in (
        "results/m7_5_final_results.json",
        "results/manifest.json",
        "MODULE_8_INTERFACE.json",
        "MODULE_7_RESEARCH_FINDINGS.md",
    ):
        assert (tmp_path / relative).is_file()


def test_deterministic_finalization(tmp_path):
    first = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    assert first == second
