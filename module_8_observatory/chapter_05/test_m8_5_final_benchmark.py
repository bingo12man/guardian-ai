
import copy
import json

import pytest

import m8_5_final_benchmark as lab


def evidence():
    return lab.validate_sources()


def test_all_source_files_exist():
    assert all(
        path.is_file()
        for path in lab.SOURCES.values()
    )


def test_full_evidence_chain():
    source = evidence()

    assert source["original"]["examples"] == 12
    assert source["robustness"]["examples"] == 24
    assert len(source["original_candidates"]) == 8
    assert len(source["robustness_candidates"]) == 5


def test_original_metrics():
    source = evidence()
    evaluation = source["original"]

    assert evaluation["correct"] == 12
    assert evaluation["accuracy"] == 1.0
    assert evaluation["macro_f1"] == 1.0
    assert evaluation[
        "binary_attack_detection"
    ]["true_positive"] == 8


def test_robustness_metrics():
    source = evidence()
    evaluation = source["robustness"]
    binary = evaluation["binary_attack_detection"]

    assert evaluation["correct"] == 5
    assert evaluation["accuracy"] == 0.2083
    assert evaluation["macro_f1"] == 0.1164

    assert binary["true_positive"] == 1
    assert binary["false_positive"] == 4
    assert binary["false_negative"] == 15
    assert binary["true_negative"] == 4
    assert binary["precision"] == 0.2
    assert binary["recall"] == 0.0625


def test_false_positive_ids():
    assert evidence()["false_positive_ids"] == [
        "robust_19",
        "robust_20",
        "robust_21",
        "robust_22",
    ]


def test_false_negative_count():
    assert len(evidence()["false_negative_ids"]) == 15


def test_robustness_candidate_count():
    source = evidence()
    assert len(source["robustness_alerts"]) == 5
    assert len(source["robustness_candidates"]) == 5
    assert all(
        row["published"] is False
        for row in source["robustness_candidates"]
    )


def test_accuracy_gap():
    final = lab.make_final(evidence())
    gap = final["performance_gap"]

    assert gap["accuracy_drop_pp"] == 79.17
    assert gap["binary_recall_drop_pp"] == 93.75


def test_no_behavioral_overclaim():
    final = lab.make_final(evidence())

    assert final[
        "real_world_generalization_established"
    ] is False
    assert final[
        "model_behavioral_improvement_established"
    ] is False
    assert final["authorization_changes"] == 0
    assert final["network_requests"] == 0


def test_module_9_contract():
    final = lab.make_final(evidence())
    contract = lab.make_module_9_contract(final)

    assert contract["producer_module"] == 8
    assert contract["consumer_module"] == 9
    assert contract["research_status"] == (
        "robustness_failure_observed"
    )
    assert contract["source_module_8_sha256"] == (
        lab.digest(final)
    )

    policy = contract["authorization_constraints"]
    assert all(value is False for value in policy.values())


def test_tampered_original_predictions(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["c2_predictions"])
    altered[0]["predicted_category"] = "authority_spoof"

    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["c2_predictions"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_lesson_candidate(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["c3_candidates"])
    altered[0]["published"] = True

    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["c3_candidates"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_robustness_prediction(tmp_path):
    paths = dict(lab.SOURCES)

    altered = lab.read_json(paths["c4_predictions"])

    # Guarantee a change to the stored prediction.
    original = altered[0]["predicted_category"]

    altered[0]["predicted_category"] = (
        "authority_spoof"
        if original != "authority_spoof"
        else "benign"
    )

    assert altered[0]["predicted_category"] != original

    path = tmp_path / "robust_predictions.json"

    path.write_text(
        json.dumps(altered),
        encoding="utf-8",
    )

    paths["c4_predictions"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_ground_truth(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["c4_labels"])
    altered["robust_01"] = "benign"

    path = tmp_path / "labels.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["c4_labels"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_final_summary_source(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["c4_summary"])
    altered["candidate_publications"] = 1

    path = tmp_path / "summary.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["c4_summary"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_tampered_detector_rule_hash(tmp_path):
    paths = dict(lab.SOURCES)
    altered = lab.read_json(paths["c4_manifest"])
    altered["frozen_rule_sha256"] = "0" * 64

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    paths["c4_manifest"] = path

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_missing_source_rejected(tmp_path):
    paths = dict(lab.SOURCES)
    paths["c4_decoys"] = tmp_path / "missing.json"

    with pytest.raises(ValueError):
        lab.validate_sources(paths)


def test_final_outputs(tmp_path):
    result = lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    summary = result["summary"]

    assert summary["original_accuracy"] == 1.0
    assert summary["robustness_accuracy"] == 0.2083
    assert summary["robustness_false_positives"] == 4
    assert summary["robustness_false_negatives"] == 15
    assert summary["published_lessons"] == 0
    assert summary["authorization_changes"] == 0
    assert summary["module_9_contract_valid"]


def test_output_files(tmp_path):
    lab.run_final(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for relative in (
        "results/m8_5_final_results.json",
        "results/manifest.json",
        "MODULE_8_RESEARCH_FINDINGS.md",
        "MODULE_9_INTERFACE.json",
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
