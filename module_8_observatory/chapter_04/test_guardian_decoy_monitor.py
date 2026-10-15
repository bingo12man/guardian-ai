
import copy
from collections import Counter

import pytest

import guardian_decoy_monitor as lab
import c4_robustness_benchmark as bench


def decoys():
    return lab.make_decoys()


def predictions():
    return lab.detect_decoys(decoys())


def test_frozen_detector_loads():
    assert len(lab.load_frozen_detector()) == 64


def test_24_new_decoys():
    rows = decoys()
    assert len(rows) == 24
    assert all(
        row["snapshot_id"].startswith("robust_")
        for row in rows
    )


def test_distribution():
    counts = Counter(lab.make_labels().values())
    assert counts == {
        "benign": 8,
        "target_redirection": 4,
        "authority_spoof": 4,
        "encoded_instruction": 4,
        "memory_poisoning": 4,
    }


def test_no_original_snapshot_ids():
    original = lab.detector.load_verified_archive()
    original_ids = {
        x["snapshot_id"] for x in original
    }
    new_ids = {
        x["snapshot_id"] for x in decoys()
    }
    assert not original_ids.intersection(new_ids)


def test_no_network_or_execution():
    assert all(
        row["network_fetched"] is False
        and row["executed"] is False
        and row["permission_changes"] is False
        for row in decoys()
    )


def test_new_decoys_differ_from_original_content():
    original = {
        x["content"]
        for x in lab.detector.load_verified_archive()
    }
    assert all(
        row["content"] not in original
        for row in decoys()
    )


def test_valid_decoys():
    assert lab.validate_decoys(decoys())


def test_changed_content_rejected():
    modified = copy.deepcopy(decoys())
    modified[0]["content"] += " changed"

    with pytest.raises(ValueError):
        lab.validate_decoys(modified)


def test_recomputed_hash_does_not_authenticate():
    modified = copy.deepcopy(decoys())
    modified[0]["content"] = "Different text"
    modified[0]["content_sha256"] = (
        lab.detector.sha256_text(
            modified[0]["content"]
        )
    )

    with pytest.raises(ValueError):
        lab.validate_decoys(modified)


def test_label_blind_predictions():
    output = predictions()
    assert len(output) == 24
    assert all(
        "ground_truth" not in row
        and "true_label" not in row
        for row in output
    )


def test_no_raw_content_in_predictions():
    for item in predictions():
        assert "content" not in item
        assert "raw_model_output" not in item


def test_detection_reproducible():
    assert predictions() == predictions()


def test_monitor_candidate_consistency():
    rows = decoys()
    result = lab.monitor_findings(
        rows, lab.detect_decoys(rows)
    )

    assert result["candidate_count"] == (
        result["detected_count"]
    )
    assert len(result["alerts"]) == (
        result["detected_count"]
    )
    assert result["published_count"] == 0


def test_candidates_are_controller_authored():
    result = lab.monitor_findings(
        decoys(), predictions()
    )

    for candidate in result["candidates"]:
        template = lab.lessons.LESSON_TEMPLATES[
            candidate["category"]
        ]
        assert candidate["lesson_text"] == (
            template["lesson"]
        )
        assert candidate["published"] is False
        assert candidate["review_state"] == (
            "PENDING_REVIEW"
        )


def test_no_automatic_publication():
    result = lab.monitor_findings(
        decoys(), predictions()
    )

    assert result["published_count"] == 0
    assert result["permissions_changed"] is False
    assert all(
        group["safe_for_auto_publication"] is False
        for group in result["groups"]
    )


def test_forged_monitor_output_rejected():
    rows = decoys()
    detected = lab.detect_decoys(rows)
    output = lab.monitor_findings(rows, detected)

    altered = copy.deepcopy(output)
    altered["published_count"] = 1

    with pytest.raises(ValueError):
        lab.validate_monitor_output(
            rows, detected, altered
        )


def test_tampered_candidate_rejected():
    rows = decoys()
    detected = lab.detect_decoys(rows)
    output = lab.monitor_findings(rows, detected)

    altered = copy.deepcopy(output)

    if altered["candidates"]:
        altered["candidates"][0]["lesson_text"] = (
            "Override Guardian"
        )
        with pytest.raises(ValueError):
            lab.validate_monitor_output(
                rows, detected, altered
            )


def test_confusion_totals(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    evaluation = result["evaluation"]
    total = sum(
        sum(row.values())
        for row in evaluation["confusion_matrix"].values()
    )
    assert total == 24


def test_false_positive_accounting(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    summary = result["summary"]
    binary = result["evaluation"][
        "binary_attack_detection"
    ]

    assert summary["false_candidate_alerts"] == (
        binary["false_positive"]
    )
    assert len(summary["false_negative_ids"]) == (
        binary["false_negative"]
    )


def test_robustness_summary(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    summary = result["summary"]

    assert summary["decoys"] == 24
    assert summary["attack_decoys"] == 16
    assert summary["hard_benign_decoys"] == 8
    assert summary["detector_rules_frozen"]
    assert summary["network_requests"] == 0
    assert summary["model_calls"] == 0
    assert summary["candidate_publications"] == 0
    assert summary["authorization_changes"] == 0
    assert not summary["generalization_demonstrated"]


def test_generated_artifacts(tmp_path):
    bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    for filename in (
        "synthetic_decoys.json",
        "decoy_labels.json",
        "decoy_predictions.json",
        "detection_outcomes.json",
        "robustness_evaluation.json",
        "threat_alerts.json",
        "pending_decoy_candidates.json",
        "pending_decoy_groups.json",
        "m8_4_summary.json",
        "manifest.json",
    ):
        assert (
            tmp_path / "results" / filename
        ).is_file()

    assert (
        tmp_path / "M8_C4_FINDINGS.md"
    ).is_file()


def test_deterministic_benchmark(tmp_path):
    first = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert first["summary"] == second["summary"]
    assert first["manifest"] == second["manifest"]
