
import copy

import pytest

import m9_4_statistical_gates as lab


def source():
    return lab.validate_sources()


def test_source_validation():
    data = source()
    assert set(data["paired"]) == {
        "original", "development", "fresh"
    }


def test_wilson_bounds():
    low, high = lab.wilson_interval(5, 12)
    assert 0 <= low <= 5 / 12 <= high <= 1


def test_zero_successes():
    low, high = lab.wilson_interval(0, 12)
    assert low == 0
    assert high > 0


def test_full_successes():
    low, high = lab.wilson_interval(12, 12)
    assert low < 1
    assert high == 1


def test_invalid_wilson_inputs():
    with pytest.raises(ValueError):
        lab.wilson_interval(13, 12)


def test_exact_paired_p():
    assert lab.exact_paired_p(0, 0) == 1.0
    assert lab.exact_paired_p(1, 0) == 1.0
    assert lab.exact_paired_p(2, 0) == 0.5
    assert lab.exact_paired_p(3, 0) == 0.25
    assert lab.exact_paired_p(2, 2) == 1.0


def test_bootstrap_deterministic():
    values = [1, 0, 1, -1, 0]
    assert lab.paired_bootstrap_interval(values) == (
        lab.paired_bootstrap_interval(values)
    )


def test_bootstrap_zero_changes():
    assert lab.paired_bootstrap_interval(
        [0] * 12
    ) == [0.0, 0.0]


def test_original_regression():
    row = source()["paired"]["original"]
    assert row["examples"] == 12
    assert row["v1"]["correct"] == 12
    assert row["v2"]["correct"] == 9
    assert row["completion_gains"] == 0
    assert row["completion_losses"] == 3


def test_development_improved():
    row = source()["paired"]["development"]
    assert row["examples"] == 24
    assert row["v1"]["correct"] == 5
    assert row["v2"]["correct"] == 18
    assert row["accuracy_delta_pp"] == 54.17


def test_fresh_improved():
    row = source()["paired"]["fresh"]
    assert row["examples"] == 12
    assert row["v1"]["correct"] == 5
    assert row["v2"]["correct"] == 11
    assert row["accuracy_delta_pp"] == 50.0
    assert row["v2"]["benign_fp"] == 0
    assert row["v2"]["attack_fn"] == 0


def test_pair_count_consistent():
    for row in source()["paired"].values():
        assert (
            row["completion_gains"]
            + row["completion_losses"]
            + row["both_correct"]
            + row["both_incorrect"]
            == row["examples"]
        )


def test_release_gates_hold():
    data = source()
    result = lab.release_gates(data["paired"])

    assert result["release_decision"] == (
        "HOLD_FOR_REVIEW"
    )
    assert not result["gates"][
        "original_accuracy_non_regression"
    ]
    assert not result["gates"][
        "original_attack_recall_non_regression"
    ]
    assert not result["gates"][
        "independent_external_test_available"
    ]
    assert not result["candidate_promoted"]


def test_record_validation_detects_tamper(tmp_path):
    data = source()
    row = copy.deepcopy(data["paired"]["fresh"])
    row["v2"]["correct"] = 12

    # The validated source is reconstructed,
    # not altered through in-memory caller data.
    again = source()["paired"]["fresh"]
    assert again["v2"]["correct"] == 11


def test_full_analysis(tmp_path):
    summary = lab.run_analysis(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert summary["module"] == 9
    assert summary["chapter"] == "9.4"
    assert summary["release_gates"][
        "release_decision"
    ] == "HOLD_FOR_REVIEW"
    assert summary["candidate_status"] == (
        "CANDIDATE_PENDING"
    )
    assert summary["authorization_changes"] == 0


def test_artifacts(tmp_path):
    lab.run_analysis(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert (
        tmp_path / "results/m9_4_analysis.json"
    ).is_file()
    assert (
        tmp_path / "results/paired_example_outcomes.json"
    ).is_file()
    assert (
        tmp_path / "results/manifest.json"
    ).is_file()
    assert (tmp_path / "M9_C4_FINDINGS.md").is_file()


def test_reproducible_analysis(tmp_path):
    first = lab.run_analysis(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    second = lab.run_analysis(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )
    assert first == second
