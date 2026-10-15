
import copy

import pytest

import guardian_collective_memory as lab
import c2_collective_benchmark as bench


def data():
    return lab.load_verified_sources()


def assembled():
    source = data()
    annotated = lab.annotate_observations(
        source["observations"],
        source["exercises"],
    )
    units = lab.build_evidence_units(annotated)
    collective = lab.build_collective_memory(units)
    return annotated, units, collective


def test_source_count():
    assert len(data()["observations"]) == 30


def test_observations_have_trusted_lesson_ids():
    annotated, _, _ = assembled()

    assert len(annotated) == 30
    assert all(
        row["lesson_id"]
        for row in annotated
    )


def test_model_observation_counts():
    annotated, _, _ = assembled()
    assert sum(
        x["model"] == "llama3.2:3b"
        for x in annotated
    ) == 16
    assert sum(
        x["model"] == "qwen2.5:3b"
        for x in annotated
    ) == 14


def test_correlated_unit_count():
    _, units, _ = assembled()
    assert len(units) == 7


def test_model_unit_counts():
    _, units, _ = assembled()

    assert sum(
        x["model"] == "llama3.2:3b"
        for x in units
    ) == 4
    assert sum(
        x["model"] == "qwen2.5:3b"
        for x in units
    ) == 3


def test_observation_accounting():
    annotated, units, collective = assembled()

    assert sum(
        unit["observations"]
        for unit in units
    ) == len(annotated)

    assert sum(
        item["observations"]
        for item in collective
    ) == len(annotated)


def test_units_preserve_correlation():
    _, units, _ = assembled()

    assert all(
        unit["independent_session_units"] == 1
        for unit in units
    )

    assert any(
        unit["observations"] > 1
        for unit in units
    )


def test_unit_outcomes_consistent():
    _, units, _ = assembled()

    for unit in units:
        assert (
            unit["correct"] + unit["incorrect"]
            == unit["observations"]
        )

        if unit["correct"] and unit["incorrect"]:
            assert unit["outcome_state"] == "MIXED"
        elif unit["correct"]:
            assert unit["outcome_state"] == "ALL_CORRECT"
        else:
            assert unit["outcome_state"] == "ALL_INCORRECT"


def test_collective_provenance_retained():
    _, units, collective = assembled()

    for item in collective:
        assert item["contributing_model_count"] == (
            len(item["contributing_models"])
        )
        assert item["correlated_units"] == len(
            item["per_model"]
        )


def test_no_confidence_overclaim():
    _, _, collective = assembled()

    for item in collective:
        assert item["confidence_calibrated"] is False
        assert (
            item["behavioral_transfer_demonstrated"]
            is False
        )
        assert (
            item["safe_for_automatic_policy_update"]
            is False
        )


def test_no_privilege_restoration():
    _, units, collective = assembled()

    assert all(
        unit["privileges_restored"] is False
        and unit["permissions_changed"] is False
        for unit in units
    )


def test_accepted_source_cannot_be_forged():
    source = data()
    forged = copy.deepcopy(source["observations"])
    forged[0]["observed_correct"] = (
        not forged[0]["observed_correct"]
    )

    with pytest.raises(ValueError):
        lab.annotate_observations(
            forged, source["exercises"]
        )


def test_unknown_exercise_rejected():
    source = data()
    forged = copy.deepcopy(source["observations"])
    forged[0]["exercise_id"] = "fabricated"

    with pytest.raises(ValueError):
        lab.annotate_observations(
            forged, source["exercises"]
        )


def test_duplicate_observation_rejected():
    source = data()
    repeated = copy.deepcopy(source["observations"])
    repeated.append(copy.deepcopy(repeated[0]))

    with pytest.raises(ValueError):
        lab.annotate_observations(
            repeated, source["exercises"]
        )


def test_incorrect_source_phase_rejected():
    source = data()
    forged = copy.deepcopy(source["observations"])

    row = next(
        x for x in forged
        if x["phase"] == "practice"
    )
    row["phase"] = "final_assessment"
    row["evidence_fingerprint"] = lab.digest({
        k: v for k, v in row.items()
        if k != "evidence_fingerprint"
    })

    with pytest.raises(ValueError):
        lab.annotate_observations(
            forged, source["exercises"]
        )


def test_evidence_unit_mutation_rejected():
    annotated, units, collective = assembled()
    bad = copy.deepcopy(units)
    bad[0]["correct"] += 1

    with pytest.raises(ValueError):
        lab.validate_memory(
            annotated, bad, collective
        )


def test_collective_mutation_rejected():
    annotated, units, collective = assembled()
    bad = copy.deepcopy(collective)
    bad[0]["correct"] += 1

    with pytest.raises(ValueError):
        lab.validate_memory(
            annotated, units, bad
        )


def test_correlated_repetitions_do_not_add_units():
    annotated, units, _ = assembled()

    chosen = annotated[0]
    same_key = [
        item for item in annotated
        if (
            item["model"],
            item["session_id"],
            item["lesson_id"],
        ) == (
            chosen["model"],
            chosen["session_id"],
            chosen["lesson_id"],
        )
    ]

    assert len(same_key) >= 1

    # A hypothetical additional correctly linked
    # attempt in the SAME session must not increase
    # the number of distinct evidence units.
    new_attempt = copy.deepcopy(chosen)
    new_attempt["observation_id"] = "synthetic_repeat"

    extended_units = lab.build_evidence_units(
        annotated + [new_attempt]
    )

    assert len(extended_units) == len(units)


def test_collective_memory_deterministic():
    a = assembled()
    b = assembled()
    assert a == b


def test_summary_has_no_authorization_changes():
    annotated, units, collective = assembled()
    summary = lab.make_summary(
        annotated, units, collective
    )

    assert summary["verified_input_observations"] == 30
    assert summary["correlated_evidence_units"] == 7
    assert summary["automatic_policy_updates"] == 0
    assert summary["permission_changes"] == 0
    assert summary["privilege_restorations"] == 0


def test_benchmark_output(tmp_path):
    result = bench.run_benchmark(
        output_dir=tmp_path / "results",
        findings_dir=tmp_path,
    )

    assert result["summary"][
        "verified_input_observations"
    ] == 30
    assert result["summary"][
        "correlated_evidence_units"
    ] == 7
    assert result["summary"][
        "aggregate_tampering_detected"
    ] is True

    for name in (
        "correlated_evidence_units.json",
        "collective_safety_memory.json",
        "m7_2_summary.json",
        "manifest.json",
    ):
        assert (tmp_path / "results" / name).is_file()

    assert (
        tmp_path / "M7_C2_FINDINGS.md"
    ).is_file()
