
"""
P5 validator tests.

These tests use saved P1-P4 results and
also corrupt copies to verify failure paths.
"""

import copy
import json

import pytest

import p5_final_analysis as p5


def saved(name):
    return p5.load_report(
        p5.RESULTS / p5.INPUTS[name],
        name,
    )


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        p5.load_report(
            tmp_path / "missing.json",
            "P1",
        )


def test_wrong_experiment_label_rejected(tmp_path):
    path = tmp_path / "wrong.json"

    path.write_text(
        json.dumps({
            "experiment": "P4",
            "repeats": 3,
            "records": [],
            "summary": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        p5.load_report(path, "P1")


def test_offline_p2_not_accepted_as_live(tmp_path):
    data = copy.deepcopy(saved("P2"))
    data["offline"] = True

    path = tmp_path / "p2.json"
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        p5.load_report(path, "P2")


@pytest.mark.parametrize(
    "name,field,validator",
    [
        ("P1", "scenario", p5.validate_p1),
        ("P2", "mode", p5.validate_p2),
        ("P3", "scenario", p5.validate_p3),
        ("P4", "mode", p5.validate_p4),
    ],
)
def test_saved_evidence_is_consistent(
    name, field, validator
):
    data = saved(name)

    p5.validate_repeats(
        data, name, field
    )

    evidence = validator(data)

    assert isinstance(evidence, dict)


@pytest.mark.parametrize(
    "name,field",
    [
        ("P1", "scenario"),
        ("P2", "mode"),
        ("P3", "scenario"),
        ("P4", "mode"),
    ],
)
def test_missing_trial_rejected(name, field):
    data = copy.deepcopy(saved(name))
    data["records"].pop()

    with pytest.raises(ValueError):
        p5.validate_repeats(
            data, name, field
        )


def test_p1_corrupted_summary_rejected():
    data = copy.deepcopy(saved("P1"))

    data["summary"]["policies"][
        "verified_receiver"
    ]["executions"] = 999

    with pytest.raises(ValueError):
        p5.validate_p1(data)


def test_p2_corrupted_summary_rejected():
    data = copy.deepcopy(saved("P2"))

    data["summary"]["total"][
        "signatures_verified"
    ] = 0

    with pytest.raises(ValueError):
        p5.validate_p2(data)


def test_p3_corrupted_summary_rejected():
    data = copy.deepcopy(saved("P3"))

    data["summary"]["policies"][
        "capability_guard"
    ]["invalid_executions"] = 1

    with pytest.raises(ValueError):
        p5.validate_p3(data)


def test_p4_corrupted_summary_rejected():
    data = copy.deepcopy(saved("P4"))

    data["summary"]["total"][
        "capability_blocks"
    ] = 100

    with pytest.raises(ValueError):
        p5.validate_p4(data)


def test_p4_unauthorized_execution_rejected():
    data = copy.deepcopy(saved("P4"))

    row = next(
        r for r in data["records"]
        if r["mode"] == "signed_structured"
    )

    row["executed"] = True
    row["completed"] = False

    with pytest.raises(ValueError):
        p5.validate_p4(data)


def test_p4_protocol_bypass_rejected():
    data = copy.deepcopy(saved("P4"))

    row = next(
        r for r in data["records"]
        if r["mode"] == "tampered_envelope"
    )

    row["worker_called"] = True

    with pytest.raises(ValueError):
        p5.validate_p4(data)


def test_research_checks_pass():
    evidence = {
        "P1": p5.validate_p1(saved("P1")),
        "P2": p5.validate_p2(saved("P2")),
        "P3": p5.validate_p3(saved("P3")),
        "P4": p5.validate_p4(saved("P4")),
    }

    checks = p5.research_checks(evidence)

    assert len(checks) == 20
    assert all(checks.values())


def test_report_mentions_limitations():
    evidence = {
        "P1": p5.validate_p1(saved("P1")),
        "P2": p5.validate_p2(saved("P2")),
        "P3": p5.validate_p3(saved("P3")),
        "P4": p5.validate_p4(saved("P4")),
    }

    checks = p5.research_checks(evidence)
    report = p5.make_findings(
        evidence, checks
    )

    assert "Limitations" in report
    assert "A2A or OpenA2A" in report
    assert "P1" in report
    assert "P4" in report
    assert "20/20" in report


def test_generate_final_artifacts(tmp_path):
    for filename in p5.INPUTS.values():
        source = p5.RESULTS / filename
        target = tmp_path / filename

        target.write_bytes(
            source.read_bytes()
        )

    findings = tmp_path / "FINDINGS.md"

    result = p5.run_analysis(
        results_dir=tmp_path,
        findings_path=findings,
    )

    assert result["passed"] == 20

    assert (
        tmp_path / "p5_final_results.json"
    ).is_file()

    assert findings.is_file()
