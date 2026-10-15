
import copy
import json

import pytest

import a5_final_analysis as a5


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        a5.load_report(
            tmp_path / "missing.json"
        )


def test_offline_report_is_rejected(tmp_path):
    path = tmp_path / "offline.json"

    path.write_text(
        json.dumps({
            "offline": True,
            "repeats": 3,
            "records": [],
            "summary": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        a5.load_report(path)


def test_missing_records_rejected(tmp_path):
    path = tmp_path / "invalid.json"

    path.write_text(
        json.dumps({
            "offline": False,
            "repeats": 3,
            "summary": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        a5.load_report(path)


def test_missing_repeat_is_rejected():
    data = {
        "records": [
            {"scenario": "visitor_public", "repeat": 1}
        ]
    }

    with pytest.raises(ValueError):
        a5.validate_trials(
            data,
            "scenario",
            a5.A1_CASES,
        )


def test_duplicate_repeat_is_rejected():
    records = []

    for scenario in a5.A1_CASES:
        for repeat in (1, 2, 3):
            records.append({
                "scenario": scenario,
                "repeat": repeat,
            })

    records[1]["repeat"] = 1

    with pytest.raises(ValueError):
        a5.validate_trials(
            {"records": records},
            "scenario",
            a5.A1_CASES,
        )


def test_count_true_only_counts_boolean_true():
    rows = [
        {"passed": True},
        {"passed": False},
        {"passed": 1},
        {"passed": None},
    ]

    assert a5.count_true(rows, "passed") == 1


def load_saved(name):
    return a5.load_report(
        a5.RESULTS / a5.INPUTS[name]
    )


@pytest.mark.parametrize(
    "name,validator",
    [
        ("A1", a5.validate_a1),
        ("A2", a5.validate_a2),
        ("A3", a5.validate_a3),
        ("A4", a5.validate_a4),
    ],
)
def test_saved_live_experiment_is_consistent(
    name, validator
):
    data = load_saved(name)

    evidence = validator(data)

    assert isinstance(evidence, dict)


def test_corrupted_a1_summary_rejected():
    data = copy.deepcopy(load_saved("A1"))

    data["summary"]["guardian_blocks"] = 99

    with pytest.raises(ValueError):
        a5.validate_a1(data)


def test_corrupted_a2_summary_rejected():
    data = copy.deepcopy(load_saved("A2"))

    data["summary"]["policies"][
        "worker_only"
    ]["unauthorized_simulated_executions"] = 0

    with pytest.raises(ValueError):
        a5.validate_a2(data)


def test_corrupted_a3_summary_rejected():
    data = copy.deepcopy(load_saved("A3"))

    data["summary"][
        "structured_spoofing"
    ]["confidential_proposals"] = 0

    with pytest.raises(ValueError):
        a5.validate_a3(data)


def test_corrupted_a4_recovery_rejected():
    data = copy.deepcopy(load_saved("A4"))

    recovery = next(
        r for r in data["records"]
        if r["mode"] == "feedback_retry"
    )

    recovery["recovered"] = False

    with pytest.raises(ValueError):
        a5.validate_a4(data)


def test_a4_retry_bound_enforced():
    data = copy.deepcopy(load_saved("A4"))

    row = data["records"][0]

    while len(row["attempts"]) <= 3:
        row["attempts"].append(
            copy.deepcopy(row["attempts"][-1])
        )

    row["attempt_count"] = len(row["attempts"])

    with pytest.raises(ValueError):
        a5.validate_a4(data)


def test_all_research_claims():
    evidence = {
        "A1": a5.validate_a1(
            load_saved("A1")
        ),
        "A2": a5.validate_a2(
            load_saved("A2")
        ),
        "A3": a5.validate_a3(
            load_saved("A3")
        ),
        "A4": a5.validate_a4(
            load_saved("A4")
        ),
    }

    checks = a5.research_checks(evidence)

    assert len(checks) == 14
    assert all(checks.values())


def test_report_contains_limitations():
    evidence = {
        "A1": a5.validate_a1(
            load_saved("A1")
        ),
        "A2": a5.validate_a2(
            load_saved("A2")
        ),
        "A3": a5.validate_a3(
            load_saved("A3")
        ),
        "A4": a5.validate_a4(
            load_saved("A4")
        ),
    }

    checks = a5.research_checks(evidence)
    markdown = a5.render_findings(
        evidence, checks
    )

    assert "Limitations" in markdown
    assert "Historical A4 pilot" in markdown
    assert "AutoGen and CAMEL" in markdown
    assert "14/14" in markdown


def test_report_generation(tmp_path):
    source_dir = a5.RESULTS

    for filename in a5.INPUTS.values():
        source = source_dir / filename
        target = tmp_path / filename

        target.write_bytes(
            source.read_bytes()
        )

    findings = tmp_path / "FINDINGS.md"

    report = a5.run_analysis(
        results_dir=tmp_path,
        findings_path=findings,
    )

    assert report["passed"] == 14

    assert (
        tmp_path / "a5_final_results.json"
    ).exists()

    assert findings.exists()

    assert report["historical_a4_pilot"][
        "validated_from_file"
    ] is False
