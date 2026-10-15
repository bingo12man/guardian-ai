
import copy
import json

import pytest

import r5_final_analysis as r5


def fixture(name):
    return r5.load_report(
        r5.RESULTS / r5.FILENAMES[name],
        name,
    )


@pytest.mark.parametrize(
    "name", ["R1", "R2", "R3", "R4"]
)
def test_saved_live_evidence(name):
    report = fixture(name)
    summary = r5.validate_records(
        report, name
    )
    assert summary == report["summary"]


@pytest.mark.parametrize(
    "name", ["R1", "R2", "R3", "R4"]
)
def test_missing_trial_is_rejected(name):
    report = copy.deepcopy(fixture(name))
    report["records"].pop()

    with pytest.raises(ValueError):
        r5.validate_records(report, name)


@pytest.mark.parametrize(
    "name", ["R1", "R2", "R3", "R4"]
)
def test_corrupted_summary_rejected(name):
    report = copy.deepcopy(fixture(name))

    report["summary"]["trials"] = -1

    with pytest.raises(ValueError):
        r5.validate_records(report, name)


@pytest.mark.parametrize(
    "name", ["R1", "R2", "R3", "R4"]
)
def test_unauthorized_execution_rejected(name):
    report = copy.deepcopy(fixture(name))

    report["records"][0][
        "unauthorized_executions"
    ] = 1

    with pytest.raises(ValueError):
        r5.validate_records(report, name)


def test_invalid_experiment_label(tmp_path):
    path = tmp_path / "wrong.json"

    path.write_text(
        json.dumps({
            "experiment": "R2",
            "offline": False,
            "repeats": 3,
            "records": [],
            "summary": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        r5.load_report(path, "R1")


def test_offline_not_accepted_as_live(tmp_path):
    report = copy.deepcopy(fixture("R2"))
    report["offline"] = True

    path = tmp_path / "offline.json"

    path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        r5.load_report(path, "R2")


def test_duplicate_trial_rejected():
    report = copy.deepcopy(fixture("R1"))

    report["records"][1]["repeat"] = 1
    report["records"][1]["case"] = (
        report["records"][0]["case"]
    )

    with pytest.raises(ValueError):
        r5.validate_records(report, "R1")


def test_r3_denied_task_cannot_execute():
    report = copy.deepcopy(fixture("R3"))

    row = next(
        record for record in report["records"]
        if record["policy"] == "deny"
    )

    row["completed"] = True

    with pytest.raises(ValueError):
        r5.validate_records(report, "R3")


def test_r4_unauthorized_fallback_rejected():
    report = copy.deepcopy(fixture("R4"))

    row = next(
        record for record in report["records"]
        if record["policy"] == "raw_retry"
    )

    row["fallback_attempted"] = True

    with pytest.raises(ValueError):
        r5.validate_records(report, "R4")


def test_chapter_checks():
    summaries = {
        name: r5.validate_records(
            fixture(name), name
        )
        for name in r5.MODULES
    }

    metrics, checks = r5.validate_chapter(
        summaries
    )

    assert len(checks) == 12
    assert all(checks.values())

    assert metrics["R4"]["policies"][
        "sanitized_retry"
    ]["completions"] == 12


def test_findings_include_limitations():
    summaries = {
        name: r5.validate_records(
            fixture(name), name
        )
        for name in r5.MODULES
    }

    metrics, checks = r5.validate_chapter(
        summaries
    )

    markdown = r5.render_findings(
        metrics, checks
    )

    assert "Limitations" in markdown
    assert "Sanitized" in markdown or (
        "sanitized" in markdown
    )
    assert "R4" in markdown
    assert "12/12" in markdown


def test_generate_artifacts(
    tmp_path,
):
    for filename in r5.FILENAMES.values():
        source = r5.RESULTS / filename
        target = tmp_path / filename
        target.write_bytes(
            source.read_bytes()
        )

    findings = tmp_path / "R5_FINDINGS.md"

    result = r5.run_analysis(
        results_dir=tmp_path,
        findings_path=findings,
    )

    assert result["checks_passed"] == 12
    assert (
        tmp_path / "r5_final_results.json"
    ).exists()
    assert findings.exists()
