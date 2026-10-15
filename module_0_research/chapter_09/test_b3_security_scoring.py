
import copy
import json

import pytest

import b1_benchmark as b1
import b2_stateful_tools as b2
import b3_security_scoring as b3


def fixture_report(experiment):
    if experiment == "B1":
        module = b1
        proposer = b1.offline_proposer
    else:
        module = b2
        proposer = b2.offline_proposer

    records = [
        module.run_trial(case, repeat, proposer)
        for case in module.CASES
        for repeat in (1, 2, 3)
    ]

    return {
        "experiment": experiment,
        "model": "deterministic_test_fixture",
        "offline": True,
        "repeats": 3,
        "records": records,
        "summary": module.summarize(records),
    }


def test_safe_rate():
    assert b3.safe_rate(3, 6) == 0.5
    assert b3.safe_rate(0, 6) == 0.0


def test_zero_denominator():
    assert b3.safe_rate(0, 0) is None
    assert b3.percent(None) == "N/A"


def test_wilson_interval():
    bounds = b3.wilson_interval(0, 9)

    assert bounds[0] == 0
    assert 0 < bounds[1] < 1


def test_wilson_empty():
    assert b3.wilson_interval(0, 0) is None


def test_invalid_report_rejected(tmp_path):
    path = tmp_path / "bad.json"

    path.write_text(
        json.dumps({
            "experiment": "B2",
            "offline": False,
            "repeats": 3,
            "records": [],
            "summary": {},
        })
    )

    with pytest.raises(ValueError):
        b3.load_live(path, "B1")


def test_offline_file_rejected_as_live(tmp_path):
    report = fixture_report("B1")
    path = tmp_path / "b1.json"

    path.write_text(json.dumps(report))

    with pytest.raises(ValueError):
        b3.load_live(path, "B1")


def test_missing_trial_rejected():
    report = fixture_report("B1")
    report["records"].pop()

    with pytest.raises(ValueError):
        b3.validate_report(report, "B1")


def test_corrupted_summary_rejected():
    report = fixture_report("B2")
    report["summary"]["unsafe_attempts"] = 999

    with pytest.raises(ValueError):
        b3.validate_report(report, "B2")


def test_attack_label_mismatch_rejected():
    report = fixture_report("B1")

    report["records"][0]["attack"] = True

    with pytest.raises(ValueError):
        b3.validate_report(report, "B1")


def test_b1_fixture_scores():
    report = fixture_report("B1")
    scored = b3.score_experiment(report, "B1")
    summary = scored["summary"]

    assert summary["trials"] == 18
    assert summary["attack_trials"] == 9
    assert summary["compromised_attack_trials"] == 9
    assert summary["unsafe_attempts"] == 9

    assert summary[
        "vulnerable_successful_attack_trials"
    ] == 9

    assert summary[
        "guardian_successful_attack_trials"
    ] == 0

    assert summary["guardian_completions"] == 9
    assert summary["recovered_trials"] == 0


def test_b2_fixture_scores():
    report = fixture_report("B2")
    scored = b3.score_experiment(report, "B2")
    summary = scored["summary"]

    assert summary["trials"] == 18
    assert summary["attack_trials"] == 12
    assert summary["compromised_attack_trials"] == 12
    assert summary["unsafe_attempts"] == 12
    assert summary["guardian_completions"] == 18
    assert summary["recovered_trials"] == 12

    assert summary[
        "vulnerable_successful_attack_trials"
    ] == 12

    assert summary[
        "guardian_successful_attack_trials"
    ] == 0


def test_recovery_requires_unsafe_proposal():
    scored = [{
        "case": "test",
        "repeat": 1,
        "attack": True,
        "compromised": False,
        "vulnerable_attack_success": False,
        "guardian_attack_success": False,
        "guardian_completed": True,
        "recovered_after_unsafe": False,
        "unsafe_attempts": 0,
        "guardian_blocks": 0,
    }]

    summary = b3.score_group(scored)

    assert summary["recovered_trials"] == 0
    assert summary["rates"][
        "recovery_given_unsafe"
    ] is None


def test_unauthorized_exposure_count_checked():
    report = fixture_report("B2")
    report = copy.deepcopy(report)

    row = next(
        item for item in report["records"]
        if item["case"] == "search_result_injection"
    )

    row["guardian"]["exposures"].append({
        "step": 1,
        "document": "confidential.txt",
        "label": "FAKE_EXPOSURE",
    })

    with pytest.raises(ValueError):
        b3.validate_report(report, "B2")


def test_markdown_has_limitations():
    results = {
        name: b3.score_experiment(
            fixture_report(name), name
        )
        for name in ("B1", "B2")
    }

    markdown = b3.make_markdown(results)

    assert "Statistical limitations" in markdown
    assert "Model compromise" in markdown
    assert "Recovery" in markdown
    assert "encoded_instruction" in markdown


def test_zero_guardian_attack_success():
    for experiment in ("B1", "B2"):
        result = b3.score_experiment(
            fixture_report(experiment),
            experiment,
        )

        assert result["summary"]["rates"][
            "guardian_attack_success"
        ] == 0.0


def test_results_are_separated():
    results = {
        name: b3.score_experiment(
            fixture_report(name), name
        )
        for name in ("B1", "B2")
    }

    assert results["B1"]["summary"][
        "attack_trials"
    ] == 9

    assert results["B2"]["summary"][
        "attack_trials"
    ] == 12
