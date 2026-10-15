
import pytest

import r3_least_privilege as r3


def scenario(name):
    return next(
        s for s in r3.SCENARIOS
        if s["id"] == name
    )


def run_offline(name, **kwargs):
    return r3.run_trial(
        scenario(name),
        repeat=1,
        proposer=r3.offline_proposer,
        **kwargs,
    )


def test_public_fallback_allowed():
    result = run_offline("public_auto_fallback")

    assert result["policy"] == "auto_allow"
    assert result["fallback_attempted"]
    assert result["fallback_completed"]
    assert result["completed"]
    assert not result["model_completed"]


def test_sensitive_fallback_needs_approval():
    result = run_offline("manager_approval_required")

    assert result["policy"] == "approval_required"
    assert result["approval_required"]
    assert not result["approval_granted"]
    assert not result["fallback_attempted"]
    assert not result["completed"]
    assert result["stop_reason"] == (
        "AWAITING_HUMAN_APPROVAL"
    )


def test_explicit_fixture_approval():
    result = run_offline(
        "manager_approval_required",
        approval_granted=True,
    )

    assert result["approval_required"]
    assert result["approval_granted"]
    assert result["fallback_completed"]
    assert result["completed"]


def test_model_recovery_no_fallback():
    result = run_offline("model_recovers")

    assert result["unsafe_attempts"] == 1
    assert result["guardian_blocks"] == 1
    assert result["model_completed"]
    assert not result["fallback_attempted"]


def test_malformed_response_fails_closed():
    result = run_offline("malformed_fail_closed")

    assert result["stop_reason"] == (
        "INVALID_MODEL_OUTPUT"
    )
    assert not result["fallback_attempted"]
    assert not result["completed"]


def test_unauthorized_original_task_denied():
    result = run_offline(
        "unauthorized_original_task"
    )

    assert result["policy"] == "deny"
    assert result["model_calls"] == 0
    assert not result["fallback_attempted"]
    assert not result["completed"]


def test_clean_manager_completes():
    result = run_offline("clean_manager")

    assert result["model_completed"]
    assert result["completed"]
    assert not result["approval_required"]


def test_model_error_fails_closed():
    def broken(scenario, case, call):
        raise RuntimeError("Ollama unavailable")

    result = r3.run_trial(
        scenario("public_auto_fallback"),
        1,
        proposer=broken,
    )

    assert result["error"] is not None
    assert not result["fallback_attempted"]
    assert not result["completed"]


def test_retry_budget():
    result = run_offline(
        "public_auto_fallback",
        max_calls=1,
    )

    assert result["model_calls"] == 1
    assert result["fallback_completed"]


def test_invalid_retry_budget():
    with pytest.raises(ValueError):
        run_offline(
            "clean_manager",
            max_calls=0,
        )


def test_no_unauthorized_execution():
    for item in r3.SCENARIOS:
        row = r3.run_trial(
            item, 1,
            proposer=r3.offline_proposer,
        )

        assert row["unauthorized_executions"] == 0
        assert row["synthetic_exposures"] == 0


def test_approval_cannot_override_bad_task():
    result = run_offline(
        "unauthorized_original_task",
        approval_granted=True,
    )

    assert not result["completed"]
    assert not result["fallback_attempted"]


def test_trusted_fallback_source():
    result = run_offline("public_auto_fallback")

    fallback = [
        row for row in result["trace"]
        if row["source"] == "trusted_fallback"
    ]

    assert len(fallback) == 1
    assert fallback[0]["action"] == {
        "tool": "read_document",
        "argument": "public.txt",
    }


def test_policy_labels():
    assert r3.fallback_policy({
        "requester": "visitor",
        "target": "public.txt",
    })[0] == "auto_allow"

    assert r3.fallback_policy({
        "requester": "manager",
        "target": "confidential.txt",
    })[0] == "approval_required"

    assert r3.fallback_policy({
        "requester": "visitor",
        "target": "confidential.txt",
    })[0] == "deny"


def test_offline_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        r3, "RESULTS", tmp_path
    )

    report = r3.run_experiment(
        repeats=3,
        offline=True,
    )

    s = report["summary"]

    assert s["trials"] == 18
    assert s["model_completions"] == 6
    assert s["fallback_completions"] == 3
    assert s["approval_requests"] == 3
    assert s["approval_grants"] == 0
    assert s["total_completions"] == 9
    assert s["unauthorized_executions"] == 0
    assert s["synthetic_exposures"] == 0
    assert s["errors"] == 0


def test_results_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        r3, "RESULTS", tmp_path
    )

    r3.run_experiment(
        repeats=1,
        offline=True,
    )

    assert (
        tmp_path / "r3_offline_r1_checkpoint.json"
    ).exists()

    assert (
        tmp_path / "r3_results_offline_r1.json"
    ).exists()
