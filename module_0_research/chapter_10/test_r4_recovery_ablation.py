
import pytest

import r4_recovery_ablation as r4


def scenario(name):
    return next(
        s for s in r4.SCENARIOS
        if s["id"] == name
    )


def trial(name, policy):
    return r4.run_trial(
        scenario(name),
        policy,
        1,
        proposer=r4.offline_proposer,
    )


@pytest.mark.parametrize("policy", r4.POLICIES)
def test_clean_task_completes(policy):
    result = trial("clean_public", policy)

    assert result["model_completed"]
    assert result["completed"]
    assert result["model_calls"] == 1


def test_stop_on_attack():
    result = trial(
        "recoverable_injection", "stop"
    )

    assert not result["completed"]
    assert result["guardian_blocks"] == 1


def test_raw_retry_stays_compromised():
    result = trial(
        "recoverable_injection", "raw_retry"
    )

    assert result["model_calls"] == 2
    assert result["unsafe_attempts"] == 2
    assert not result["completed"]


def test_sanitized_retry_recovers():
    result = trial(
        "recoverable_injection",
        "sanitized_retry",
    )

    assert result["model_calls"] == 2
    assert result["model_completed"]
    assert result["completed"]
    assert not result["fallback_attempted"]


def test_sanitized_fallback_recovers_by_model():
    result = trial(
        "recoverable_injection",
        "sanitized_fallback",
    )

    assert result["model_completed"]
    assert not result["fallback_attempted"]


def test_persistent_attack_needs_fallback():
    result = trial(
        "persistent_injection",
        "sanitized_fallback",
    )

    assert result["guardian_blocks"] == 2
    assert result["fallback_attempted"]
    assert result["fallback_completed"]
    assert not result["model_completed"]
    assert result["completed"]


def test_persistent_attack_without_fallback():
    result = trial(
        "persistent_injection",
        "sanitized_retry",
    )

    assert not result["completed"]
    assert not result["fallback_attempted"]


def test_sensitive_fallback_requires_approval():
    result = trial(
        "sensitive_fallback",
        "sanitized_fallback",
    )

    assert result["approval_required"]
    assert not result["fallback_attempted"]
    assert not result["completed"]


@pytest.mark.parametrize("policy", r4.POLICIES)
def test_unauthorized_original_task_denied(policy):
    result = trial(
        "invalid_original_task", policy
    )

    assert result["model_calls"] == 0
    assert not result["completed"]
    assert result["stop_reason"] == (
        "INVALID_ORIGINAL_TASK"
    )


@pytest.mark.parametrize("policy", r4.POLICIES)
def test_no_unauthorized_execution(policy):
    for item in r4.SCENARIOS:
        result = r4.run_trial(
            item,
            policy,
            1,
            proposer=r4.offline_proposer,
        )

        assert result["unauthorized_executions"] == 0
        assert result["synthetic_exposures"] == 0


def test_model_error_fails_closed():
    def broken(scenario, case, attempt, sanitized):
        raise RuntimeError("Model disconnected")

    result = r4.run_trial(
        scenario("persistent_injection"),
        "sanitized_fallback",
        1,
        proposer=broken,
    )

    assert result["error"] is not None
    assert not result["fallback_attempted"]
    assert not result["completed"]


def test_invalid_output_no_fallback():
    def invalid(scenario, case, attempt, sanitized):
        return {
            "tool": "read_document",
            "argument": 123,
        }

    result = r4.run_trial(
        scenario("persistent_injection"),
        "sanitized_fallback",
        1,
        proposer=invalid,
    )

    assert result["stop_reason"] == (
        "INVALID_MODEL_OUTPUT"
    )
    assert not result["fallback_attempted"]


def test_unknown_policy():
    with pytest.raises(ValueError):
        trial("clean_public", "unknown")


def test_expected_offline_ablation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        r4, "RESULTS", tmp_path
    )

    report = r4.run_experiment(
        repeats=3,
        offline=True,
    )

    summary = report["summary"]

    assert summary["trials"] == 60

    expected = {
        "stop": 3,
        "raw_retry": 3,
        "sanitized_retry": 6,
        "sanitized_fallback": 9,
    }

    for policy, completions in expected.items():
        assert (
            summary["policies"][policy][
                "total_completions"
            ] == completions
        )

    assert summary["policies"][
        "sanitized_fallback"
    ]["fallback_completions"] == 3

    assert summary["policies"][
        "sanitized_fallback"
    ]["approval_requests"] == 3

    assert summary["unauthorized_executions"] == 0
    assert summary["synthetic_exposures"] == 0


def test_checkpoint_written(tmp_path, monkeypatch):
    monkeypatch.setattr(
        r4, "RESULTS", tmp_path
    )

    r4.run_experiment(
        repeats=1,
        offline=True,
    )

    assert (
        tmp_path /
        "r4_offline_r1_checkpoint.json"
    ).exists()
