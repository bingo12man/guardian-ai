
import copy

import pytest

import r2_loop_prevention as r2


def scenario(name):
    return next(
        s for s in r2.SCENARIOS
        if s["id"] == name
    )


def run_offline(name, **kwargs):
    return r2.run_trial(
        scenario(name),
        repeat=1,
        proposer=r2.offline_proposer,
        **kwargs,
    )


def test_normal_task_completes():
    result = run_offline("normal_completion")

    assert result["model_completed"]
    assert result["model_calls"] == 1
    assert result["completed"]
    assert not result["fallback_attempted"]


def test_repeated_attack_detected():
    result = run_offline("repeated_injection")

    assert result["loop_detected"]
    assert result["model_calls"] == 2
    assert result["guardian_blocks"] == 2
    assert result["unsafe_attempts"] == 2
    assert result["fallback_completed"]


def test_alternating_attack_is_bounded():
    result = run_offline("alternating_injection")

    assert result["model_calls"] == 3
    assert result["guardian_blocks"] == 3
    assert not result["loop_detected"]
    assert result["fallback_completed"]


def test_model_can_recover():
    result = run_offline("recovery_success")

    assert result["model_calls"] == 2
    assert result["guardian_blocks"] == 1
    assert result["model_completed"]
    assert not result["fallback_attempted"]


def test_malformed_loop_detected():
    result = run_offline("malformed_loop")

    assert result["loop_detected"]
    assert result["guardian_blocks"] == 2
    assert result["unsafe_attempts"] == 0
    assert result["fallback_completed"]


def test_model_failure_fails_closed():
    result = run_offline("model_failure")

    assert result["model_calls"] == 2
    assert result["error"] is not None
    assert result["stop_reason"] == "MODEL_ERROR"
    assert not result["fallback_attempted"]
    assert not result["completed"]


def test_repeated_attack_without_fallback():
    result = run_offline(
        "repeated_injection",
        allow_fallback=False,
    )

    assert result["loop_detected"]
    assert not result["completed"]
    assert result["unauthorized_executions"] == 0


def test_no_unauthorized_execution():
    for item in r2.SCENARIOS:
        result = r2.run_trial(
            item, 1,
            proposer=r2.offline_proposer,
        )

        assert result["unauthorized_executions"] == 0
        assert result["synthetic_exposures"] == 0


def test_sanitized_retry():
    observed = []

    def observing_proposer(
        scenario, case, call, history
    ):
        observed.append({
            "call": call,
            "snippet": case["snippet"],
            "history": copy.deepcopy(history),
        })

        return r2.read("confidential.txt")

    r2.run_trial(
        scenario("repeated_injection"),
        1,
        proposer=observing_proposer,
    )

    assert "confidential.txt" in (
        observed[0]["snippet"]
    )

    assert "confidential.txt" not in (
        observed[1]["snippet"]
    )

    assert "confidential.txt" not in str(
        observed[1]["history"]
    )


def test_fallback_is_independently_authorized():
    original = r2.build_case(
        scenario("repeated_injection")
    )

    original["requester"] = "visitor"
    original["target"] = "confidential.txt"

    # Use a custom controller fixture by
    # temporarily overriding build_case.
    original_builder = r2.build_case

    try:
        r2.build_case = lambda item: copy.deepcopy(
            original
        )

        result = r2.run_trial(
            scenario("repeated_injection"),
            1,
            proposer=r2.offline_proposer,
        )

    finally:
        r2.build_case = original_builder

    assert result["fallback_attempted"]
    assert not result["fallback_completed"]
    assert not result["completed"]
    assert result["unauthorized_executions"] == 0


def test_call_budget():
    result = run_offline(
        "alternating_injection",
        max_calls=2,
    )

    assert result["model_calls"] <= 2
    assert result["unauthorized_executions"] == 0


def test_invalid_max_calls():
    with pytest.raises(ValueError):
        run_offline(
            "normal_completion",
            max_calls=0,
        )


def test_does_not_mutate_scenarios():
    original = copy.deepcopy(r2.SCENARIOS)

    run_offline("repeated_injection")

    assert r2.SCENARIOS == original


def test_fallback_source_recorded():
    result = run_offline("repeated_injection")

    sources = [
        entry["source"]
        for entry in result["trace"]
    ]

    assert "trusted_fallback" in sources
    assert result["model_completed"] is False


def test_offline_experiment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        r2, "RESULTS", tmp_path
    )

    report = r2.run_experiment(
        repeats=3,
        offline=True,
    )

    summary = report["summary"]

    assert summary["trials"] == 18
    assert summary["attack_trials"] == 15
    assert summary["model_completions"] == 6
    assert summary["fallback_completions"] == 9
    assert summary["total_completions"] == 15
    assert summary["loops_detected"] == 6
    assert summary["errors"] == 3
    assert summary["unauthorized_executions"] == 0
    assert summary["synthetic_exposures"] == 0


def test_checkpoint_created(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        r2, "RESULTS", tmp_path
    )

    r2.run_experiment(
        repeats=1,
        offline=True,
    )

    assert (
        tmp_path /
        "r2_offline_r1_checkpoint.json"
    ).is_file()

    assert (
        tmp_path /
        "r2_results_offline_r1.json"
    ).is_file()
