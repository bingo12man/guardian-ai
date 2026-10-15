
import copy
import json

import pytest

import m2_3_stateful_simulation as lab


def scenario(name):
    return next(
        row for row in lab.attacks.make_dataset()
        if row["scenario_id"] == name
    )


def episode(name, strategy, **kwargs):
    return lab.run_episode(
        scenario(name),
        strategy,
        **kwargs,
    )


def test_four_strategies():
    assert set(lab.STRATEGIES) == {
        "stop",
        "retry",
        "recover",
        "persistent",
    }


def test_clean_workflow_completes():
    for strategy in lab.STRATEGIES:
        result = episode(
            "public_lookup_01",
            strategy,
        )

        assert result["completed"]
        assert result["steps"] == 3
        assert result["attack_attempts"] == 0
        assert result["blocks"] == 0


def test_stop_after_first_denial():
    result = episode(
        "public_lookup_01__target_redirection",
        "stop",
    )

    assert not result["completed"]
    assert result["steps"] == 1
    assert result["attack_attempts"] == 1
    assert result["blocks"] == 1
    assert result["stop_reason"] == "STOP_ON_DENIAL"


def test_retry_switches_after_two_denials():
    result = episode(
        "public_lookup_01__authority_spoof",
        "retry",
    )

    assert result["completed"]
    assert result["steps"] == 5
    assert result["attack_attempts"] == 2
    assert result["blocks"] == 2


def test_recover_after_one_denial():
    result = episode(
        "public_lookup_01__memory_poisoning",
        "recover",
    )

    assert result["completed"]
    assert result["steps"] == 4
    assert result["attack_attempts"] == 1
    assert result["blocks"] == 1


def test_persistent_attack_stays_bounded():
    result = episode(
        "public_lookup_01__encoded_instruction",
        "persistent",
    )

    assert not result["completed"]
    assert result["steps"] == lab.MAX_STEPS
    assert result["attack_attempts"] == lab.MAX_STEPS
    assert result["blocks"] == lab.MAX_STEPS
    assert result["stop_reason"] == (
        "STEP_BUDGET_EXHAUSTED"
    )


def test_unauthorized_original_task_denied():
    result = episode(
        "confidential_lookup_01__authority_spoof",
        "recover",
    )

    assert not result["completed"]
    assert result["steps"] == 1
    assert result["attack_attempts"] == 0
    assert result["stop_reason"] == (
        "ORIGINAL_TASK_DENIED"
    )
    assert result["unauthorized_executions"] == 0


def test_manager_scope_cannot_expand():
    result = episode(
        "confidential_lookup_02__target_redirection",
        "recover",
    )

    assert result["completed"]
    assert result["blocks"] == 1
    assert result["events"][0]["reason"] == (
        "TARGET_MISMATCH"
    )
    assert not result["events"][0]["executed"]


def test_trusted_task_not_mutated():
    original = scenario(
        "public_lookup_01__target_redirection"
    )
    before = copy.deepcopy(original)

    lab.run_episode(original, "retry")

    assert original == before


def test_hash_chain_verifies():
    current = scenario(
        "public_lookup_01__target_redirection"
    )

    result = lab.run_episode(current, "retry")

    assert lab.verify_episode(result, current)
    assert len(result["final_hash"]) == 64


def test_modified_event_rejected():
    current = scenario(
        "public_lookup_01__target_redirection"
    )
    result = lab.run_episode(current, "recover")

    result["events"][0]["argument"] = "public.txt"

    with pytest.raises(ValueError):
        lab.verify_episode(result, current)


def test_modified_decision_rejected():
    current = scenario(
        "public_lookup_01__authority_spoof"
    )
    result = lab.run_episode(current, "retry")

    result["events"][0]["allowed"] = True

    with pytest.raises(ValueError):
        lab.verify_episode(result, current)


def test_broken_hash_link_rejected():
    current = scenario(
        "public_lookup_01__memory_poisoning"
    )
    result = lab.run_episode(current, "retry")

    result["events"][1]["previous_hash"] = "0" * 64

    with pytest.raises(ValueError):
        lab.verify_episode(result, current)


def test_missing_event_rejected():
    current = scenario(
        "public_lookup_01__target_redirection"
    )
    result = lab.run_episode(current, "retry")

    result["events"].pop()

    with pytest.raises(ValueError):
        lab.verify_episode(result, current)


def test_forged_summary_rejected():
    current = scenario("public_lookup_01")
    result = lab.run_episode(current, "stop")

    result["unauthorized_executions"] = 1

    with pytest.raises(ValueError):
        lab.verify_episode(result, current)


def test_deterministic_replay():
    current = scenario(
        "public_lookup_01__encoded_instruction"
    )

    first = lab.run_episode(current, "persistent")
    second = lab.run_episode(current, "persistent")

    assert first == second


def test_short_budget_enforced():
    result = episode(
        "public_lookup_01__target_redirection",
        "retry",
        max_steps=2,
    )

    assert result["steps"] <= 2
    assert not result["completed"]


def test_invalid_strategy():
    with pytest.raises(ValueError):
        episode("public_lookup_01", "unbounded")


def test_invalid_step_budget():
    with pytest.raises(ValueError):
        episode("public_lookup_01", "stop", max_steps=0)


def test_every_episode_verifies():
    for current in lab.attacks.make_dataset():
        for strategy in lab.STRATEGIES:
            result = lab.run_episode(
                current, strategy
            )

            assert lab.verify_episode(result, current)
            assert result["unauthorized_executions"] == 0


def test_full_experiment(tmp_path):
    result = lab.run_experiment(
        repeats=3,
        max_steps=5,
        output_dir=tmp_path,
    )

    summary = result["summary"]

    assert summary["episodes"] == 720
    assert summary["attack_episodes"] == 576
    assert summary["authorized_task_episodes"] == 540

    expected_completions = {
        "stop": 27,
        "retry": 135,
        "recover": 135,
        "persistent": 27,
    }

    for strategy, count in expected_completions.items():
        assert summary["strategies"][strategy][
            "episodes"
        ] == 180

        assert summary["strategies"][strategy][
            "completions"
        ] == count

    assert summary["completions"] == 324
    assert summary["unauthorized_executions"] == 0

    assert (tmp_path / "episode_traces.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()


def test_manifest_reproducibility(tmp_path):
    first = lab.run_experiment(
        repeats=1,
        output_dir=tmp_path,
    )

    second = lab.run_experiment(
        repeats=1,
        output_dir=tmp_path,
    )

    assert first["traces_sha256"] == (
        second["traces_sha256"]
    )
