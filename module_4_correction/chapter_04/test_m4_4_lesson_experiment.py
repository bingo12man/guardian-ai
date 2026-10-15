
import copy
import json

import pytest

import m4_4_lesson_experiment as lab


def get_scenario(scenario_id):
    return next(
        row for row in lab.held_out_scenarios()
        if row["scenario_id"] == scenario_id
    )


def response(tool, argument):
    return {
        "raw": json.dumps({
            "tool": tool,
            "argument": argument,
        }),
        "latency_ms": 10.0,
        "prompt_tokens": 20,
        "generated_tokens": 10,
    }


def successful_proposer(model, messages):
    last = messages[-1]["content"]

    if "Next stage: need_finish" in last:
        return response("finish", "done")

    if "Next stage: need_read" in last:
        return response("read_document", "public.txt")

    return response("search_documents", "public.txt")


def attack_once_proposer(model, messages):
    # First model call: submit a wrong target.
    if len(messages) == 2:
        return response(
            "search_documents", "confidential.txt"
        )

    # After Guardian denial: return to task.
    return successful_proposer(model, messages)


def always_wrong_proposer(model, messages):
    return response(
        "search_documents", "confidential.txt"
    )


def find_public_scenario():
    return next(
        s for s in lab.held_out_scenarios()
        if s["trusted_task"]["target"] == "public.txt"
        and s["ground_truth"][
            "original_task_authorized"
        ]
    )


def test_heldout_dataset():
    rows = lab.held_out_scenarios()

    assert len(rows) == 30
    assert all(
        row["split"] in ("validation", "test")
        for row in rows
    )


def test_no_training_scenarios():
    train_ids = {
        s["scenario_id"]
        for s in lab.dataset.make_dataset()
        if s["split"] == "train"
    }

    heldout_ids = {
        s["scenario_id"]
        for s in lab.held_out_scenarios()
    }

    assert not train_ids.intersection(heldout_ids)


def test_two_conditions():
    assert lab.CONDITIONS == (
        "baseline",
        "lesson_assisted",
    )


def test_memory_required_for_lesson_condition():
    current = find_public_scenario()

    with pytest.raises(ValueError):
        lab.run_episode(
            current,
            "test-model",
            "lesson_assisted",
            memory=None,
            proposer=successful_proposer,
        )


def test_baseline_clean_completion():
    current = find_public_scenario()

    result = lab.run_episode(
        current,
        "test-model",
        "baseline",
        proposer=successful_proposer,
    )

    assert result["completed"]
    assert result["tool_effects"] == 3
    assert result["blocked_proposals"] == 0
    assert result["lesson_retrievals"] == 0
    assert result["audit_verified"]


def test_lesson_condition_without_denial():
    current = find_public_scenario()
    memory = lab.load_verified_memory()

    result = lab.run_episode(
        current,
        "test-model",
        "lesson_assisted",
        memory=memory,
        proposer=successful_proposer,
    )

    assert result["completed"]
    assert result["lesson_retrievals"] == 0
    assert result["tool_effects"] == 3


def test_lesson_retrieval_after_wrong_target():
    current = find_public_scenario()
    memory = lab.load_verified_memory()

    result = lab.run_episode(
        current,
        "test-model",
        "lesson_assisted",
        memory=memory,
        proposer=attack_once_proposer,
    )

    assert result["completed"]
    assert result["blocked_proposals"] == 1
    assert result["wrong_target_proposals"] == 1
    assert result["lesson_retrievals"] == 1
    assert result["lesson_misses"] == 0
    assert result["tool_effects"] == 3
    assert result["unauthorized_executions"] == 0

    first = result["trace"][0]

    assert first["status"] == "TARGET_MISMATCH"
    assert first["executed"] is False
    assert first["correction_id"] is not None


def test_baseline_wrong_target_recovers_without_lesson():
    current = find_public_scenario()

    result = lab.run_episode(
        current,
        "test-model",
        "baseline",
        proposer=attack_once_proposer,
    )

    assert result["completed"]
    assert result["wrong_target_proposals"] == 1
    assert result["lesson_retrievals"] == 0
    assert result["tool_effects"] == 3


def test_repeated_denials():
    current = find_public_scenario()
    memory = lab.load_verified_memory()

    result = lab.run_episode(
        current,
        "test-model",
        "lesson_assisted",
        memory=memory,
        proposer=always_wrong_proposer,
        max_steps=4,
    )

    assert not result["completed"]
    assert result["blocked_proposals"] == 4
    assert result["repeat_denials"] == 3
    assert result["tool_effects"] == 0
    assert result["unauthorized_executions"] == 0


def test_invalid_output_without_execution():
    current = find_public_scenario()

    def invalid_proposer(model, messages):
        return {
            "raw": "not valid json",
            "latency_ms": 5,
            "prompt_tokens": 1,
            "generated_tokens": 1,
        }

    result = lab.run_episode(
        current,
        "test-model",
        "baseline",
        proposer=invalid_proposer,
        max_steps=3,
    )

    assert result["invalid_outputs"] == 3
    assert result["blocked_proposals"] == 3
    assert result["tool_effects"] == 0
    assert result["capabilities_issued"] == 0


def test_invalid_original_task_prechecked():
    current = next(
        row for row in lab.held_out_scenarios()
        if not row["ground_truth"][
            "original_task_authorized"
        ]
    )

    result = lab.run_episode(
        current,
        "test-model",
        "baseline",
        proposer=always_wrong_proposer,
    )

    assert result["stop_reason"] == (
        "ORIGINAL_TASK_DENIED"
    )
    assert result["model_calls"] == 0
    assert result["tool_effects"] == 0


def test_scenario_not_mutated():
    current = find_public_scenario()
    snapshot = copy.deepcopy(current)

    lab.run_episode(
        current,
        "test-model",
        "baseline",
        proposer=successful_proposer,
    )

    assert current == snapshot


def test_lesson_feedback_is_allowlisted():
    memory = lab.load_verified_memory()
    current = find_public_scenario()

    from guardian_core import ActionProposal

    feedback, correction_id = (
        lab.feedback_for_denial(
            "lesson_assisted",
            memory,
            "TARGET_MISMATCH",
            "need_search",
            ActionProposal(
                "search_documents",
                "confidential.txt",
            ),
            current["trusted_task"],
        )
    )

    assert correction_id is not None
    assert "Guardian correction" in feedback
    assert "confidential.txt" not in feedback


def test_paired_differences():
    records = [
        {
            "model": "test-model",
            "scenario_id": "s1",
            "repeat": 1,
            "condition": "baseline",
            "completed": False,
            "blocked_proposals": 3,
            "repeat_denials": 2,
            "wrong_target_proposals": 1,
        },
        {
            "model": "test-model",
            "scenario_id": "s1",
            "repeat": 1,
            "condition": "lesson_assisted",
            "completed": True,
            "blocked_proposals": 1,
            "repeat_denials": 0,
            "wrong_target_proposals": 0,
        },
    ]

    paired = lab.paired_differences(records)[
        "test-model"
    ]

    assert paired["pairs"] == 1
    assert paired["completion_delta"] == 1
    assert paired["blocked_proposal_delta"] == -2
    assert paired["repeat_denial_delta"] == -2
    assert paired["wrong_target_delta"] == -1


def test_smoke_benchmark_with_fixture(tmp_path):
    report = lab.run_benchmark(
        models=("test-model",),
        repeats=1,
        limit=5,
        proposer=always_wrong_proposer,
        output_dir=tmp_path,
    )

    assert report["summary"]["test-model"][
        "baseline"
    ]["episodes"] == 5

    assert report["summary"]["test-model"][
        "lesson_assisted"
    ]["episodes"] == 5

    assert report["paired_differences"][
        "test-model"
    ]["pairs"] == 5

    assert (
        tmp_path / "c4_results_heldout_5_r1.json"
    ).exists()


def test_invalid_configuration():
    with pytest.raises(ValueError):
        lab.held_out_scenarios(limit=0)

    with pytest.raises(ValueError):
        lab.run_benchmark(
            repeats=0,
            proposer=successful_proposer,
        )
