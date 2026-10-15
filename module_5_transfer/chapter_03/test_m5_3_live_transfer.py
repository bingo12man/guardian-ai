
import copy
import json

import pytest

import m5_3_live_transfer as lab


def action(tool, argument):
    return {
        "raw": json.dumps({
            "tool": tool,
            "argument": argument,
        }),
        "latency_ms": 4.0,
        "prompt_tokens": 10,
        "generated_tokens": 5,
    }


def public_scenario():
    return next(
        row for row in lab.scenarios_for_evaluation()
        if row["trusted_task"]["target"] == "public.txt"
        and row["ground_truth"][
            "original_task_authorized"
        ]
    )


def successful_proposer(model, messages):
    last = messages[-1]["content"]

    if "Next stage: need_finish" in last:
        return action("finish", "done")

    if "Next stage: need_read" in last:
        return action("read_document", "public.txt")

    return action("search_documents", "public.txt")


def wrong_then_recover(model, messages):
    if len(messages) == 2:
        return action(
            "search_documents",
            "confidential.txt",
        )

    return successful_proposer(model, messages)


def always_wrong(model, messages):
    return action(
        "search_documents",
        "confidential.txt",
    )


def invalid_then_recover(model, messages):
    if len(messages) == 2:
        return {
            "raw": "not valid JSON",
            "latency_ms": 4.0,
            "prompt_tokens": 10,
            "generated_tokens": 5,
        }

    return successful_proposer(model, messages)


def test_heldout_count():
    rows = lab.scenarios_for_evaluation()
    assert len(rows) == 30
    assert all(
        row["split"] in ("validation", "test")
        for row in rows
    )


def test_condition_definitions():
    assert lab.CONDITIONS == (
        "baseline",
        "same_model",
        "cross_model",
    )


def test_donor_mapping():
    assert lab.donor_for(
        "llama3.2:3b", "baseline"
    ) is None

    assert lab.donor_for(
        "llama3.2:3b", "same_model"
    ) == "llama3.2:3b"

    assert lab.donor_for(
        "llama3.2:3b", "cross_model"
    ) == "qwen2.5:3b"

    assert lab.donor_for(
        "qwen2.5:3b", "cross_model"
    ) == "llama3.2:3b"


def test_clean_baseline_completion():
    result = lab.run_episode(
        public_scenario(),
        "llama3.2:3b",
        "baseline",
        proposer=successful_proposer,
    )

    assert result["completed"]
    assert result["tool_effects"] == 3
    assert result["blocked_proposals"] == 0
    assert result["lesson_hits"] == 0
    assert result["audit_verified"]


def test_clean_transfer_completion():
    memory = lab.load_verified_memory()

    for condition in (
        "same_model", "cross_model"
    ):
        result = lab.run_episode(
            public_scenario(),
            "llama3.2:3b",
            condition,
            memory=memory,
            proposer=successful_proposer,
        )

        assert result["completed"]
        assert result["tool_effects"] == 3
        assert result["lesson_hits"] == 0
        assert result["lesson_misses"] == 0


def test_wrong_target_blocked():
    memory = lab.load_verified_memory()

    result = lab.run_episode(
        public_scenario(),
        "qwen2.5:3b",
        "cross_model",
        memory=memory,
        proposer=wrong_then_recover,
    )

    assert result["completed"]
    assert result["wrong_target_proposals"] == 1
    assert result["blocked_proposals"] == 1
    assert result["tool_effects"] == 3
    assert result["unauthorized_executions"] == 0


def test_wrong_target_never_causes_effect():
    memory = lab.load_verified_memory()

    result = lab.run_episode(
        public_scenario(),
        "qwen2.5:3b",
        "cross_model",
        memory=memory,
        proposer=always_wrong,
        max_steps=3,
    )

    assert not result["completed"]
    assert result["tool_effects"] == 0
    assert result["blocked_proposals"] == 3
    assert result["repeat_denials"] == 2
    assert result["audit_verified"]


def test_llama_donor_schema_transfer():
    memory = lab.load_verified_memory()
    scenario = public_scenario()

    result = lab.run_episode(
        scenario,
        "qwen2.5:3b",
        "cross_model",
        memory=memory,
        proposer=invalid_then_recover,
    )

    assert result["completed"]
    assert result["invalid_outputs"] == 1
    assert result["lesson_hits"] == 1
    assert result["lesson_misses"] == 0


def test_missing_qwen_schema_does_not_fallback():
    memory = lab.load_verified_memory()
    scenario = public_scenario()

    result = lab.run_episode(
        scenario,
        "qwen2.5:3b",
        "same_model",
        memory=memory,
        proposer=invalid_then_recover,
    )

    assert result["completed"]
    assert result["invalid_outputs"] == 1
    assert result["lesson_hits"] == 0
    assert result["lesson_misses"] == 1


def test_invalid_original_task_skips_model():
    scenario = next(
        row for row in lab.scenarios_for_evaluation()
        if not row["ground_truth"][
            "original_task_authorized"
        ]
    )

    result = lab.run_episode(
        scenario,
        "llama3.2:3b",
        "baseline",
        proposer=always_wrong,
    )

    assert result["model_calls"] == 0
    assert result["tool_effects"] == 0
    assert result["stop_reason"] == (
        "ORIGINAL_TASK_DENIED"
    )


def test_transfer_memory_required():
    with pytest.raises(ValueError):
        lab.run_episode(
            public_scenario(),
            "llama3.2:3b",
            "cross_model",
            memory=None,
            proposer=successful_proposer,
        )


def test_model_error_stops_safely():
    def failing(model, messages):
        raise TimeoutError("Injected model failure")

    result = lab.run_episode(
        public_scenario(),
        "llama3.2:3b",
        "baseline",
        proposer=failing,
    )

    assert result["stop_reason"] == "MODEL_CALL_ERROR"
    assert result["tool_effects"] == 0
    assert not result["completed"]


def test_scenario_not_mutated():
    scenario = public_scenario()
    original = copy.deepcopy(scenario)

    lab.run_episode(
        scenario,
        "llama3.2:3b",
        "baseline",
        proposer=successful_proposer,
    )

    assert scenario == original


def test_donor_feedback_contains_no_document_names():
    memory = lab.load_verified_memory()
    scenario = public_scenario()

    from guardian_core import ActionProposal

    feedback, correction_id, transfer_id = (
        lab.feedback_for_denial(
            memory,
            "cross_model",
            "qwen2.5:3b",
            "TARGET_MISMATCH",
            "need_search",
            ActionProposal(
                "search_documents",
                "confidential.txt",
            ),
            scenario["trusted_task"],
        )
    )

    assert feedback
    assert "confidential.txt" not in feedback

    if transfer_id is not None:
        assert correction_id is not None


def test_paired_comparison_math():
    records = []

    for recipient in lab.MODELS:
        for condition, completed, blocked in (
            ("baseline", False, 3),
            ("same_model", True, 1),
            ("cross_model", True, 2),
        ):
            records.append({
                "recipient_model": recipient,
                "scenario_id": "s1",
                "repeat": 1,
                "condition": condition,
                "original_task_authorized": True,
                "completed": completed,
                "blocked_proposals": blocked,
                "repeat_denials": blocked - 1,
                "wrong_target_proposals": 0,
            })

    paired = lab.paired_differences(records)

    for recipient in lab.MODELS:
        assert paired[recipient][
            "same_model"
        ]["completion_delta"] == 1

        assert paired[recipient][
            "cross_model"
        ]["blocked_delta"] == -1


def test_fixture_smoke_benchmark(tmp_path):
    report = lab.run_benchmark(
        models=lab.MODELS,
        repeats=1,
        limit=5,
        proposer=always_wrong,
        output_dir=tmp_path,
    )

    assert len(report["records"]) == 30

    for model in lab.MODELS:
        for condition in lab.CONDITIONS:
            assert report["summary"][model][
                condition
            ]["episodes"] == 5

    assert (
        tmp_path / "c3_results_heldout_5_r1.json"
    ).exists()


def test_invalid_limit():
    with pytest.raises(ValueError):
        lab.scenarios_for_evaluation(limit=0)


def test_invalid_condition():
    with pytest.raises(ValueError):
        lab.donor_for(
            "llama3.2:3b",
            "untrusted_mode",
        )
