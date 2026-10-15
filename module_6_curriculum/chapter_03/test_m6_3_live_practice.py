
import copy
import json
import re

import pytest

import m6_3_live_practice as lab


def data():
    return lab.load_verified_dataset()


def correct_proposer(model, messages):
    prompt = messages[-1]["content"]

    stage = re.search(
        r"Current Guardian stage: (\w+)",
        prompt,
    ).group(1)

    target = re.search(
        r"Trusted target: ([\w.]+)",
        prompt,
    ).group(1)

    action = lab.c1.expected_action(
        stage, target
    )

    return {
        "raw": json.dumps(action),
        "prompt_tokens": 20,
        "generated_tokens": 10,
        "latency_ms": 5.0,
    }


def always_invalid(model, messages):
    return {
        "raw": "not valid JSON",
        "prompt_tokens": 15,
        "generated_tokens": 8,
        "latency_ms": 5.0,
    }


def failing_proposer(model, messages):
    raise TimeoutError("Injected Ollama timeout")


def test_dataset_loads():
    source = data()

    assert len(source["exercises"]) == 21
    assert len(source["profiles"]) == 2


def test_base_messages():
    messages = lab.base_messages()

    assert len(messages) == 1
    assert messages[0]["role"] == "system"


def test_correct_model_response_is_scored():
    source = data()
    exercise = source["exercises"][0]

    messages = lab.base_messages()
    messages.append({
        "role": "user",
        "content": exercise["prompt"],
    })

    output = lab.call_model(
        exercise["model"],
        messages,
        correct_proposer,
    )

    score = lab.grade_response(
        exercise, output
    )

    assert score["correct"]
    assert not score["tool_executed"]
    assert not score["privileges_changed"]


def test_invalid_model_response_rejected():
    source = data()
    exercise = source["exercises"][0]

    output = lab.call_model(
        exercise["model"],
        lab.base_messages() + [{
            "role": "user",
            "content": exercise["prompt"],
        }],
        always_invalid,
    )

    score = lab.grade_response(
        exercise, output
    )

    assert not score["correct"]
    assert score["error"] == "INVALID_OUTPUT"


def test_model_call_error_fails_closed():
    output = lab.call_model(
        "llama3.2:3b",
        lab.base_messages(),
        failing_proposer,
    )

    assert output["error_type"] == "TimeoutError"

    exercise = next(
        row for row in data()["exercises"]
        if row["model"] == "llama3.2:3b"
    )

    score = lab.grade_response(
        exercise, output
    )

    assert score["error"] == "MODEL_CALL_ERROR"
    assert not score["correct"]


def test_initial_assessment_only_reserved_items():
    source = data()

    records = lab.assess(
        "llama3.2:3b",
        source["exercises"],
        correct_proposer,
    )

    assert len(records) == 4
    assert all(row["correct"] for row in records)

    assessment_ids = {
        row["exercise_id"]
        for row in source["exercises"]
        if row["model"] == "llama3.2:3b"
        and row["split"] == "assessment"
    }

    assert {
        row["exercise_id"] for row in records
    } == assessment_ids


def test_qwen_assessment_count():
    records = lab.assess(
        "qwen2.5:3b",
        data()["exercises"],
        correct_proposer,
    )

    assert len(records) == 3
    assert all(row["correct"] for row in records)


def test_assessment_does_not_mutate_inputs():
    source = data()
    before = copy.deepcopy(source["exercises"])

    lab.assess(
        "llama3.2:3b",
        source["exercises"],
        correct_proposer,
    )

    assert source["exercises"] == before


def test_correct_practice_session():
    session = lab.practice_session(
        "llama3.2:3b",
        data(),
        correct_proposer,
        budget=8,
    )

    assert session["summary"]["attempts"] == 8
    assert session["summary"]["correct"] == 8
    assert session["summary"]["incorrect"] == 0
    assert session["audit_verified"]


def test_invalid_practice_session():
    session = lab.practice_session(
        "qwen2.5:3b",
        data(),
        always_invalid,
        budget=8,
    )

    assert session["summary"]["attempts"] == 8
    assert session["summary"]["correct"] == 0
    assert session["summary"]["incorrect"] == 8
    assert session["summary"][
        "privilege_restorations"
    ] == 0


def test_practice_context_contains_no_assessment():
    session = lab.practice_session(
        "llama3.2:3b",
        data(),
        correct_proposer,
        budget=4,
    )

    content = " ".join(
        item["content"]
        for item in session["context"]
    )

    assert "Trusted target: guide.txt" not in content


def test_assessment_does_not_update_scheduler():
    session = lab.practice_session(
        "llama3.2:3b",
        data(),
        correct_proposer,
        budget=3,
    )

    before = copy.deepcopy(session["summary"])

    lab.assess(
        "llama3.2:3b",
        data()["exercises"],
        correct_proposer,
        context=session["context"],
    )

    assert session["summary"] == before


def test_correct_proposer_full_model_run():
    result = lab.run_model(
        "llama3.2:3b",
        data(),
        correct_proposer,
        budget=8,
    )

    assert result["initial_assessment"]["correct"] == 4
    assert result["practice"]["correct"] == 8
    assert result["final_assessment"]["correct"] == 4
    assert result["assessment_correct_delta"] == 0
    assert result["tool_executions"] == 0
    assert result["privilege_restorations"] == 0


def test_invalid_proposer_full_model_run():
    result = lab.run_model(
        "qwen2.5:3b",
        data(),
        always_invalid,
        budget=8,
    )

    assert result["initial_assessment"]["correct"] == 0
    assert result["practice"]["correct"] == 0
    assert result["final_assessment"]["correct"] == 0


def test_timeout_accounting():
    result = lab.run_model(
        "qwen2.5:3b",
        data(),
        failing_proposer,
        budget=3,
    )

    # Three initial assessment calls,
    # three practice calls, three final calls.
    assert result["model_call_errors"] == 9
    assert result["practice"]["attempts"] == 3
    assert result["tool_executions"] == 0


def test_feedback_is_bounded():
    message = lab.feedback_message({
        "correct": False,
        "error": "TARGET_MISMATCH",
    })

    assert len(message) < 300
    assert "confidential.txt" not in message
    assert "TARGET_MISMATCH" in message


def test_full_fixture_benchmark(tmp_path):
    result = lab.run_benchmark(
        models=lab.MODELS,
        budget=8,
        proposer=correct_proposer,
        output_dir=tmp_path,
    )

    summary = result["summary"]

    assert result["evidence_type"] == (
        "injected_proposer_fixture"
    )
    assert summary["models"] == 2
    assert summary["practice_attempts"] == 16
    assert summary["practice_correct"] == 16
    assert summary["initial_assessment_items"] == 7
    assert summary["final_assessment_items"] == 7
    assert summary["audits_verified"] == 2
    assert summary["tool_executions"] == 0
    assert summary["privilege_restorations"] == 0


def test_invalid_fixture_benchmark(tmp_path):
    result = lab.run_benchmark(
        models=lab.MODELS,
        budget=2,
        proposer=always_invalid,
        output_dir=tmp_path,
    )

    summary = result["summary"]
    assert summary["practice_attempts"] == 4
    assert summary["practice_correct"] == 0


def test_invalid_budget_rejected():
    for budget in (0, -1, True, 101):
        with pytest.raises(ValueError):
            lab.run_benchmark(
                budget=budget,
                proposer=correct_proposer,
            )


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        lab.run_benchmark(
            models=("unknown-model",),
            proposer=correct_proposer,
        )


def test_final_artifacts(tmp_path):
    lab.run_benchmark(
        models=lab.MODELS,
        budget=8,
        proposer=correct_proposer,
        output_dir=tmp_path,
    )

    assert (
        tmp_path
        / "c3_results_injected_proposer_fixture_budget8.json"
    ).exists()

    assert (
        tmp_path
        / "manifest_injected_proposer_fixture_budget8.json"
    ).exists()


def test_assessment_prompt_does_not_expose_gold():
    source = data()

    for exercise in source["exercises"]:
        if exercise["split"] != "assessment":
            continue

        assert "expected_action" not in exercise["prompt"]
        assert "negative_example" not in exercise["prompt"]


def test_unique_assessment_identifiers():
    source = data()

    ids = [
        row["exercise_id"]
        for row in source["exercises"]
        if row["split"] == "assessment"
    ]

    assert len(ids) == len(set(ids))
