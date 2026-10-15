
import json

import pytest

import m2_multistep_agents as lab


def task(name):
    return next(
        t for t in lab.SCENARIOS
        if t["id"] == name
    )


def action(tool, argument):
    return json.dumps({
        "tool": tool,
        "argument": argument,
    })


def test_scenario_count():
    assert len(lab.SCENARIOS) == 8

    assert len({
        t["id"] for t in lab.SCENARIOS
    }) == 8


def test_permissions():
    assert lab.can_access(
        "visitor", "public.txt"
    )

    assert not lab.can_access(
        "visitor", "confidential.txt"
    )

    assert lab.can_access(
        "manager", "confidential.txt"
    )

    assert not lab.can_access(
        "unknown", "public.txt"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "Not JSON",
        "[]",
        '{"tool":"delete_file","argument":"public.txt"}',
        '{"tool":"read_document","argument":5}',
        '{"tool":"read_document","argument":"unknown.txt"}',
        '{"tool":"finish","argument":"other"}',
        '{"tool":"read_document"}',
        '{"tool":"read_document","argument":"public.txt","extra":1}',
    ],
)
def test_invalid_actions(raw):
    assert lab.parse_action(raw) is None


def test_valid_actions():
    for tool, argument in [
        ("search_documents", "public.txt"),
        ("read_document", "public.txt"),
        ("finish", "done"),
    ]:
        assert lab.parse_action(
            action(tool, argument)
        ) is not None


def test_correct_tool_order():
    current = task("public_clean")

    event, _, stage = lab.execute_synthetic(
        current,
        {
            "tool": "search_documents",
            "argument": "public.txt",
        },
        "need_search",
    )

    assert event == "search"
    assert stage == "need_read"

    event, _, stage = lab.execute_synthetic(
        current,
        {
            "tool": "read_document",
            "argument": "public.txt",
        },
        stage,
    )

    assert event == "read"
    assert stage == "need_finish"

    event, _, stage = lab.execute_synthetic(
        current,
        {
            "tool": "finish",
            "argument": "done",
        },
        stage,
    )

    assert event == "finished"
    assert stage == "completed"


def test_wrong_target_blocked():
    event, reason, stage = (
        lab.execute_synthetic(
            task("public_clean"),
            {
                "tool": "read_document",
                "argument": "confidential.txt",
            },
            "need_read",
        )
    )

    assert event == "blocked"
    assert reason == "TARGET_MISMATCH"
    assert stage == "need_read"


def test_premature_finish_blocked():
    event, _, _ = lab.execute_synthetic(
        task("public_clean"),
        {
            "tool": "finish",
            "argument": "done",
        },
        "need_search",
    )

    assert event == "blocked"


def test_clean_offline_trial():
    row = lab.run_trial(
        "llama-test",
        task("public_clean"),
        1,
        offline=True,
    )

    assert row["completed"]
    assert row["model_calls"] == 3
    assert row["unsafe_proposals"] == 0
    assert row["guardian_blocks"] == 0
    assert row["error"] is None


def test_manager_offline_trial():
    row = lab.run_trial(
        "qwen-test",
        task("manager_clean"),
        1,
        offline=True,
    )

    assert row["completed"]
    assert row["final_stage"] == "completed"


def test_adversarial_fixture_recovery():
    row = lab.run_trial(
        "llama-test",
        task("private_instruction"),
        1,
        offline=True,
        adversarial_fixture=True,
    )

    assert row["completed"]
    assert row["model_calls"] == 4
    assert row["unsafe_proposals"] == 1
    assert row["guardian_blocks"] == 1
    assert row["unauthorized_executions"] == 0


def test_adversarial_fixture_step_budget():
    row = lab.run_trial(
        "llama-test",
        task("private_instruction"),
        1,
        offline=True,
        adversarial_fixture=True,
        max_steps=3,
    )

    assert not row["completed"]
    assert row["model_calls"] == 3
    assert row["stopped_reason"] == (
        "STEP_BUDGET_EXHAUSTED"
    )


def test_invalid_output_recovery():
    def proposer(model, task, stage, step):
        if step == 1:
            return "not-json", 0.0

        return lab.offline_proposer(
            model, task, stage, step
        )

    row = lab.run_trial(
        "llama-test",
        task("public_clean"),
        1,
        proposer=proposer,
    )

    assert row["invalid_outputs"] == 1
    assert row["guardian_blocks"] == 1
    assert row["completed"]
    assert row["model_calls"] == 4


def test_model_error():
    def broken(model, task, stage, step):
        raise ConnectionError("Ollama unavailable")

    row = lab.run_trial(
        "llama-test",
        task("public_clean"),
        1,
        proposer=broken,
    )

    assert row["error"] is not None
    assert not row["completed"]
    assert row["stopped_reason"] == "MODEL_ERROR"


def test_restricted_original_request():
    unauthorized = {
        **task("public_clean"),
        "target": "confidential.txt",
    }

    row = lab.run_trial(
        "llama-test",
        unauthorized,
        1,
        offline=True,
    )

    assert row["model_calls"] == 0
    assert not row["completed"]
    assert row["stopped_reason"] == (
        "INVALID_ORIGINAL_TASK"
    )


def test_offline_experiment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        lab, "RESULTS", tmp_path
    )

    result = lab.run_experiment(
        ["llama-test", "qwen-test"],
        repeats=3,
        offline=True,
    )

    assert result["total_trials"] == 48

    for value in result["summary"].values():
        assert value["trials"] == 24
        assert value["completions"] == 24
        assert value["unsafe_proposals"] == 0
        assert value["errors"] == 0


def test_adversarial_experiment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        lab, "RESULTS", tmp_path
    )

    result = lab.run_experiment(
        ["llama-test", "qwen-test"],
        repeats=3,
        offline=True,
        adversarial_fixture=True,
    )

    assert result["total_trials"] == 48

    for value in result["summary"].values():
        assert value["trials"] == 24
        assert value["completions"] == 24
        assert value["unsafe_proposals"] == 15
        assert value["guardian_blocks"] == 15


def test_invalid_step_budget():
    with pytest.raises(ValueError):
        lab.run_trial(
            "llama-test",
            task("public_clean"),
            1,
            offline=True,
            max_steps=0,
        )


def test_duplicate_models_rejected():
    with pytest.raises(ValueError):
        lab.run_experiment(
            ["same", "same"],
            offline=True,
        )
