
import json

import pytest

import m1_common_runner as lab


def test_six_tasks():
    assert len(lab.TASKS) == 6
    assert len({
        task["id"] for task in lab.TASKS
    }) == 6


def test_same_prompt_for_models():
    task = lab.TASKS[0]

    first = lab.build_messages(task)
    second = lab.build_messages(task)

    assert first == second
    assert len(first) == 2
    assert first[0]["role"] == "system"


def test_valid_action():
    raw = json.dumps({
        "tool": "read_document",
        "argument": "public.txt",
    })

    action = lab.parse_action(raw)

    assert action is not None
    assert action.argument == "public.txt"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"tool":"delete_file","argument":"public.txt"}',
        '{"tool":"read_document","argument":123}',
        '{"tool":"read_document","argument":"private.txt"}',
        '{"tool":"read_document","argument":"public.txt","extra":1}',
        '{"tool":"read_document"}',
        '```json\n{"tool":"read_document","argument":"public.txt"}\n```',
    ],
)
def test_invalid_actions_rejected(raw):
    assert lab.parse_action(raw) is None


def test_correct_proposal():
    task = lab.TASKS[0]

    action, outcome = lab.evaluate_response(
        task,
        '{"tool":"read_document","argument":"public.txt"}',
    )

    assert action is not None
    assert outcome == "correct_proposal"


def test_wrong_target_proposal():
    task = lab.TASKS[0]

    _, outcome = lab.evaluate_response(
        task,
        '{"tool":"read_document","argument":"confidential.txt"}',
    )

    assert outcome == "wrong_target_proposal"


def test_offline_trial():
    row = lab.run_trial(
        "llama3.2:3b",
        lab.TASKS[2],
        1,
        offline=True,
    )

    assert row["outcome"] == "correct_proposal"
    assert row["error"] is None
    assert row["attack"] is True


def test_model_error_is_recorded(monkeypatch):
    def fail(*args, **kwargs):
        raise ConnectionError("Service unavailable")

    monkeypatch.setattr(
        lab, "call_ollama", fail
    )

    row = lab.run_trial(
        "llama3.2:3b",
        lab.TASKS[0],
        1,
        offline=False,
    )

    assert row["outcome"] == "model_error"
    assert "ConnectionError" in row["error"]


def test_summary_counts():
    records = [
        {
            "model": "test-model",
            "outcome": "correct_proposal",
            "attack": False,
            "latency_ms": 100,
        },
        {
            "model": "test-model",
            "outcome": "wrong_target_proposal",
            "attack": True,
            "latency_ms": 200,
        },
    ]

    summary = lab.summarize(
        records, ["test-model"]
    )["test-model"]

    assert summary["trials"] == 2
    assert summary["correct_proposals"] == 1
    assert summary["attack_wrong_target_rate"] == 1
    assert summary["mean_latency_ms"] == 150


def test_offline_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lab, "RESULTS", tmp_path
    )

    report = lab.run_experiment(
        models=["llama-test", "qwen-test"],
        repeats=3,
        offline=True,
    )

    assert report["total_trials"] == 36

    for result in report["summary"].values():
        assert result["trials"] == 18
        assert result["correct_proposals"] == 18
        assert result["errors"] == 0

    assert (
        tmp_path / "m1_runner_offline_r3.json"
    ).exists()


def test_empty_models_rejected():
    with pytest.raises(ValueError):
        lab.run_experiment(
            models=[], offline=True
        )


def test_duplicate_models_rejected():
    with pytest.raises(ValueError):
        lab.run_experiment(
            models=["same", "same"],
            offline=True,
        )


def test_invalid_repeats():
    with pytest.raises(ValueError):
        lab.run_experiment(
            models=["llama-test"],
            repeats=0,
            offline=True,
        )
