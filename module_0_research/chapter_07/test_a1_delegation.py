
import pytest

import a1_delegation as a1


def read_action(document):
    return {
        "tool": "read_document",
        "document": document,
    }


def fake_planner(task, document):
    return read_action(document)


def fake_worker(delegation):
    return read_action(
        delegation["document"]
    )


def scenario(name):
    return next(
        s for s in a1.SCENARIOS
        if s["id"] == name
    )


def test_visitor_can_read_public():
    result = a1.execute_guarded(
        read_action("public.txt"),
        "visitor",
        "worker",
        "public.txt",
    )

    assert result["status"] == "success"
    assert result["executed"] is True


def test_visitor_cannot_delegate_confidential_read():
    result = a1.execute_guarded(
        read_action("confidential.txt"),
        "visitor",
        "worker",
        "confidential.txt",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "REQUESTER_UNAUTHORIZED"
    assert result["executed"] is False


def test_manager_can_delegate_confidential_read():
    result = a1.execute_guarded(
        read_action("confidential.txt"),
        "manager",
        "worker",
        "confidential.txt",
    )

    assert result["status"] == "success"


def test_requester_identity_cannot_be_forged():
    test_case = scenario(
        "visitor_fake_manager"
    )

    result = a1.run_trial(
        test_case,
        1,
        planner=fake_planner,
        worker_fn=fake_worker,
    )

    assert result["blocked"] is True
    assert result["execution"]["reason"] == (
        "REQUESTER_UNAUTHORIZED"
    )


def test_worker_cannot_read_unrequested_document():
    result = a1.execute_guarded(
        read_action("confidential.txt"),
        "manager",
        "worker",
        "public.txt",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "TASK_MISMATCH"


def test_unknown_requester_is_denied():
    result = a1.execute_guarded(
        read_action("public.txt"),
        "unknown",
        "worker",
        "public.txt",
    )

    assert result["status"] == "blocked"


def test_unknown_worker_is_denied():
    result = a1.execute_guarded(
        read_action("public.txt"),
        "visitor",
        "missing_worker",
        "public.txt",
    )

    assert result["status"] == "blocked"


@pytest.mark.parametrize(
    "raw",
    [
        {},
        [],
        None,
        {"tool": "read_document"},
        {
            "tool": "read_document",
            "document": "public.txt",
            "admin": True,
        },
        {
            "tool": "read_document",
            "document": 123,
        },
        {
            "tool": "finish",
            "document": "public.txt",
        },
    ],
)
def test_invalid_model_actions_rejected(raw):
    assert a1.parse_action(raw) is None


def test_model_generated_action_is_not_permission():
    action = read_action(
        "confidential.txt"
    )

    allowed, reason = a1.guardian_check(
        action,
        "visitor",
        "worker",
        "confidential.txt",
    )

    assert allowed is False
    assert reason == "REQUESTER_UNAUTHORIZED"


def test_planner_and_worker_both_run():
    calls = {
        "planner": 0,
        "worker": 0,
    }

    def planner(task, document):
        calls["planner"] += 1
        return read_action(document)

    def worker(delegation):
        calls["worker"] += 1
        return read_action(
            delegation["document"]
        )

    result = a1.run_trial(
        scenario("visitor_public"),
        1,
        planner=planner,
        worker_fn=worker,
    )

    assert calls["planner"] == 1
    assert calls["worker"] == 1
    assert result["completed"] is True


def test_worker_tool_change_is_blocked():
    result = a1.run_trial(
        scenario("manager_public"),
        1,
        planner=fake_planner,
        worker_fn=lambda delegation: (
            read_action("confidential.txt")
        ),
    )

    assert result["blocked"] is True
    assert result["execution"]["reason"] == (
        "TASK_MISMATCH"
    )


def test_model_error_prevents_execution():
    def failing_planner(task, document):
        raise ValueError(
            "Invalid model response"
        )

    result = a1.run_trial(
        scenario("manager_public"),
        1,
        planner=failing_planner,
        worker_fn=fake_worker,
    )

    assert result["error"] is not None
    assert result["execution"]["executed"] is False


def test_planner_finish_skips_worker():
    def unexpected_worker(delegation):
        raise AssertionError(
            "Worker should not be called"
        )

    result = a1.run_trial(
        scenario("visitor_public"),
        1,
        planner=lambda task, document: {
            "tool": "finish",
            "document": "",
        },
        worker_fn=unexpected_worker,
    )

    assert result["execution"]["status"] == "finished"
    assert result["completed"] is False


def test_synthetic_confidential_case_is_safe():
    result = a1.run_trial(
        scenario("visitor_confidential"),
        1,
        planner=fake_planner,
        worker_fn=fake_worker,
    )

    assert result["unauthorized_proposal"] is True
    assert result["blocked"] is True
    assert result["unauthorized_execution"] is False


def test_offline_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(
        a1, "RESULTS", tmp_path
    )

    report = a1.run_experiment(
        repeats=3,
        offline=True,
    )

    summary = report["summary"]

    assert summary["trials"] == 15
    assert summary["valid_planner_actions"] == 15
    assert summary["valid_worker_actions"] == 15
    assert summary["guardian_blocks"] == 6
    assert summary["legitimate_completions"] == 9
    assert summary["unauthorized_executions"] == 0
    assert summary["model_errors"] == 0


def test_failed_worker_call_has_no_execution():
    def failing_worker(delegation):
        raise ValueError("Worker unavailable")

    result = a1.run_trial(
        scenario("visitor_public"),
        1,
        planner=fake_planner,
        worker_fn=failing_worker,
    )

    assert result["execution"]["status"] == (
        "model_error"
    )
    assert result["execution"]["executed"] is False
