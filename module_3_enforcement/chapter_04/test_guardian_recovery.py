
import copy

import pytest

from guardian_recovery import (
    GuardianRecoveryGateway,
    SyntheticFailureBackend,
    run_bounded_action,
    verify_audit,
)


KEY = b"R" * 32


def setup(fail_before=0, ambiguous_before=0):
    backend = SyntheticFailureBackend(
        fail_before=fail_before,
        ambiguous_before=ambiguous_before,
    )
    gateway = GuardianRecoveryGateway(
        backend=backend,
        signing_key=KEY,
    )
    task = gateway.register_task(
        "visitor", "read_document", "public.txt"
    )
    return gateway, task


def submit(gateway, task, tool, argument):
    token = gateway.issue_capability(
        task.task_id, tool, argument
    )
    return gateway.agent_port().submit(
        task.task_id,
        {"tool": tool, "argument": argument},
        token,
    )


def test_normal_execution():
    gateway, task = setup()
    result = submit(
        gateway, task,
        "search_documents", "public.txt",
    )
    assert result["status"] == "SUCCEEDED"
    assert result["effect_executed"]
    assert gateway.get_stage(task.task_id) == "need_read"


def test_backend_failure_does_not_advance():
    gateway, task = setup(fail_before=1)
    result = submit(
        gateway, task,
        "search_documents", "public.txt",
    )
    assert result["status"] == "RETRYABLE_FAILURE"
    assert not result["effect_executed"]
    assert gateway.get_stage(task.task_id) == "need_search"
    assert gateway.tool_effects() == []


def test_bounded_retry_recovers():
    gateway, task = setup(fail_before=1)
    result = run_bounded_action(
        gateway, task.task_id,
        "search_documents", "public.txt",
        max_attempts=3,
    )
    assert result["recovered"]
    assert result["attempts"] == 2
    assert result["final"]["status"] == "SUCCEEDED"
    assert len(gateway.tool_effects()) == 1


def test_exhaustion_never_advances():
    gateway, task = setup(fail_before=10)
    result = run_bounded_action(
        gateway, task.task_id,
        "search_documents", "public.txt",
        max_attempts=3,
    )
    assert result["attempts"] == 3
    assert result["budget_exhausted"]
    assert not result["recovered"]
    assert gateway.get_stage(task.task_id) == "need_search"
    assert gateway.tool_effects() == []


def test_failure_consumes_capability():
    gateway, task = setup(fail_before=1)
    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )
    action = {
        "tool": "search_documents",
        "argument": "public.txt",
    }
    first = gateway.agent_port().submit(
        task.task_id, action, token
    )
    second = gateway.agent_port().submit(
        task.task_id, action, token
    )
    assert first["status"] == "RETRYABLE_FAILURE"
    assert second["reason"] == "CAPABILITY_REPLAY"
    assert gateway.tool_effects() == []


def test_wrong_target_never_reaches_backend():
    gateway, task = setup()
    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )
    result = gateway.agent_port().submit(
        task.task_id,
        {
            "tool": "search_documents",
            "argument": "confidential.txt",
        },
        token,
    )
    assert result["status"] == "DENIED"
    assert not result["effect_executed"]
    assert gateway.tool_effects() == []


def test_identity_spoof_denied():
    gateway, task = setup()
    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )
    response = gateway.agent_port().submit(
        task.task_id,
        {
            "tool": "search_documents",
            "argument": "public.txt",
            "requester": "manager",
        },
        token,
    )
    assert response["reason"] == "INVALID_ACTION"
    assert gateway.tool_effects() == []


def test_full_workflow():
    gateway, task = setup()
    for tool, argument in (
        ("search_documents", "public.txt"),
        ("read_document", "public.txt"),
        ("finish", "done"),
    ):
        result = run_bounded_action(
            gateway, task.task_id,
            tool, argument,
        )
        assert result["final"]["status"] == "SUCCEEDED"

    assert result["final"]["completed"]
    assert gateway.get_stage(task.task_id) == "completed"
    assert len(gateway.tool_effects()) == 3


def test_failure_during_read_preserves_stage():
    gateway, task = setup(fail_before=1)

    run_bounded_action(
        gateway, task.task_id,
        "search_documents", "public.txt",
    )

    result = submit(
        gateway, task, "read_document", "public.txt"
    )

    assert result["status"] == "RETRYABLE_FAILURE"
    assert gateway.get_stage(task.task_id) == "need_read"
    assert len(gateway.tool_effects()) == 1


def test_ambiguous_outcome_requires_reconciliation():
    gateway, task = setup(ambiguous_before=1)

    result = run_bounded_action(
        gateway, task.task_id,
        "search_documents", "public.txt",
    )

    assert result["attempts"] == 1
    assert result["final"]["status"] == (
        "RECONCILIATION_REQUIRED"
    )
    assert gateway.get_stage(task.task_id) == "need_search"

    with pytest.raises(RuntimeError):
        gateway.issue_capability(
            task.task_id,
            "search_documents",
            "public.txt",
        )


def test_ambiguous_task_rejects_further_submission():
    gateway, task = setup(ambiguous_before=1)
    token = gateway.issue_capability(
        task.task_id,
        "search_documents",
        "public.txt",
    )
    action = {
        "tool": "search_documents",
        "argument": "public.txt",
    }

    first = gateway.agent_port().submit(
        task.task_id, action, token
    )
    second = gateway.agent_port().submit(
        task.task_id, action, token
    )

    assert first["status"] == "RECONCILIATION_REQUIRED"
    assert second["reason"] == "TASK_REQUIRES_RECONCILIATION"


def test_invalid_retry_budget():
    gateway, task = setup()

    for budget in (0, -1, True):
        with pytest.raises(ValueError):
            run_bounded_action(
                gateway, task.task_id,
                "search_documents", "public.txt",
                max_attempts=budget,
            )


def test_unauthorized_registration_denied():
    gateway, _ = setup()

    with pytest.raises(PermissionError):
        gateway.register_task(
            "visitor",
            "read_document",
            "confidential.txt",
        )


def test_audit_chain():
    gateway, task = setup(fail_before=1)
    run_bounded_action(
        gateway, task.task_id,
        "search_documents", "public.txt",
    )
    assert verify_audit(
        gateway.audit_events(),
        gateway.final_audit_hash(),
    )


def test_tampered_audit_rejected():
    gateway, task = setup()
    submit(
        gateway, task,
        "search_documents", "public.txt",
    )
    forged = copy.deepcopy(gateway.audit_events())
    forged[0]["response"]["effect_executed"] = False

    with pytest.raises(ValueError):
        verify_audit(
            forged, gateway.final_audit_hash()
        )


def test_denied_action_never_causes_effect():
    gateway, task = setup()

    response = gateway.agent_port().submit(
        task.task_id,
        {
            "tool": "search_documents",
            "argument": "public.txt",
        },
        "invalid",
    )

    assert response["status"] == "DENIED"
    assert gateway.summary()[
        "denied_actions_with_effects"
    ] == 0
    assert gateway.tool_effects() == []
