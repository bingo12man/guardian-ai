
import copy

import pytest

from guardian_capabilities import (
    CapabilityError,
    CapabilityGateway,
)


KEY = b"K" * 32


def setup_gateway(clock=None):
    gateway = CapabilityGateway(
        signing_key=KEY,
        clock=clock,
    )
    task = gateway.register_task(
        "visitor", "read_document", "public.txt"
    )
    return gateway, task


def search_action(target="public.txt"):
    return {
        "tool": "search_documents",
        "argument": target,
    }


def test_valid_capability():
    gateway, task = setup_gateway()
    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    result = gateway.submit(
        task.task_id, search_action(), token
    )

    assert result["allowed"]
    assert result["executed"]
    assert result["reason"] == "SEARCH_OK"


def test_missing_capability_denied():
    gateway, task = setup_gateway()

    result = gateway.submit(
        task.task_id, search_action(), None
    )

    assert not result["allowed"]
    assert not result["executed"]


def test_task_id_alone_insufficient():
    gateway, task = setup_gateway()

    result = gateway.submit(
        task.task_id, search_action(), ""
    )

    assert not result["allowed"]


def test_tampered_signature():
    gateway, task = setup_gateway()

    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    body, signature = token.split(".")
    changed = (
        "A" if signature[0] != "A" else "B"
    ) + signature[1:]

    result = gateway.submit(
        task.task_id,
        search_action(),
        body + "." + changed,
    )

    assert result["reason"] == "INVALID_SIGNATURE"
    assert not result["executed"]


def test_wrong_target_with_valid_capability():
    gateway, task = setup_gateway()

    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    result = gateway.submit(
        task.task_id,
        search_action("confidential.txt"),
        token,
    )

    assert result["reason"] == (
        "CAPABILITY_SCOPE_MISMATCH"
    )
    assert not result["executed"]
    assert gateway.get_stage(task.task_id) == (
        "need_search"
    )


def test_capability_cannot_be_issued_for_wrong_target():
    gateway, task = setup_gateway()

    with pytest.raises(CapabilityError):
        gateway.issue_capability(
            task.task_id,
            "search_documents",
            "confidential.txt",
        )


def test_replay_denied():
    gateway, task = setup_gateway()

    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    first = gateway.submit(
        task.task_id, search_action(), token
    )
    second = gateway.submit(
        task.task_id, search_action(), token
    )

    assert first["allowed"]
    assert second["reason"] == "CAPABILITY_REPLAY"
    assert not second["executed"]


def test_expired_token():
    now = [1000.0]

    gateway, task = setup_gateway(
        clock=lambda: now[0]
    )

    token = gateway.issue_capability(
        task.task_id,
        "search_documents",
        "public.txt",
        ttl_seconds=5,
    )

    now[0] += 6

    result = gateway.submit(
        task.task_id, search_action(), token
    )

    assert result["reason"] == (
        "EXPIRED_OR_INVALID_TIME"
    )
    assert not result["executed"]


def test_future_dated_token_denied_after_clock_rollback():
    now = [1000.0]
    gateway, task = setup_gateway(
        clock=lambda: now[0]
    )

    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    now[0] = 999.0

    result = gateway.submit(
        task.task_id, search_action(), token
    )

    assert result["reason"] == (
        "EXPIRED_OR_INVALID_TIME"
    )


def test_wrong_task_binding():
    gateway, first = setup_gateway()

    second = gateway.register_task(
        "visitor", "read_document", "public.txt"
    )

    token = gateway.issue_capability(
        first.task_id,
        "search_documents",
        "public.txt",
    )

    result = gateway.submit(
        second.task_id, search_action(), token
    )

    assert result["reason"] == (
        "TASK_BINDING_MISMATCH"
    )


def test_invalid_key_rejected():
    with pytest.raises(ValueError):
        CapabilityGateway(signing_key=b"short")


def test_invalid_ttl_rejected():
    gateway, task = setup_gateway()

    for value in (-1, 0, 301, float("nan")):
        with pytest.raises(CapabilityError):
            gateway.issue_capability(
                task.task_id,
                "search_documents",
                "public.txt",
                ttl_seconds=value,
            )


def test_action_schema_spoofing():
    gateway, task = setup_gateway()

    token = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    result = gateway.submit(
        task.task_id,
        {
            "tool": "search_documents",
            "argument": "public.txt",
            "requester": "manager",
        },
        token,
    )

    assert result["reason"] == "INVALID_ACTION"


def test_full_workflow_needs_three_capabilities():
    gateway, task = setup_gateway()

    for tool, argument in (
        ("search_documents", "public.txt"),
        ("read_document", "public.txt"),
        ("finish", "done"),
    ):
        token = gateway.issue_capability(
            task.task_id, tool, argument
        )

        result = gateway.submit(
            task.task_id,
            {"tool": tool, "argument": argument},
            token,
        )

        assert result["allowed"]

    assert result["completed"]
    assert gateway.get_stage(task.task_id) == "completed"


def test_capability_for_prior_stage_denied():
    gateway, task = setup_gateway()

    first = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )
    unused = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    gateway.submit(
        task.task_id, search_action(), first
    )

    result = gateway.submit(
        task.task_id, search_action(), unused
    )

    assert result["reason"] == (
        "CAPABILITY_STAGE_MISMATCH"
    )


def test_capability_not_issued_after_completion():
    gateway, task = setup_gateway()

    for tool, argument in (
        ("search_documents", "public.txt"),
        ("read_document", "public.txt"),
        ("finish", "done"),
    ):
        token = gateway.issue_capability(
            task.task_id, tool, argument
        )
        gateway.submit(
            task.task_id,
            {"tool": tool, "argument": argument},
            token,
        )

    with pytest.raises(CapabilityError):
        gateway.issue_capability(
            task.task_id, "finish", "done"
        )


def test_unauthorized_registration():
    gateway = CapabilityGateway(signing_key=KEY)

    with pytest.raises(PermissionError):
        gateway.register_task(
            "visitor",
            "read_document",
            "confidential.txt",
        )


def test_audit_copy_cannot_mutate_original():
    gateway, task = setup_gateway()

    gateway.submit(
        task.task_id, search_action(), "bad"
    )

    copied = gateway.audit_events()
    copied[0]["decision"]["allowed"] = True

    assert not gateway.audit_events()[0][
        "decision"
    ]["allowed"]


def test_separate_signing_keys():
    first, task = setup_gateway()

    token = first.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    second = CapabilityGateway(
        signing_key=b"Z" * 32
    )

    result = second.submit(
        task.task_id, search_action(), token
    )

    assert not result["allowed"]
    assert result["reason"] == "INVALID_SIGNATURE"


def test_no_unauthorized_synthetic_execution():
    gateway, task = setup_gateway()

    legitimate = gateway.issue_capability(
        task.task_id, "search_documents", "public.txt"
    )

    attacks = [
        (search_action("confidential.txt"), legitimate),
        (search_action(), "invalid"),
        ({"tool": "read_document",
          "argument": "confidential.txt"}, legitimate),
        ({"tool": "finish",
          "argument": "done"}, legitimate),
    ]

    for action, token in attacks:
        decision = gateway.submit(
            task.task_id, action, token
        )
        assert not decision["executed"]

    assert gateway.summary()[
        "unauthorized_simulated_executions"
    ] == 0
