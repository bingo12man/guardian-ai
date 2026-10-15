
import copy

import pytest

import p3_capabilities as p3


def test_manager_can_receive_confidential_capability():
    token = p3.issue_capability(
        "manager",
        "worker",
        "read_document",
        "confidential.txt",
        "task-1",
        "cap-1",
    )

    assert token is not None
    assert "signature" in token


def test_visitor_cannot_receive_confidential_capability():
    token = p3.issue_capability(
        "visitor",
        "worker",
        "read_document",
        "confidential.txt",
        "task-1",
        "cap-1",
    )

    assert token is None


def test_valid_manager_delegation():
    record = p3.run_trial(
        "valid_delegation", 1
    )

    assert record["verified"]["executed"] is True
    assert record["verified"]["reason"] == (
        "CAPABILITY_VERIFIED"
    )


def test_valid_visitor_public_delegation():
    record = p3.run_trial(
        "valid_public", 1
    )

    assert record["verified"]["executed"] is True


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("other_resource", "RESOURCE_MISMATCH"),
        ("wrong_worker", "WORKER_MISMATCH"),
        ("wrong_task", "TASK_MISMATCH"),
        ("expired_token", "TOKEN_EXPIRED"),
        ("revoked_token", "TOKEN_REVOKED"),
        ("tampered_scope", "BAD_SIGNATURE"),
        ("replayed_token", "TOKEN_REPLAY"),
        ("visitor_escalation", "MISSING_TOKEN"),
    ],
)
def test_invalid_capabilities_are_rejected(
    scenario, reason
):
    record = p3.run_trial(scenario, 1)

    assert record["verified"]["executed"] is False
    assert record["verified"]["reason"] == reason
    assert record["verified_invalid_execution"] is False


def test_weak_baseline_ignores_resource_scope():
    record = p3.run_trial("other_resource", 1)

    assert record["baseline"]["executed"] is True
    assert record["baseline_invalid_execution"] is True


def test_weak_baseline_ignores_revocation():
    record = p3.run_trial("revoked_token", 1)

    assert record["baseline"]["executed"] is True
    assert record["verified"]["executed"] is False


def test_tampering_breaks_signature():
    token, context, revoked, consumed = (
        p3.make_scenario("valid_delegation", 1)
    )

    modified = copy.deepcopy(token)
    modified["payload"]["resource"] = "internal.txt"

    allowed, reason = p3.verify_capability(
        modified, context, revoked, consumed
    )

    assert allowed is False
    assert reason == "BAD_SIGNATURE"


def test_capability_is_one_time_use():
    token, context, revoked, consumed = (
        p3.make_scenario("valid_delegation", 1)
    )

    first = p3.verify_capability(
        token, context, revoked, consumed
    )

    second = p3.verify_capability(
        token, context, revoked, consumed
    )

    assert first == (True, "CAPABILITY_VERIFIED")
    assert second == (False, "TOKEN_REPLAY")


def test_failed_request_does_not_consume_token():
    token, context, revoked, consumed = (
        p3.make_scenario("valid_delegation", 1)
    )

    incorrect = dict(context)
    incorrect["resource"] = "internal.txt"

    first = p3.verify_capability(
        token, incorrect, revoked, consumed
    )

    assert first == (False, "RESOURCE_MISMATCH")
    assert consumed == set()

    second = p3.verify_capability(
        token, context, revoked, consumed
    )

    assert second == (True, "CAPABILITY_VERIFIED")


def test_revocation_after_issuance():
    token, context, revoked, consumed = (
        p3.make_scenario("valid_delegation", 1)
    )

    revoked.add(token["payload"]["token_id"])

    allowed, reason = p3.verify_capability(
        token, context, revoked, consumed
    )

    assert allowed is False
    assert reason == "TOKEN_REVOKED"


def test_invalid_signature_rejected():
    token, context, revoked, consumed = (
        p3.make_scenario("valid_delegation", 1)
    )

    token["signature"] = "0" * 64

    allowed, reason = p3.verify_capability(
        token, context, revoked, consumed
    )

    assert allowed is False
    assert reason == "BAD_SIGNATURE"


def test_unknown_policy_rejected():
    token, context, revoked, consumed = (
        p3.make_scenario("valid_delegation", 1)
    )

    with pytest.raises(ValueError):
        p3.evaluate(
            "nonexistent_policy",
            token,
            context,
        )


def test_offline_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(p3, "RESULTS", tmp_path)

    report = p3.run_experiment(repeats=3)
    summary = report["summary"]

    assert summary["trials"] == 30

    baseline = summary["policies"][
        "bearer_baseline"
    ]

    guardian = summary["policies"][
        "capability_guard"
    ]

    assert baseline["executions"] == 27
    assert baseline["invalid_executions"] == 21
    assert baseline["legitimate_completions"] == 6

    assert guardian["executions"] == 6
    assert guardian["invalid_executions"] == 0
    assert guardian["legitimate_completions"] == 6


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        p3.make_scenario("unknown", 1)
