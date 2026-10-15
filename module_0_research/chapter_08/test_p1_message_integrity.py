
import copy

import pytest

import p1_message_integrity as p1


def test_valid_public_signature():
    envelope, context = p1.make_scenario(
        "valid_public", 1
    )

    allowed, reason = p1.verify_envelope(
        envelope, context, set()
    )

    assert allowed is True
    assert reason == "VERIFIED"


def test_valid_manager_access():
    record = p1.run_trial(
        "valid_manager", 1
    )

    assert record["verified"]["executed"] is True
    assert record["verified_unauthorized_access"] is False


def test_tampering_invalidates_signature():
    envelope, context = p1.make_scenario(
        "tampered_document", 1
    )

    allowed, reason = p1.verify_envelope(
        envelope, context, set()
    )

    assert allowed is False
    assert reason == "BAD_SIGNATURE"


def test_spoofed_sender_is_rejected():
    envelope, context = p1.make_scenario(
        "spoofed_sender", 1
    )

    allowed, reason = p1.verify_envelope(
        envelope, context, set()
    )

    assert allowed is False
    assert reason == "SENDER_MISMATCH"


def test_replay_is_rejected():
    envelope, context = p1.make_scenario(
        "valid_public", 1
    )

    seen = set()

    first = p1.verify_envelope(
        envelope, context, seen
    )

    second = p1.verify_envelope(
        envelope, context, seen
    )

    assert first == (True, "VERIFIED")
    assert second == (False, "REPLAY")


def test_expired_message_rejected():
    envelope, context = p1.make_scenario(
        "expired_message", 1
    )

    allowed, reason = p1.verify_envelope(
        envelope, context, set()
    )

    assert allowed is False
    assert reason == "EXPIRED_OR_NOT_YET_VALID"


def test_signed_request_cannot_escalate_scope():
    envelope, context = p1.make_scenario(
        "scope_escalation", 1
    )

    allowed, reason = p1.verify_envelope(
        envelope, context, set()
    )

    assert allowed is False
    assert reason == "REQUESTER_UNAUTHORIZED"


def test_naive_worker_only_policy_is_vulnerable():
    record = p1.run_trial(
        "tampered_document", 1
    )

    assert record["naive"]["executed"] is True
    assert record["naive_unauthorized_access"] is True

    assert record["verified"]["executed"] is False


def test_valid_signature_does_not_grant_permission():
    record = p1.run_trial(
        "scope_escalation", 1
    )

    assert record["naive"]["executed"] is True
    assert record["verified"]["executed"] is False


def test_modified_signature_rejected():
    envelope, context = p1.make_scenario(
        "valid_public", 1
    )

    envelope = copy.deepcopy(envelope)
    envelope["signature"] = "0" * 64

    allowed, reason = p1.verify_envelope(
        envelope, context, set()
    )

    assert allowed is False
    assert reason == "BAD_SIGNATURE"


def test_malformed_envelope_rejected():
    assert p1.verify_envelope(
        {}, {}, set()
    ) == (False, "INVALID_ENVELOPE")


def test_unrecognized_requester_rejected():
    assert p1.authorized(
        "unknown", "public.txt"
    ) is False


def test_unknown_document_rejected():
    assert p1.worker_authorized(
        "missing.txt"
    ) is False


def test_unknown_policy_raises():
    envelope, context = p1.make_scenario(
        "valid_public", 1
    )

    with pytest.raises(ValueError):
        p1.evaluate(
            envelope,
            context,
            "unknown_policy",
            set(),
        )


def test_same_message_cannot_be_executed_twice():
    envelope, context = p1.make_scenario(
        "valid_public", 1
    )

    seen = set()

    first = p1.evaluate(
        envelope,
        context,
        "verified_receiver",
        seen,
    )

    second = p1.evaluate(
        envelope,
        context,
        "verified_receiver",
        seen,
    )

    assert first["executed"] is True
    assert second["executed"] is False
    assert second["reason"] == "REPLAY"


def test_three_repeat_benchmark(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        p1, "RESULTS", tmp_path
    )

    report = p1.run_experiment(
        repeats=3
    )

    summary = report["summary"]

    assert summary["trials"] == 21

    naive = summary["policies"][
        "naive_receiver"
    ]

    verified = summary["policies"][
        "verified_receiver"
    ]

    assert naive["executions"] == 21
    assert naive["invalid_message_executions"] == 15
    assert naive["unauthorized_access_executions"] == 6

    assert verified["executions"] == 6
    assert verified["invalid_message_executions"] == 0
    assert verified["unauthorized_access_executions"] == 0


def test_all_negative_cases_denied():
    invalid_cases = (
        "tampered_document",
        "spoofed_sender",
        "replay_message",
        "expired_message",
        "scope_escalation",
    )

    for name in invalid_cases:
        record = p1.run_trial(name, 1)

        assert record["verified"]["executed"] is False
        assert record[
            "verified_invalid_message_execution"
        ] is False
