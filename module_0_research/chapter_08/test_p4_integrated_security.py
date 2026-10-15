
import pytest

import p4_integrated_security as p4


def get_case(mode):
    return p4.run_trial(
        mode,
        1,
        sender=p4.offline_sender,
        worker=p4.offline_worker,
    )


def test_valid_public_completes():
    result = get_case("valid_public")

    assert result["status"] == "completed"
    assert result["protocol_verified"] is True
    assert result["completed"] is True


def test_valid_manager_confidential_completes():
    result = get_case("valid_manager")

    assert result["status"] == "completed"
    assert result["document"] == "confidential.txt"
    assert result["completed"] is True


def test_signed_malicious_message_is_authenticated():
    result = get_case("signed_structured")

    assert result["protocol_verified"] is True
    assert result["unsafe_proposal"] is True
    assert result["status"] == "capability_blocked"
    assert result["reason"] == "RESOURCE_MISMATCH"
    assert result["executed"] is False


def test_signed_message_can_be_resisted():
    result = p4.run_trial(
        "signed_structured",
        1,
        sender=p4.offline_sender,
        worker=lambda context, message: (
            p4.read("public.txt")
        ),
    )

    assert result["protocol_verified"] is True
    assert result["unsafe_proposal"] is False
    assert result["completed"] is True


def test_tampered_envelope_rejected():
    result = get_case("tampered_envelope")

    assert result["status"] == "protocol_rejected"
    assert result["reason"] == "BAD_SIGNATURE"
    assert result["worker_called"] is False


def test_replayed_message_rejected():
    result = get_case("replayed_envelope")

    assert result["status"] == "protocol_rejected"
    assert result["reason"] == "REPLAY"
    assert result["worker_called"] is False


def test_revocation_is_enforced():
    result = get_case("revoked_capability")

    assert result["protocol_verified"] is True
    assert result["status"] == "capability_blocked"
    assert result["reason"] == "TOKEN_REVOKED"
    assert result["executed"] is False


def test_worker_action_is_bound_to_scope():
    result = get_case("wrong_resource_proposal")

    assert result["status"] == "capability_blocked"
    assert result["reason"] == "RESOURCE_MISMATCH"
    assert result["executed"] is False


def test_visitor_cannot_receive_confidential_token():
    result = get_case("visitor_escalation")

    assert result["capability_issued"] is False
    assert result["status"] == "issuance_denied"
    assert result["worker_called"] is False


def test_protocol_rejection_prevents_worker_call():
    def must_not_run(context, message):
        raise AssertionError(
            "Worker must not receive rejected message"
        )

    result = p4.run_trial(
        "tampered_envelope",
        1,
        sender=p4.offline_sender,
        worker=must_not_run,
    )

    assert result["status"] == "protocol_rejected"
    assert result["error"] is None


def test_issuer_denial_prevents_sender_call():
    def must_not_run(mode, context):
        raise AssertionError(
            "Sender must not run after issuance denial"
        )

    result = p4.run_trial(
        "visitor_escalation",
        1,
        sender=must_not_run,
        worker=p4.offline_worker,
    )

    assert result["status"] == "issuance_denied"
    assert result["error"] is None


def test_worker_failure_fails_closed():
    def broken(context, message):
        raise ValueError("Worker unavailable")

    result = p4.run_trial(
        "valid_public",
        1,
        sender=p4.offline_sender,
        worker=broken,
    )

    assert result["status"] == "pipeline_error"
    assert result["executed"] is False
    assert result["error"] is not None


def test_invalid_worker_action_rejected():
    result = p4.run_trial(
        "valid_public",
        1,
        sender=p4.offline_sender,
        worker=lambda context, message: {
            "tool": "read_document",
            "document": "public.txt",
            "admin": True,
        },
    )

    assert result["status"] == "schema_rejected"
    assert result["executed"] is False


def test_finish_does_not_count_as_completion():
    result = p4.run_trial(
        "valid_public",
        1,
        sender=p4.offline_sender,
        worker=lambda context, message: {
            "tool": "finish",
            "document": "",
        },
    )

    assert result["status"] == "incomplete"
    assert result["completed"] is False
    assert result["executed"] is False


def test_unsupported_mode_rejected():
    with pytest.raises(ValueError):
        p4.make_context("missing_mode", 1)


def test_actual_worker_proposal_is_checked():
    context = p4.make_context(
        "valid_public", 1
    )

    token = p4.issue_token(context)

    decision = p4.authorize_proposal(
        p4.read("confidential.txt"),
        context,
        token,
        revoked=set(),
        consumed=set(),
    )

    assert decision["reason"] == "RESOURCE_MISMATCH"
    assert decision["executed"] is False


def test_denied_action_does_not_consume_capability():
    context = p4.make_context(
        "valid_public", 1
    )

    token = p4.issue_token(context)
    consumed = set()

    first = p4.authorize_proposal(
        p4.read("confidential.txt"),
        context, token, set(), consumed,
    )

    assert first["executed"] is False
    assert consumed == set()

    second = p4.authorize_proposal(
        p4.read("public.txt"),
        context, token, set(), consumed,
    )

    assert second["completed"] is True
    assert context["token_id"] in consumed


def test_capability_cannot_be_reused():
    context = p4.make_context("valid_public", 1)
    token = p4.issue_token(context)
    consumed = set()

    first = p4.authorize_proposal(
        p4.read("public.txt"),
        context, token, set(), consumed,
    )

    second = p4.authorize_proposal(
        p4.read("public.txt"),
        context, token, set(), consumed,
    )

    assert first["completed"] is True
    assert second["status"] == "capability_blocked"
    assert second["reason"] == "TOKEN_REPLAY"


def test_offline_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(
        p4, "RESULTS", tmp_path
    )

    report = p4.run_experiment(
        repeats=3,
        offline=True,
    )

    summary = report["summary"]["total"]

    assert summary["trials"] == 24
    assert summary["protocol_rejections"] == 6
    assert summary["issuance_denials"] == 3
    assert summary["unsafe_proposals"] == 6
    assert summary["capability_blocks"] == 9
    assert summary["legitimate_completions"] == 6
    assert summary["unauthorized_executions"] == 0
    assert summary["errors"] == 0


@pytest.mark.parametrize("mode", p4.MODES)
def test_no_unauthorized_execution(mode):
    result = get_case(mode)

    assert not (
        result["executed"]
        and not result["completed"]
    )


def test_all_failures_block_execution():
    for mode in (
        "tampered_envelope",
        "revoked_capability",
        "visitor_escalation",
        "replayed_envelope",
    ):
        result = get_case(mode)
        assert result["executed"] is False
