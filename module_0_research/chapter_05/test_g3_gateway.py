
from pathlib import Path

import pytest

import g3_gateway as g3
import g1_gateway as g1


def mock_decision(allowed, version="g3-v1"):
    return {
        "allowed": allowed,
        "policy_version": version,
        "reason_code": (
            "ALLOW" if allowed else "DENIED_BY_POLICY"
        ),
        "decision_id": None,
    }


def find_case(case_id):
    return next(
        c for c in g3.CASES
        if c["id"] == case_id
    )


def test_trusted_visitor_clearance():
    state = g1.new_state()

    data = g3.trusted_input(
        g1.read("confidential.txt"),
        "visitor",
        state,
    )

    assert data["principal"]["clearance_level"] == 0
    assert data["document"]["level"] == 1


def test_unauthorized_read_reaches_policy_client():
    state = g1.new_state()
    calls = []

    def policy_client(data):
        calls.append(data)
        return mock_decision(False)

    result = g3.authorize_and_execute(
        g1.read("confidential.txt"),
        "visitor",
        state,
        "g3-v1",
        policy_client=policy_client,
    )

    assert len(calls) == 1
    assert result["status"] == "blocked_opa"
    assert state["read_documents"] == set()


def test_unknown_document_reaches_opa():
    calls = []

    def policy_client(data):
        calls.append(data)
        return mock_decision(False)

    result = g3.authorize_and_execute(
        g1.read("missing.txt"),
        "manager",
        g1.new_state(),
        "g3-v1",
        policy_client=policy_client,
    )

    assert result["status"] == "blocked_opa"
    assert calls[0]["document"]["exists"] is False


def test_confidential_provenance_is_trusted():
    state = g1.scenario_state("private_publish")

    data = g3.trusted_input(
        g1.publish("private_report"),
        "manager",
        state,
    )

    assert data["summary"]["classification"] == "confidential"


def test_approval_not_in_model_action():
    state = g1.scenario_state("public_publish")

    data = g3.trusted_input(
        g1.publish("public_report"),
        "manager",
        state,
        approvals=set(),
    )

    assert data["approval"]["approved"] is False


def test_trusted_approval_record():
    state = g1.scenario_state("public_publish")

    data = g3.trusted_input(
        g1.publish("public_report"),
        "manager",
        state,
        approvals={"public_report"},
    )

    assert data["approval"]["approved"] is True


def test_model_cannot_supply_approval():
    state = g1.scenario_state("public_publish")

    result = g3.authorize_and_execute(
        {
            "tool": "publish_summary",
            "summary_id": "public_report",
            "approved": True,
        },
        "manager",
        state,
        "g3-v2",
        policy_client=lambda data: mock_decision(
            True, "g3-v2"
        ),
    )

    assert result["status"] == "rejected_schema"
    assert state["published"] == []


def test_authorized_action_executes():
    state = g1.scenario_state("public_publish")

    result = g3.authorize_and_execute(
        g1.publish("public_report"),
        "manager",
        state,
        "g3-v1",
        policy_client=lambda data: mock_decision(True),
    )

    assert result["executed"] is True
    assert state["published"] == ["public_report"]


def test_opa_denial_prevents_execution():
    state = g1.scenario_state("private_publish")

    result = g3.authorize_and_execute(
        g1.publish("private_report"),
        "manager",
        state,
        "g3-v1",
        policy_client=lambda data: mock_decision(False),
    )

    assert result["status"] == "blocked_opa"
    assert state["published"] == []


def test_wrong_policy_version_fails_closed():
    state = g1.scenario_state("public_publish")

    result = g3.authorize_and_execute(
        g1.publish("public_report"),
        "manager",
        state,
        "g3-v2",
        policy_client=lambda data: mock_decision(
            True, "g3-v1"
        ),
    )

    assert result["executed"] is False
    assert result["reason_code"] == "POLICY_VERSION_MISMATCH"
    assert state["published"] == []


@pytest.mark.parametrize(
    "decision",
    [
        {},
        None,
        {"allowed": "true", "policy_version": "g3-v1"},
        {
            "allowed": True,
            "policy_version": "g3-v1",
            "reason_code": "DENIED_BY_POLICY",
        },
        {
            "allowed": True,
            "policy_version": "g3-v1",
            "reason_code": "INVALID_REASON",
        },
    ],
)
def test_malformed_decision_fails_closed(decision):
    state = g1.new_state()

    result = g3.authorize_and_execute(
        g1.read("public.txt"),
        "visitor",
        state,
        "g3-v1",
        policy_client=lambda data: decision,
    )

    assert result["executed"] is False
    assert state["read_documents"] == set()


def test_policy_client_exception_fails_closed():
    def offline(data):
        raise ConnectionError("OPA offline")

    state = g1.new_state()

    result = g3.authorize_and_execute(
        g1.read("public.txt"),
        "visitor",
        state,
        "g3-v1",
        policy_client=offline,
    )

    assert result["executed"] is False


def test_single_policy_query():
    calls = []

    def counted(data):
        calls.append(data)
        return mock_decision(True)

    state = g1.new_state()

    result = g3.authorize_and_execute(
        g1.read("public.txt"),
        "visitor",
        state,
        "g3-v1",
        policy_client=counted,
    )

    assert result["executed"] is True
    assert len(calls) == 1


def test_audit_file_records_decision(tmp_path):
    path = tmp_path / "audit.jsonl"

    event = g3.audit_attempt(
        action=g1.read("public.txt"),
        principal_id="visitor",
        state=g1.new_state(),
        expected_version="g3-v1",
        policy_client=lambda data: mock_decision(True),
        audit_path=path,
    )

    assert path.exists()
    assert event["allowed"] is True
    assert event["executed"] is True

    import json

    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
    ]

    assert len(rows) == 1
    assert rows[0]["audit_id"] == event["audit_id"]


def test_rejected_schema_is_audited(tmp_path):
    path = tmp_path / "audit.jsonl"

    event = g3.audit_attempt(
        action={
            "tool": "read_document",
            "document": "public.txt",
            "admin": True,
        },
        principal_id="visitor",
        state=g1.new_state(),
        expected_version="g3-v1",
        audit_path=path,
    )

    assert event["status"] == "rejected_schema"
    assert event["decision_source"] == "gateway"
    assert event["executed"] is False
    assert path.exists()
