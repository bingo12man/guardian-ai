
from unittest.mock import Mock

import pytest
import requests

import g1_gateway as g1


def fake_opa_response(allowed, version="g1-v1"):
    return {
        "allowed": allowed,
        "reason": (
            "authorized"
            if allowed
            else "denied_by_policy"
        ),
        "policy_version": version,
    }


def test_manager_has_trusted_clearance():
    state = g1.new_state()

    policy_input, error = g1.trusted_input(
        g1.read("restricted.txt"),
        "manager",
        state,
    )

    assert error is None
    assert (
        policy_input["principal"][
            "clearance_level"
        ] == 2
    )


def test_visitor_cannot_read_confidential():
    policy_input, error = g1.trusted_input(
        g1.read("confidential.txt"),
        "visitor",
        g1.new_state(),
    )

    assert policy_input is None
    assert error == "Insufficient clearance"


def test_unknown_identity_denied():
    policy_input, error = g1.trusted_input(
        g1.read("public.txt"),
        "fake_admin",
        g1.new_state(),
    )

    assert policy_input is None
    assert error == "Unknown identity"


def test_private_classification_from_trusted_state():
    state = g1.scenario_state("private_publish")

    policy_input, error = g1.trusted_input(
        g1.publish("private_report"),
        "manager",
        state,
    )

    assert error is None
    assert (
        policy_input["summary"][
            "classification"
        ] == "confidential"
    )


def test_mixed_summary_stays_restricted():
    state = g1.scenario_state("mixed_publish")

    policy_input, error = g1.trusted_input(
        g1.publish("mixed_report"),
        "manager",
        state,
    )

    assert error is None
    assert (
        policy_input["summary"][
            "classification"
        ] == "restricted"
    )


def test_unread_source_denied():
    policy_input, error = g1.trusted_input(
        g1.create(
            "draft", ["public.txt"]
        ),
        "manager",
        g1.new_state(),
    )

    assert policy_input is None
    assert error == "Source was not read"


def test_fake_authorization_field_rejected():
    state = g1.new_state()

    action = {
        "tool": "read_document",
        "document": "public.txt",
        "authorized": True,
    }

    policy_input, error = g1.trusted_input(
        action, "manager", state
    )

    assert policy_input is None
    assert error == "Invalid action schema"


def test_opa_allows_explicit_boolean_true(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "result": {
            "allow": True,
            "reason": "authorized",
            "policy_version": "g1-v1",
        }
    }

    monkeypatch.setattr(
        g1.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = g1.query_opa({"action": {}})

    assert result["allowed"] is True


def test_opa_denies_explicit_false(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "result": {
            "allow": False,
            "reason": "denied_by_policy",
            "policy_version": "g1-v1",
        }
    }

    monkeypatch.setattr(
        g1.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = g1.query_opa({"action": {}})

    assert result["allowed"] is False


@pytest.mark.parametrize(
    "response_body",
    [
        {},
        {"result": {}},
        {"result": {"allow": "true",
                    "policy_version": "g1-v1"}},
        {"result": {"allow": 1,
                    "policy_version": "g1-v1"}},
        {"result": {"allow": True}},
        {"result": None},
    ],
)
def test_invalid_opa_response_fails_closed(
    monkeypatch, response_body
):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = response_body

    monkeypatch.setattr(
        g1.requests,
        "post",
        lambda *args, **kwargs: response,
    )

    result = g1.query_opa({})

    assert result["allowed"] is False


def test_connection_failure_fails_closed(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError(
            "OPA is offline"
        )

    monkeypatch.setattr(
        g1.requests,
        "post",
        raise_connection_error,
    )

    result = g1.query_opa({})

    assert result["allowed"] is False


def test_policy_does_not_execute_if_opa_down(monkeypatch):
    monkeypatch.setattr(
        g1,
        "query_opa",
        lambda policy_input: fake_opa_response(False),
    )

    state = g1.scenario_state("public_publish")

    outcome = g1.run_step(
        g1.publish("public_report"),
        "manager",
        state,
        g1.opa_policy,
    )

    assert outcome["status"] == "blocked"
    assert state["published"] == []


def test_authorized_publication_with_mocked_opa(monkeypatch):
    monkeypatch.setattr(
        g1,
        "query_opa",
        lambda policy_input: fake_opa_response(
            policy_input["summary"]["classification"]
            == "public"
        ),
    )

    state = g1.scenario_state("public_publish")

    outcome = g1.run_step(
        g1.publish("public_report"),
        "manager",
        state,
        g1.opa_policy,
    )

    assert (
        outcome["status"]
        == "simulated_publication"
    )
    assert state["published"] == ["public_report"]
