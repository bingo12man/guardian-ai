
import pytest

import g2_agent as g2


def raw_action(
    tool,
    document="",
    summary_id="",
    sources=None,
    reason="Synthetic test",
):
    if sources is None:
        sources = []

    return {
        "tool": tool,
        "document": document,
        "summary_id": summary_id,
        "sources": sources,
        "reason": reason,
    }


def allow_policy(policy_input):
    return {
        "allowed": True,
        "reason": "authorized",
        "policy_version": "g1-v1",
    }


def deny_policy(policy_input):
    return {
        "allowed": False,
        "reason": "denied_by_policy",
        "policy_version": "g1-v1",
    }


def realistic_policy(policy_input):
    """Synthetic approximation for unit tests."""
    tool = policy_input["action"]["tool"]

    if tool == "publish_summary":
        allowed = (
            policy_input["summary"]["classification"]
            == "public"
        )

    else:
        allowed = True

    return {
        "allowed": allowed,
        "reason": (
            "authorized" if allowed
            else "denied_by_policy"
        ),
        "policy_version": "g1-v1",
    }


def find_case(case_id):
    return next(
        case for case in g2.SCENARIOS
        if case["id"] == case_id
    )


# ---------------------------------------
# 1. Parse valid actions
# ---------------------------------------

def test_parse_read():
    parsed = g2.parse_proposal(
        raw_action(
            "read_document",
            document="public.txt",
        )
    )
    assert parsed == g2.g1.read("public.txt")


def test_parse_create_summary():
    parsed = g2.parse_proposal(
        raw_action(
            "create_summary",
            summary_id="report_1",
            sources=["public.txt"],
        )
    )
    assert parsed == g2.g1.create(
        "report_1",
        ["public.txt"],
    )


def test_parse_publication():
    parsed = g2.parse_proposal(
        raw_action(
            "publish_summary",
            summary_id="public_report",
        )
    )
    assert parsed == g2.g1.publish(
        "public_report"
    )


def test_parse_finish():
    assert g2.parse_proposal(
        raw_action("finish")
    ) == {"tool": "finish"}


# ---------------------------------------
# 2. Reject malformed actions
# ---------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        raw_action(
            "read_document",
            document=123,
        ),
        raw_action(
            "create_summary",
            summary_id="",
            sources=["public.txt"],
        ),
        raw_action(
            "publish_summary",
            document="public.txt",
            summary_id="public_report",
        ),
        {
            **raw_action(
                "read_document",
                document="confidential.txt",
            ),
            "authorized": True,
        },
        {
            **raw_action("finish"),
            "principal": "admin",
        },
    ],
)
def test_bad_proposal_rejected(raw):
    assert g2.parse_proposal(raw) is None


# ---------------------------------------
# 3. Authorized execution
# ---------------------------------------

def test_public_publication_executes():
    state = g2.g1.scenario_state(
        "public_publish"
    )

    outcome = g2.authorize_and_execute(
        g2.g1.publish("public_report"),
        "manager",
        state,
        policy_client=realistic_policy,
    )

    assert outcome["executed"] is True
    assert outcome["status"] == "simulated_publication"
    assert state["published"] == ["public_report"]


# ---------------------------------------
# 4. OPA denies confidential publication
# ---------------------------------------

def test_private_publication_denied():
    state = g2.g1.scenario_state(
        "private_publish"
    )

    outcome = g2.authorize_and_execute(
        g2.g1.publish("private_report"),
        "manager",
        state,
        policy_client=realistic_policy,
    )

    assert outcome["executed"] is False
    assert outcome["status"] == "blocked_opa"
    assert state["published"] == []


# ---------------------------------------
# 5. Fake admin cannot change identity
# ---------------------------------------

def test_visitor_remains_visitor():
    state = g2.g1.new_state()

    outcome = g2.authorize_and_execute(
        g2.g1.read("confidential.txt"),
        "visitor",
        state,
        policy_client=allow_policy,
    )

    assert outcome["executed"] is False
    assert outcome["status"] == "blocked_precheck"
    assert state["read_documents"] == set()


# ---------------------------------------
# 6. Unknown tool is never executed
# ---------------------------------------

def test_invalid_tool_does_not_call_opa():
    state = g2.g1.new_state()
    calls = []

    def track_policy(policy_input):
        calls.append(policy_input)
        return allow_policy(policy_input)

    outcome = g2.authorize_and_execute(
        None,
        "manager",
        state,
        policy_client=track_policy,
    )

    assert outcome["status"] == "rejected_schema"
    assert calls == []


# ---------------------------------------
# 7. Explicit OPA denial prevents execution
# ---------------------------------------

def test_explicit_policy_denial():
    state = g2.g1.new_state()

    outcome = g2.authorize_and_execute(
        g2.g1.read("public.txt"),
        "manager",
        state,
        policy_client=deny_policy,
    )

    assert outcome["executed"] is False
    assert state["read_documents"] == set()


# ---------------------------------------
# 8. Missing policy version fails closed
# ---------------------------------------

def test_missing_policy_version_denied():
    state = g2.g1.new_state()

    outcome = g2.authorize_and_execute(
        g2.g1.read("public.txt"),
        "manager",
        state,
        policy_client=lambda _: {
            "allowed": True,
            "reason": "authorized",
        },
    )

    assert outcome["executed"] is False
    assert state["read_documents"] == set()


# ---------------------------------------
# 9. Truthy string is NOT boolean true
# ---------------------------------------

def test_string_allow_rejected():
    state = g2.g1.new_state()

    outcome = g2.authorize_and_execute(
        g2.g1.read("public.txt"),
        "manager",
        state,
        policy_client=lambda _: {
            "allowed": "true",
            "policy_version": "g1-v1",
        },
    )

    assert outcome["executed"] is False


# ---------------------------------------
# 10. One decision per tool invocation
# ---------------------------------------

def test_single_opa_decision():
    state = g2.g1.new_state()
    calls = []

    def tracked_policy(policy_input):
        calls.append(policy_input)
        return allow_policy(policy_input)

    outcome = g2.authorize_and_execute(
        g2.g1.read("public.txt"),
        "manager",
        state,
        policy_client=tracked_policy,
    )

    assert outcome["executed"] is True
    assert len(calls) == 1


# ---------------------------------------
# 11. Finish never executes a tool
# ---------------------------------------

def test_finish_no_execution():
    state = g2.g1.new_state()

    outcome = g2.authorize_and_execute(
        {"tool": "finish"},
        "manager",
        state,
        policy_client=allow_policy,
    )

    assert outcome["status"] == "finished"
    assert outcome["executed"] is False


# ---------------------------------------
# 12. Model proposal + policy denial
# ---------------------------------------

def test_model_trial_private_block():
    scenario = find_case("manager_private")

    result = g2.run_trial(
        scenario,
        proposer=lambda case, state: raw_action(
            "publish_summary",
            summary_id="private_report",
        ),
        policy_client=realistic_policy,
    )

    assert result["valid_action"] is True
    assert result["requested_action_match"] is True
    assert result["outcome"]["executed"] is False
    assert result["published"] == []


# ---------------------------------------
# 13. Model proposal + authorized success
# ---------------------------------------

def test_model_trial_public_success():
    scenario = find_case("manager_public")

    result = g2.run_trial(
        scenario,
        proposer=lambda case, state: raw_action(
            "publish_summary",
            summary_id="public_report",
        ),
        policy_client=realistic_policy,
    )

    assert result["valid_action"] is True
    assert result["executed_requested_action"] is True
    assert result["published"] == ["public_report"]


# ---------------------------------------
# 14. Model error cannot execute tools
# ---------------------------------------

def test_model_failure_fails_closed():
    scenario = find_case("manager_public")

    def failing_model(*args):
        raise ValueError("Invalid model response")

    result = g2.run_trial(
        scenario,
        proposer=failing_model,
        policy_client=allow_policy,
    )

    assert result["valid_action"] is False
    assert result["outcome"]["executed"] is False
    assert result["model_error"] is not None


# ---------------------------------------
# 15. Policy decisions and audit summary
# ---------------------------------------

def test_summary_counts():
    public = g2.run_trial(
        find_case("manager_public"),
        proposer=lambda *args: raw_action(
            "publish_summary",
            summary_id="public_report",
        ),
        policy_client=realistic_policy,
    )

    private = g2.run_trial(
        find_case("manager_private"),
        proposer=lambda *args: raw_action(
            "publish_summary",
            summary_id="private_report",
        ),
        policy_client=realistic_policy,
    )

    summary = g2.summarize([public, private])

    assert summary["trials"] == 2
    assert summary["valid_proposals"] == 2
    assert summary["tool_executions"] == 1
    assert summary["policy_denials"] == 1
    assert summary["simulated_private_publications"] == 0
    assert summary["opa_versions_observed"] == ["g1-v1"]
