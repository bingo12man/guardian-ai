
import pytest

from t1_information_flow import (
    DOCUMENTS,
    WORKFLOWS,
    action_only_policy,
    history_aware_policy,
    new_state,
    run_step,
    run_workflow,
)


def read(name):
    return {
        "tool": "read_document",
        "document": name,
    }


def summarize(summary_id, sources):
    return {
        "tool": "create_summary",
        "summary_id": summary_id,
        "sources": sources,
    }


def publish(summary_id):
    return {
        "tool": "publish_summary",
        "summary_id": summary_id,
    }


# Test 1: Public workflow succeeds.
def test_public_workflow_allowed():
    state = run_workflow(
        WORKFLOWS["public_summary"],
        "manager",
        history_aware_policy,
    )

    assert state["published"] == ["report_1"]


# Test 2: Confidential publication is blocked.
def test_confidential_publication_blocked():
    state = run_workflow(
        WORKFLOWS["confidential_summary"],
        "manager",
        history_aware_policy,
    )

    assert state["published"] == []
    assert (
        state["history"][-1]["outcome"]["status"]
        == "blocked"
    )


# Test 3: Restricted data taints mixed summaries.
def test_mixed_summary_is_restricted():
    state = run_workflow(
        WORKFLOWS["mixed_summary"],
        "manager",
        history_aware_policy,
    )

    assert (
        state["summaries"]["report_3"][
            "classification"
        ] == "restricted"
    )

    assert state["published"] == []


# Test 4: Baseline misses the information-flow risk.
def test_action_only_misses_confidential_flow():
    state = run_workflow(
        WORKFLOWS["confidential_summary"],
        "manager",
        action_only_policy,
    )

    assert state["published"] == ["report_2"]


# Test 5: Visitor cannot read confidential data.
def test_visitor_confidential_read_blocked():
    state = new_state()

    result = run_step(
        read("confidential.txt"),
        "visitor",
        state,
    )

    assert result["status"] == "blocked"
    assert state["read_documents"] == set()


# Test 6: Manager may read confidential data locally.
def test_manager_local_read_allowed():
    state = new_state()

    result = run_step(
        read("confidential.txt"),
        "manager",
        state,
    )

    assert result["status"] == "success"
    assert "confidential.txt" in state["read_documents"]


# Test 7: Cannot summarize unread documents.
def test_unread_source_blocked():
    state = new_state()

    result = run_step(
        summarize(
            "draft",
            ["confidential.txt"],
        ),
        "manager",
        state,
    )

    assert result["status"] == "blocked"


# Test 8: Unknown documents are denied.
def test_unknown_document_blocked():
    state = new_state()

    result = run_step(
        read("unknown.txt"),
        "manager",
        state,
    )

    assert result["status"] == "blocked"


# Test 9: Unknown summaries cannot be published.
def test_unknown_summary_blocked():
    state = new_state()

    result = run_step(
        publish("nonexistent"),
        "manager",
        state,
    )

    assert result["status"] == "blocked"


# Test 10: Model cannot supply its own data label.
def test_untrusted_classification_field_blocked():
    state = new_state()

    result = run_step(
        {
            "tool": "read_document",
            "document": "confidential.txt",
            "classification": "public",
        },
        "manager",
        state,
    )

    assert result["status"] == "blocked"


# Test 11: Duplicate summary IDs are rejected.
def test_duplicate_summary_id_blocked():
    state = new_state()

    run_step(
        read("public.txt"),
        "manager",
        state,
    )

    first = run_step(
        summarize("draft", ["public.txt"]),
        "manager",
        state,
    )

    second = run_step(
        summarize("draft", ["public.txt"]),
        "manager",
        state,
    )

    assert first["status"] == "success"
    assert second["status"] == "blocked"


# Test 12: All tested classifications preserve
# the trusted registry's data sensitivity.
@pytest.mark.parametrize(
    "document,expected",
    [
        ("public.txt", "public"),
        ("confidential.txt", "confidential"),
        ("restricted.txt", "restricted"),
    ],
)
def test_summary_classification(document, expected):
    state = new_state()

    run_step(
        read(document),
        "manager",
        state,
    )

    result = run_step(
        summarize("draft", [document]),
        "manager",
        state,
    )

    assert result["classification"] == expected


# Test 13: Malformed action cannot execute.
@pytest.mark.parametrize(
    "action",
    [
        None,
        {},
        {"tool": "run_shell"},
        {"tool": "publish_summary", "summary_id": 123},
        {"tool": "create_summary",
         "summary_id": "x",
         "sources": []},
    ],
)
def test_invalid_action_blocked(action):
    state = new_state()

    result = run_step(
        action,
        "manager",
        state,
    )

    assert result["status"] == "blocked"


# Test 14: Policy decisions don't mutate documents.
def test_source_documents_unchanged():
    snapshot = {
        key: dict(value)
        for key, value in DOCUMENTS.items()
    }

    run_workflow(
        WORKFLOWS["confidential_summary"],
        "manager",
        history_aware_policy,
    )

    assert DOCUMENTS == snapshot
