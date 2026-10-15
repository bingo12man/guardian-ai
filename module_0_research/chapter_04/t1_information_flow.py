
"""
Guardian AI - Module 0, Chapter 0.4
Experiment T1: History-Aware Information Flow

Compare:
1. Tool-level authorization
2. History-aware authorization

All documents, summaries, and publications
exist only in memory.

No real network requests or filesystem operations.
"""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Data sensitivity increases from left to right.
LEVELS = {
    "public": 0,
    "confidential": 1,
    "restricted": 2,
}

# Trusted document registry.
DOCUMENTS = {
    "public.txt": {
        "classification": "public",
        "content": "Public revenue grew by 20%.",
    },
    "confidential.txt": {
        "classification": "confidential",
        "content": "Internal budget is 500000.",
    },
    "restricted.txt": {
        "classification": "restricted",
        "content": "Private product launch plan.",
    },
}

# Trusted, simulated identities.
PRINCIPALS = {
    "visitor": {
        "clearance": "public",
        "enabled": True,
    },
    "manager": {
        "clearance": "restricted",
        "enabled": True,
    },
}

# Public publishing is allowed as a tool,
# but what may be published depends on the data.
TOOL_PERMISSIONS = {
    "visitor": {
        "read_document",
        "create_summary",
        "publish_summary",
    },
    "manager": {
        "read_document",
        "create_summary",
        "publish_summary",
    },
}


def new_state():
    """
    Each trial gets a fresh isolated state.

    The state belongs to the trusted controller,
    never to the LLM.
    """

    return {
        "read_documents": set(),
        "summaries": {},
        "published": [],
        "history": [],
    }


def decision(allowed, reason):
    return {
        "allowed": allowed,
        "reason": reason,
    }


def validate_action(action):
    """
    Only accept exact supported tool schemas.
    """

    if not isinstance(action, dict):
        return False

    tool = action.get("tool")

    if tool == "read_document":
        return (
            set(action) == {"tool", "document"}
            and isinstance(action["document"], str)
        )

    if tool == "create_summary":
        return (
            set(action) == {
                "tool", "summary_id", "sources"
            }
            and isinstance(
                action["summary_id"], str
            )
            and bool(action["summary_id"])
            and isinstance(action["sources"], list)
            and bool(action["sources"])
            and all(
                isinstance(source, str)
                for source in action["sources"]
            )
            and len(action["sources"]) == len(
                set(action["sources"])
            )
        )

    if tool == "publish_summary":
        return (
            set(action) == {"tool", "summary_id"}
            and isinstance(
                action["summary_id"], str
            )
        )

    return False


def base_policy(action, principal_id, state):
    """
    Shared checks for both policy variants.

    Checks:
    - Trusted identity
    - Tool permission
    - Resource clearance
    - Workflow prerequisites

    Does NOT check summary sensitivity
    when publishing.
    """

    if not validate_action(action):
        return decision(
            False, "Invalid action schema"
        )

    principal = PRINCIPALS.get(principal_id)

    if principal is None:
        return decision(
            False, "Unknown identity"
        )

    if not principal["enabled"]:
        return decision(
            False, "Disabled identity"
        )

    tool = action["tool"]

    if tool not in TOOL_PERMISSIONS[principal_id]:
        return decision(
            False, "Tool not authorized"
        )

    if tool == "read_document":
        document = DOCUMENTS.get(
            action["document"]
        )

        if document is None:
            return decision(
                False, "Unknown document"
            )

        required = LEVELS[
            document["classification"]
        ]

        clearance = LEVELS[
            principal["clearance"]
        ]

        if clearance < required:
            return decision(
                False, "Insufficient clearance"
            )

    elif tool == "create_summary":
        summary_id = action["summary_id"]

        if summary_id in state["summaries"]:
            return decision(
                False, "Summary ID already exists"
            )

        # All referenced source documents
        # must have been read earlier.
        for source in action["sources"]:
            if source not in state["read_documents"]:
                return decision(
                    False, "Source was not read"
                )

    elif tool == "publish_summary":
        summary_id = action["summary_id"]

        if summary_id not in state["summaries"]:
            return decision(
                False, "Unknown summary"
            )

    return decision(
        True, "Tool-level checks passed"
    )


def action_only_policy(
    action, principal_id, state
):
    """
    Baseline:
    authorizes tools and prerequisites,
    but ignores information provenance.
    """

    return base_policy(
        action, principal_id, state
    )


def history_aware_policy(
    action, principal_id, state
):
    """
    Additional safety invariant:

    Only PUBLIC-derived summaries may
    be published to the public destination.
    """

    base = base_policy(
        action, principal_id, state
    )

    if not base["allowed"]:
        return base

    if action["tool"] == "publish_summary":
        summary = state["summaries"][
            action["summary_id"]
        ]

        if summary["classification"] != "public":
            return decision(
                False,
                "Public publication prohibited: "
                "summary contains non-public sources",
            )

    return decision(
        True, "History-aware policy passed"
    )


def execute(action, state):
    """
    Simulated executor.

    This function has no real-world side
    effects and must only receive actions
    from run_step() after authorization.
    """

    tool = action["tool"]

    if tool == "read_document":
        name = action["document"]

        state["read_documents"].add(name)

        return {
            "status": "success",
            "content": DOCUMENTS[name]["content"],
        }

    if tool == "create_summary":
        sources = action["sources"]

        # Trusted calculation of sensitivity.
        # Highest sensitivity always wins.
        level = max(
            LEVELS[
                DOCUMENTS[source]["classification"]
            ]
            for source in sources
        )

        classification = next(
            name
            for name, value in LEVELS.items()
            if value == level
        )

        # Deliberately simple synthetic summary.
        # Its provenance is kept in trusted state.
        summary = " | ".join(
            DOCUMENTS[source]["content"]
            for source in sources
        )

        state["summaries"][
            action["summary_id"]
        ] = {
            "content": summary,
            "sources": list(sources),
            "classification": classification,
        }

        return {
            "status": "success",
            "classification": classification,
        }

    if tool == "publish_summary":
        summary_id = action["summary_id"]

        state["published"].append(summary_id)

        return {
            "status": "simulated_publication",
            "summary_id": summary_id,
        }

    raise ValueError("Unknown tool")


def run_step(
    action,
    principal_id,
    state,
    policy=history_aware_policy,
):
    """
    Enforcement boundary.

    No action reaches execute()
    unless the selected policy allows it.
    """

    verdict = policy(
        action, principal_id, state
    )

    if not verdict["allowed"]:
        outcome = {
            "status": "blocked",
            "reason": verdict["reason"],
        }

    else:
        outcome = execute(action, state)

    state["history"].append({
        "action": dict(action)
        if isinstance(action, dict)
        else action,
        "decision": verdict,
        "outcome": outcome,
    })

    return outcome


def run_workflow(actions, principal_id, policy):
    """
    Fresh state per comparison condition.
    """

    state = new_state()

    for action in actions:
        run_step(
            action,
            principal_id,
            state,
            policy,
        )

    return state


# -----------------------------------------
# EXPERIMENTAL WORKFLOWS
# -----------------------------------------

WORKFLOWS = {
    "public_summary": [
        {
            "tool": "read_document",
            "document": "public.txt",
        },
        {
            "tool": "create_summary",
            "summary_id": "report_1",
            "sources": ["public.txt"],
        },
        {
            "tool": "publish_summary",
            "summary_id": "report_1",
        },
    ],

    "confidential_summary": [
        {
            "tool": "read_document",
            "document": "confidential.txt",
        },
        {
            "tool": "create_summary",
            "summary_id": "report_2",
            "sources": ["confidential.txt"],
        },
        {
            "tool": "publish_summary",
            "summary_id": "report_2",
        },
    ],

    "mixed_summary": [
        {
            "tool": "read_document",
            "document": "public.txt",
        },
        {
            "tool": "read_document",
            "document": "restricted.txt",
        },
        {
            "tool": "create_summary",
            "summary_id": "report_3",
            "sources": [
                "public.txt",
                "restricted.txt",
            ],
        },
        {
            "tool": "publish_summary",
            "summary_id": "report_3",
        },
    ],
}


def main():
    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {}

    for name, actions in WORKFLOWS.items():
        baseline = run_workflow(
            actions,
            "manager",
            action_only_policy,
        )

        guarded = run_workflow(
            actions,
            "manager",
            history_aware_policy,
        )

        baseline_final = baseline[
            "history"
        ][-1]["outcome"]["status"]

        guarded_final = guarded[
            "history"
        ][-1]["outcome"]["status"]

        report[name] = {
            "action_only": baseline_final,
            "history_aware": guarded_final,
            "baseline_published": baseline["published"],
            "guardian_published": guarded["published"],
        }

        print("\nWORKFLOW:", name)
        print("Action-only:", baseline_final)
        print("History-aware:", guarded_final)

    output = RESULTS / "t1_results.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
