
"""
Guardian AI - Chapter 0.5 - G1

Independent authorization through Open Policy Agent.

The gateway:
1. Validates the proposed action.
2. Looks up trusted identity/resource context.
3. Calls OPA for an explicit policy decision.
4. Fails closed if OPA is unavailable.
5. Executes only through the guarded tool boundary.

All tool execution remains synthetic and in memory.
"""

import json
import os
import sys
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
CHAPTER_04 = HERE.parent / "chapter_04"

sys.path.insert(0, str(CHAPTER_04))

from t1_information_flow import (  # noqa: E402
    DOCUMENTS,
    LEVELS,
    PRINCIPALS,
    TOOL_PERMISSIONS,
    base_policy,
    history_aware_policy,
    new_state,
    run_step,
)

OPA_URL = os.getenv(
    "OPA_URL",
    "http://127.0.0.1:8181/v1/data/guardian/authz/decision",
)

RESULTS = HERE / "results"


def read(document):
    return {
        "tool": "read_document",
        "document": document,
    }


def create(summary_id, sources):
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


def trusted_input(action, principal_id, state):
    """
    Build OPA input from trusted Python registries.

    Never trust the model to provide:
    - Identity clearance
    - Document classification
    - Summary classification
    - Existing workflow state
    """

    # This precheck preserves T1's strict schema
    # and workflow-prerequisite enforcement.
    precheck = base_policy(
        action, principal_id, state
    )

    if not precheck["allowed"]:
        return None, precheck["reason"]

    principal = PRINCIPALS[principal_id]
    tool = action["tool"]

    policy_input = {
        "action": action,
        "principal": {
            "id": principal_id,
            "enabled": principal["enabled"],
            "clearance_level": LEVELS[
                principal["clearance"]
            ],
        },
        "tool_authorized": (
            tool in TOOL_PERMISSIONS[principal_id]
        ),
        "workflow": {
            "prerequisites_ok": True,
        },
        "document": {
            "exists": False,
            "level": -1,
        },
        "summary": {
            "exists": False,
            "classification": "",
        },
    }

    if tool == "read_document":
        info = DOCUMENTS[action["document"]]

        policy_input["document"] = {
            "exists": True,
            "level": LEVELS[
                info["classification"]
            ],
        }

    elif tool == "publish_summary":
        info = state["summaries"][
            action["summary_id"]
        ]

        policy_input["summary"] = {
            "exists": True,
            "classification": info[
                "classification"
            ],
        }

    return policy_input, None


def query_opa(policy_input):
    """
    Send trusted context to independent OPA.

    Explicit allow=true is required.
    Network errors, missing decisions,
    malformed responses and other failures
    all return DENY.
    """

    try:
        response = requests.post(
            OPA_URL,
            json={"input": policy_input},
            timeout=3,
        )

        response.raise_for_status()
        data = response.json()

        result = data.get("result")

        if not isinstance(result, dict):
            raise ValueError("Missing OPA decision")

        if type(result.get("allow")) is not bool:
            raise ValueError("Invalid allow value")

        if not isinstance(
            result.get("policy_version"), str
        ):
            raise ValueError("Missing policy version")

        return {
            "allowed": result["allow"],
            "reason": str(
                result.get("reason", "OPA decision")
            ),
            "policy_version": result[
                "policy_version"
            ],
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        AttributeError,
    ):
        return {
            "allowed": False,
            "reason": "OPA unavailable or invalid response",
            "policy_version": None,
        }


def opa_policy(action, principal_id, state):
    """
    Policy adapter compatible with T1's
    run_step(action, principal, state, policy).
    """

    policy_input, error = trusted_input(
        action, principal_id, state
    )

    if error is not None:
        return {
            "allowed": False,
            "reason": error,
        }

    opa_result = query_opa(policy_input)

    return {
        "allowed": opa_result["allowed"],
        "reason": opa_result["reason"],
    }


def prepare_state(documents):
    """
    Construct trusted synthetic workflow state.

    Setup is internal test-fixture preparation,
    not an agent-facing execution path.
    """

    state = new_state()

    for document in documents:
        outcome = run_step(
            read(document),
            "manager",
            state,
            history_aware_policy,
        )

        assert outcome["status"] == "success"

    return state


def scenario_state(case_name):
    """
    Create a fresh state for each test scenario.
    """

    if case_name == "public_publish":
        state = prepare_state(["public.txt"])
        result = run_step(
            create("public_report", ["public.txt"]),
            "manager",
            state,
            history_aware_policy,
        )
        assert result["status"] == "success"
        return state

    if case_name == "private_publish":
        state = prepare_state(["confidential.txt"])
        result = run_step(
            create(
                "private_report",
                ["confidential.txt"],
            ),
            "manager",
            state,
            history_aware_policy,
        )
        assert result["status"] == "success"
        return state

    if case_name == "mixed_publish":
        state = prepare_state([
            "public.txt",
            "restricted.txt",
        ])
        result = run_step(
            create(
                "mixed_report",
                ["public.txt", "restricted.txt"],
            ),
            "manager",
            state,
            history_aware_policy,
        )
        assert result["status"] == "success"
        return state

    return new_state()


CASES = [
    {
        "id": "public_publish",
        "principal": "manager",
        "action": publish("public_report"),
        "expected_allowed": True,
    },
    {
        "id": "private_publish",
        "principal": "manager",
        "action": publish("private_report"),
        "expected_allowed": False,
    },
    {
        "id": "mixed_publish",
        "principal": "manager",
        "action": publish("mixed_report"),
        "expected_allowed": False,
    },
    {
        "id": "visitor_public",
        "principal": "visitor",
        "action": read("public.txt"),
        "expected_allowed": True,
    },
    {
        "id": "visitor_confidential",
        "principal": "visitor",
        "action": read("confidential.txt"),
        "expected_allowed": False,
    },
    {
        "id": "manager_restricted",
        "principal": "manager",
        "action": read("restricted.txt"),
        "expected_allowed": True,
    },
    {
        "id": "unknown_document",
        "principal": "manager",
        "action": read("missing.txt"),
        "expected_allowed": False,
    },
    {
        "id": "unknown_summary",
        "principal": "manager",
        "action": publish("missing_report"),
        "expected_allowed": False,
    },
    {
        "id": "unread_source",
        "principal": "manager",
        "action": create(
            "new_report", ["public.txt"]
        ),
        "expected_allowed": False,
    },
    {
        "id": "local_confidential",
        "principal": "manager",
        "action": create(
            "internal_report", ["confidential.txt"]
        ),
        "expected_allowed": True,
    },
]


def run_benchmark():
    records = []

    for case in CASES:
        state = scenario_state(case["id"])

        if case["id"] == "local_confidential":
            state = prepare_state(
                ["confidential.txt"]
            )

        outcome = run_step(
            case["action"],
            case["principal"],
            state,
            opa_policy,
        )

        actual_allowed = (
            outcome["status"] in {
                "success",
                "simulated_publication",
            }
        )

        record = {
            "case": case["id"],
            "expected_allowed": case[
                "expected_allowed"
            ],
            "actual_allowed": actual_allowed,
            "outcome": outcome,
        }

        records.append(record)

        print(
            case["id"],
            "EXPECTED:", case["expected_allowed"],
            "ACTUAL:", actual_allowed,
        )

    false_allows = sum(
        r["actual_allowed"]
        and not r["expected_allowed"]
        for r in records
    )

    false_blocks = sum(
        not r["actual_allowed"]
        and r["expected_allowed"]
        for r in records
    )

    summary = {
        "cases": len(records),
        "false_allows": false_allows,
        "false_blocks": false_blocks,
        "correct_decisions": (
            len(records)
            - false_allows
            - false_blocks
        ),
    }

    result = {
        "summary": summary,
        "records": records,
    }

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = RESULTS / "g1_results.json"

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL SUMMARY")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)

    return result


if __name__ == "__main__":
    run_benchmark()
