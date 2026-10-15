
"""
Guardian AI - Chapter 0.5 - G3

Independent OPA authorization and audit logging.

Unlike G2, Python only:
- validates the structure of tool requests
- constructs trusted authorization attributes
- obtains one decision from OPA
- enforces the decision before execution
- records the result

Synthetic in-memory tools only.
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

import g1_gateway as g1

from t1_information_flow import (
    DOCUMENTS,
    LEVELS,
    PRINCIPALS,
    TOOL_PERMISSIONS,
    execute,
    validate_action,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

OPA_URL = os.getenv(
    "G3_OPA_URL",
    "http://127.0.0.1:8181/v1/data/guardian/g3/decision",
)


def trusted_input(action, principal_id, state, approvals=None):
    """
    Constructs facts; makes NO authorization decision.

    Identity, classification, state, and approval
    come from trusted Python-controlled sources.

    approvals is a trusted set of summary IDs.
    Never use an LLM-provided approvals value.
    """

    if not validate_action(action):
        return None

    if approvals is None:
        approvals = set()

    principal = PRINCIPALS.get(principal_id)

    principal_exists = principal is not None

    clearance = (
        LEVELS[principal["clearance"]]
        if principal_exists else -1
    )

    enabled = (
        principal["enabled"]
        if principal_exists else False
    )

    tool = action["tool"]

    tool_authorized = (
        tool in TOOL_PERMISSIONS.get(
            principal_id, set()
        )
    )

    document_info = None
    summary_info = None

    sources_read = False
    summary_id_available = False

    if tool == "read_document":
        document_info = DOCUMENTS.get(
            action["document"]
        )

    if tool == "create_summary":
        sources_read = all(
            source in state["read_documents"]
            and source in DOCUMENTS
            for source in action["sources"]
        )

        summary_id_available = (
            action["summary_id"]
            not in state["summaries"]
        )

    if tool == "publish_summary":
        summary_info = state["summaries"].get(
            action["summary_id"]
        )

    document_level = (
        LEVELS[document_info["classification"]]
        if document_info is not None else -1
    )

    summary_classification = (
        summary_info["classification"]
        if summary_info is not None else ""
    )

    approved = (
        tool == "publish_summary"
        and action["summary_id"] in approvals
    )

    return {
        "action": action,
        "principal": {
            "id": principal_id,
            "exists": principal_exists,
            "enabled": enabled,
            "clearance_level": clearance,
        },
        "tool_authorized": tool_authorized,
        "document": {
            "exists": document_info is not None,
            "level": document_level,
        },
        "workflow": {
            "sources_read": sources_read,
            "summary_id_available": summary_id_available,
        },
        "summary": {
            "exists": summary_info is not None,
            "classification": summary_classification,
        },
        "approval": {
            "approved": approved,
        },
    }


def query_opa(policy_input):
    """
    Query the separate OPA service.

    Transport failure and malformed response deny.
    """

    try:
        response = requests.post(
            OPA_URL,
            json={"input": policy_input},
            timeout=3,
        )

        response.raise_for_status()
        payload = response.json()

        result = payload.get("result")

        if not isinstance(result, dict):
            raise ValueError("Missing decision")

        if type(result.get("allow")) is not bool:
            raise ValueError("Invalid allow type")

        version = result.get("policy_version")
        reason = result.get("reason_code")

        if not isinstance(version, str) or not version:
            raise ValueError("Missing policy version")

        if reason not in {
            "ALLOW", "DENIED_BY_POLICY"
        }:
            raise ValueError("Invalid reason code")

        return {
            "allowed": result["allow"],
            "policy_version": version,
            "reason_code": reason,
            "decision_id": payload.get("decision_id"),
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        AttributeError,
    ):
        return {
            "allowed": False,
            "policy_version": None,
            "reason_code": "OPA_UNAVAILABLE_OR_INVALID",
            "decision_id": None,
        }


def authorize_and_execute(
    action,
    principal_id,
    state,
    expected_version,
    approvals=None,
    policy_client=None,
):
    """
    Single-OPA-decision enforcement.

    Only the expected policy version is accepted.
    A failed or malformed policy decision cannot
    cause execution.

    This prototype assumes the trusted controller
    owns the state and invokes this gateway without
    concurrent unauthorized state changes.
    """

    if not validate_action(action):
        return {
            "status": "rejected_schema",
            "allowed": False,
            "executed": False,
            "reason_code": "INVALID_SCHEMA",
            "decision_source": "gateway",
            "policy_version": None,
            "decision_id": None,
        }

    policy_input = trusted_input(
        action, principal_id, state, approvals
    )

    client = policy_client or query_opa

    try:
        verdict = client(policy_input)
    except Exception:
        verdict = None

    if not isinstance(verdict, dict):
        verdict = {}

    version = verdict.get("policy_version")
    allow = verdict.get("allowed") is True
    reason = verdict.get(
        "reason_code", "INVALID_DECISION"
    )

    if version != expected_version:
        allow = False
        reason = "POLICY_VERSION_MISMATCH"

    if reason not in {
        "ALLOW",
        "DENIED_BY_POLICY",
        "OPA_UNAVAILABLE_OR_INVALID",
        "INVALID_DECISION",
        "POLICY_VERSION_MISMATCH",
    }:
        allow = False
        reason = "INVALID_DECISION"

    if allow and reason != "ALLOW":
        allow = False
        reason = "INVALID_DECISION"

    if not allow and reason == "ALLOW":
        reason = "INVALID_DECISION"

    if not allow:
        return {
            "status": "blocked_opa",
            "allowed": False,
            "executed": False,
            "reason_code": reason,
            "decision_source": "opa",
            "policy_version": version,
            "decision_id": verdict.get("decision_id"),
        }

    outcome = execute(action, state)

    return {
        "status": outcome["status"],
        "allowed": True,
        "executed": True,
        "reason_code": "ALLOW",
        "decision_source": "opa",
        "policy_version": version,
        "decision_id": verdict.get("decision_id"),
    }


def audit_attempt(
    action,
    principal_id,
    state,
    expected_version,
    approvals=None,
    policy_client=None,
    audit_path=None,
):
    """
    Records one authorization attempt.

    The audit file contains synthetic action metadata.
    Production systems need access controls,
    redaction, and tamper-evident storage.
    """

    event_id = str(uuid.uuid4())

    result = authorize_and_execute(
        action=action,
        principal_id=principal_id,
        state=state,
        expected_version=expected_version,
        approvals=approvals,
        policy_client=policy_client,
    )

    event = {
        "audit_id": event_id,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "principal": principal_id,
        "action": action,
        "expected_policy_version": expected_version,
        "decision_source": result["decision_source"],
        "policy_version": result["policy_version"],
        "decision_id": result["decision_id"],
        "reason_code": result["reason_code"],
        "allowed": result["allowed"],
        "executed": result["executed"],
        "status": result["status"],
    }

    if audit_path is not None:
        path = Path(audit_path)
        path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with path.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(event) + "\n"
            )

    return event


# ----------------------------------------
# FIXED G3 BENCHMARK
# ----------------------------------------

CASES = [
    {
        "id": "public_publish",
        "principal": "manager",
        "state": "public_publish",
        "action": g1.publish("public_report"),
        "approved": False,
        "v1": True,
        "v2": False,
    },
    {
        "id": "approved_public_publish",
        "principal": "manager",
        "state": "public_publish",
        "action": g1.publish("public_report"),
        "approved": True,
        "v1": True,
        "v2": True,
    },
    {
        "id": "confidential_publish",
        "principal": "manager",
        "state": "private_publish",
        "action": g1.publish("private_report"),
        "approved": True,
        "v1": False,
        "v2": False,
    },
    {
        "id": "restricted_publish",
        "principal": "manager",
        "state": "mixed_publish",
        "action": g1.publish("mixed_report"),
        "approved": True,
        "v1": False,
        "v2": False,
    },
    {
        "id": "visitor_public",
        "principal": "visitor",
        "state": None,
        "action": g1.read("public.txt"),
        "approved": False,
        "v1": True,
        "v2": True,
    },
    {
        "id": "visitor_confidential",
        "principal": "visitor",
        "state": None,
        "action": g1.read("confidential.txt"),
        "approved": False,
        "v1": False,
        "v2": False,
    },
    {
        "id": "manager_restricted",
        "principal": "manager",
        "state": None,
        "action": g1.read("restricted.txt"),
        "approved": False,
        "v1": True,
        "v2": True,
    },
    {
        "id": "unknown_document",
        "principal": "manager",
        "state": None,
        "action": g1.read("missing.txt"),
        "approved": False,
        "v1": False,
        "v2": False,
    },
    {
        "id": "unread_source",
        "principal": "manager",
        "state": None,
        "action": g1.create(
            "draft", ["public.txt"]
        ),
        "approved": False,
        "v1": False,
        "v2": False,
    },
    {
        "id": "local_confidential",
        "principal": "manager",
        "state": "local_confidential",
        "action": g1.create(
            "internal_report",
            ["confidential.txt"],
        ),
        "approved": False,
        "v1": True,
        "v2": True,
    },
]


def prepare_case(case):
    state_name = case["state"]

    if state_name == "local_confidential":
        return g1.prepare_state(
            ["confidential.txt"]
        )

    if state_name is None:
        return g1.new_state()

    return g1.scenario_state(state_name)


def run_benchmark(version):
    if version not in {"g3-v1", "g3-v2"}:
        raise ValueError("Invalid expected version")

    RESULTS.mkdir(
        parents=True, exist_ok=True
    )

    audit_path = RESULTS / f"g3_audit_{version}.jsonl"

    # Fresh benchmark log on every run.
    audit_path.write_text("", encoding="utf-8")

    records = []

    for case in CASES:
        state = prepare_case(case)

        approvals = set()

        if case["approved"]:
            if "summary_id" in case["action"]:
                approvals.add(
                    case["action"]["summary_id"]
                )

        event = audit_attempt(
            action=case["action"],
            principal_id=case["principal"],
            state=state,
            expected_version=version,
            approvals=approvals,
            audit_path=audit_path,
        )

        expected = case[
            "v1" if version == "g3-v1" else "v2"
        ]

        records.append({
            "case": case["id"],
            "expected_allowed": expected,
            "event": event,
        })

        print(
            case["id"],
            "| expected:", expected,
            "| actual:", event["allowed"],
            "| source:", event["decision_source"],
            "| reason:", event["reason_code"],
        )

    false_allows = sum(
        r["event"]["allowed"]
        and not r["expected_allowed"]
        for r in records
    )

    false_blocks = sum(
        not r["event"]["allowed"]
        and r["expected_allowed"]
        for r in records
    )

    summary = {
        "version": version,
        "cases": len(records),
        "opa_decisions": sum(
            r["event"]["decision_source"] == "opa"
            for r in records
        ),
        "opa_denials": sum(
            r["event"]["status"] == "blocked_opa"
            for r in records
        ),
        "gateway_schema_rejections": sum(
            r["event"]["status"] == "rejected_schema"
            for r in records
        ),
        "false_allows": false_allows,
        "false_blocks": false_blocks,
        "correct_decisions": (
            len(records) - false_allows - false_blocks
        ),
    }

    output = RESULTS / f"g3_results_{version}.json"

    output.write_text(
        json.dumps({
            "summary": summary,
            "records": records,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL SUMMARY")
    print(json.dumps(summary, indent=2))
    print("Saved:", output)
    print("Audit:", audit_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        required=True,
        choices=["g3-v1", "g3-v2"],
    )

    args = parser.parse_args()
    run_benchmark(args.version)
