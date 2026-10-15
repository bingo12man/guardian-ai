
"""
Guardian AI - Chapter 0.8 - P3

Scoped, signed delegation capabilities.

Compare:
    bearer_baseline (deliberately vulnerable)
    capability_guard (strict authorization)

Synthetic document operations only.
No real document contents or network actions.
"""

import argparse
import copy
import hashlib
import hmac
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

NOW = 1050

# Testing key only. Never hardcode production secrets.
ISSUER_KEY = b"guardian-p3-test-issuer-key"

DOCUMENTS = {
    "public.txt": "public",
    "confidential.txt": "confidential",
    "internal.txt": "confidential",
}

CLEARANCES = {
    "visitor": "public",
    "manager": "confidential",
}

LEVELS = {
    "public": 0,
    "confidential": 1,
}

ALLOWED_WORKERS = {"worker", "backup_worker"}

SCENARIOS = (
    "valid_delegation",
    "other_resource",
    "wrong_worker",
    "wrong_task",
    "expired_token",
    "revoked_token",
    "tampered_scope",
    "replayed_token",
    "visitor_escalation",
    "valid_public",
)


# ------------------------------------
# 1. TRUSTED ISSUER
# ------------------------------------

def authorized(requester, resource):
    clearance = CLEARANCES.get(requester)
    classification = DOCUMENTS.get(resource)

    if clearance is None or classification is None:
        return False

    return (
        LEVELS[clearance]
        >= LEVELS[classification]
    )


def canonical(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def signature_for(payload):
    return hmac.new(
        ISSUER_KEY,
        canonical(payload),
        hashlib.sha256,
    ).hexdigest()


def issue_capability(
    requester,
    worker,
    action,
    resource,
    task_id,
    token_id,
    issued_at=1000,
    expires_at=1100,
):
    """
    Trusted issuer refuses unauthorized grants.

    This function is NOT accessible to an LLM
    as an unrestricted tool.
    """

    if not authorized(requester, resource):
        return None

    if worker not in ALLOWED_WORKERS:
        return None

    if action != "read_document":
        return None

    if not task_id or not token_id:
        return None

    if not issued_at < expires_at:
        return None

    payload = {
        "issuer": "guardian_issuer",
        "token_id": token_id,
        "requester": requester,
        "worker": worker,
        "action": action,
        "resource": resource,
        "task_id": task_id,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }

    return {
        "payload": payload,
        "signature": signature_for(payload),
    }


# ------------------------------------
# 2. STRICT CAPABILITY GUARD
# ------------------------------------

FIELDS = {
    "issuer",
    "token_id",
    "requester",
    "worker",
    "action",
    "resource",
    "task_id",
    "issued_at",
    "expires_at",
}


def verify_capability(
    token,
    context,
    revoked,
    consumed,
    now=NOW,
):
    """
    Verifies both the credential and the
    requested tool operation.

    revoked: trusted revocation registry.
    consumed: trusted one-time-use registry.

    Both must persist atomically in a real system.
    """

    if not isinstance(token, dict):
        return False, "MISSING_TOKEN"

    if set(token) != {"payload", "signature"}:
        return False, "INVALID_TOKEN"

    payload = token["payload"]
    signature = token["signature"]

    if (
        not isinstance(payload, dict)
        or not isinstance(signature, str)
        or set(payload) != FIELDS
    ):
        return False, "INVALID_TOKEN"

    string_fields = FIELDS - {
        "issued_at", "expires_at"
    }

    if not all(
        isinstance(payload[key], str)
        and bool(payload[key])
        for key in string_fields
    ):
        return False, "INVALID_FIELDS"

    if not (
        type(payload["issued_at"]) is int
        and type(payload["expires_at"]) is int
    ):
        return False, "INVALID_TIME"

    if payload["issuer"] != "guardian_issuer":
        return False, "UNKNOWN_ISSUER"

    expected = signature_for(payload)

    if not hmac.compare_digest(
        signature, expected
    ):
        return False, "BAD_SIGNATURE"

    if not (
        payload["issued_at"] <= now
        < payload["expires_at"]
    ):
        return False, "TOKEN_EXPIRED"

    token_id = payload["token_id"]

    if token_id in revoked:
        return False, "TOKEN_REVOKED"

    if token_id in consumed:
        return False, "TOKEN_REPLAY"

    if payload["requester"] != context["requester"]:
        return False, "REQUESTER_MISMATCH"

    if payload["worker"] != context["worker"]:
        return False, "WORKER_MISMATCH"

    if payload["task_id"] != context["task_id"]:
        return False, "TASK_MISMATCH"

    if payload["action"] != context["action"]:
        return False, "ACTION_MISMATCH"

    if payload["resource"] != context["resource"]:
        return False, "RESOURCE_MISMATCH"

    if not authorized(
        payload["requester"],
        payload["resource"],
    ):
        return False, "REQUESTER_UNAUTHORIZED"

    if payload["worker"] not in ALLOWED_WORKERS:
        return False, "INVALID_WORKER"

    # This simulates an atomic one-time claim.
    # It is not a durable distributed replay store.
    consumed.add(token_id)

    return True, "CAPABILITY_VERIFIED"


# ------------------------------------
# 3. DELIBERATELY WEAK BASELINE
# ------------------------------------

def bearer_baseline_check(token, context):
    """
    Vulnerable experimental baseline.

    Treats any token-shaped object as sufficient
    to authorize a known document read.

    Ignores signature, requester, scope,
    worker binding, task, expiry and revocation.
    """

    if (
        not isinstance(token, dict)
        or "payload" not in token
    ):
        return False, "MISSING_TOKEN"

    if context["action"] != "read_document":
        return False, "UNKNOWN_ACTION"

    if context["resource"] not in DOCUMENTS:
        return False, "UNKNOWN_DOCUMENT"

    return True, "BEARER_ONLY_ALLOW"


# ------------------------------------
# 4. SYNTHETIC EXECUTION
# ------------------------------------

def evaluate(
    policy,
    token,
    context,
    revoked=None,
    consumed=None,
    now=NOW,
):
    if revoked is None:
        revoked = set()

    if consumed is None:
        consumed = set()

    if policy == "bearer_baseline":
        allowed, reason = bearer_baseline_check(
            token, context
        )

    elif policy == "capability_guard":
        allowed, reason = verify_capability(
            token,
            context,
            revoked,
            consumed,
            now,
        )

    else:
        raise ValueError("Unknown policy")

    return {
        "policy": policy,
        "allowed": allowed,
        "executed": bool(allowed),
        "reason": reason,
        "resource": (
            context["resource"]
            if allowed else None
        ),
    }


# ------------------------------------
# 5. SCENARIO FIXTURES
# ------------------------------------

def make_scenario(name, repeat):
    if name not in SCENARIOS:
        raise ValueError("Unknown scenario")

    requester = (
        "visitor"
        if name in ("visitor_escalation", "valid_public")
        else "manager"
    )

    resource = (
        "public.txt"
        if name == "valid_public"
        else "confidential.txt"
    )

    context = {
        "requester": requester,
        "worker": "worker",
        "action": "read_document",
        "resource": resource,
        "task_id": f"task-{name}-{repeat}",
    }

    token = issue_capability(
        requester=requester,
        worker="worker",
        action="read_document",
        resource=resource,
        task_id=context["task_id"],
        token_id=f"cap-{name}-{repeat}",
    )

    revoked = set()
    consumed = set()

    if name == "other_resource":
        context["resource"] = "internal.txt"

    elif name == "wrong_worker":
        context["worker"] = "backup_worker"

    elif name == "wrong_task":
        context["task_id"] += "-different"

    elif name == "expired_token":
        # Fresh signature so only expiry fails.
        token["payload"]["expires_at"] = 1020
        token["signature"] = signature_for(
            token["payload"]
        )

    elif name == "revoked_token":
        revoked.add(
            token["payload"]["token_id"]
        )

    elif name == "tampered_scope":
        # Alter scope AFTER signing.
        token["payload"]["resource"] = (
            "internal.txt"
        )

    elif name == "replayed_token":
        # Simulate an already consumed capability.
        consumed.add(
            token["payload"]["token_id"]
        )

    return token, context, revoked, consumed


# ------------------------------------
# 6. PAIRED COMPARISON
# ------------------------------------

def run_trial(name, repeat):
    token, context, revoked, consumed = (
        make_scenario(name, repeat)
    )

    baseline = evaluate(
        "bearer_baseline",
        token,
        context,
        revoked=revoked,
        consumed=consumed.copy(),
    )

    verified = evaluate(
        "capability_guard",
        token,
        context,
        revoked=revoked,
        consumed=consumed.copy(),
    )

    expected_valid = name in (
        "valid_delegation",
        "valid_public",
    )

    return {
        "scenario": name,
        "repeat": repeat,
        "expected_valid": expected_valid,
        "context": context,
        "token_present": token is not None,
        "token_payload": (
            copy.deepcopy(token["payload"])
            if token else None
        ),
        "revoked": sorted(revoked),
        "previously_consumed": sorted(consumed),
        "baseline": baseline,
        "verified": verified,
        "baseline_invalid_execution": (
            baseline["executed"]
            and not expected_valid
        ),
        "verified_invalid_execution": (
            verified["executed"]
            and not expected_valid
        ),
    }


# ------------------------------------
# 7. SUMMARY
# ------------------------------------

def summarize(records):
    summary = {
        "trials": len(records),
        "conditions": {},
        "policies": {},
    }

    for name in SCENARIOS:
        rows = [
            row for row in records
            if row["scenario"] == name
        ]

        summary["conditions"][name] = {
            "trials": len(rows),
            "baseline_executions": sum(
                row["baseline"]["executed"]
                for row in rows
            ),
            "verified_executions": sum(
                row["verified"]["executed"]
                for row in rows
            ),
            "verified_reasons": sorted(set(
                row["verified"]["reason"]
                for row in rows
            )),
        }

    for policy, field in (
        ("bearer_baseline", "baseline"),
        ("capability_guard", "verified"),
    ):
        summary["policies"][policy] = {
            "executions": sum(
                row[field]["executed"]
                for row in records
            ),
            "invalid_executions": sum(
                row[field]["executed"]
                and not row["expected_valid"]
                for row in records
            ),
            "legitimate_completions": sum(
                row[field]["executed"]
                and row["expected_valid"]
                for row in records
            ),
        }

    return summary


# ------------------------------------
# 8. RUN EXPERIMENT
# ------------------------------------

def run_experiment(repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")

    records = []

    for name in SCENARIOS:
        for repeat in range(1, repeats + 1):
            record = run_trial(name, repeat)
            records.append(record)

            print(
                name,
                repeat,
                "| baseline:",
                record["baseline"]["reason"],
                "| verified:",
                record["verified"]["reason"],
            )

    report = {
        "experiment": "P3",
        "model": "deterministic_capability_simulation",
        "repeats": repeats,
        "summary": summarize(records),
        "records": records,
    }

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULTS / (
        f"p3_results_r{repeats}.json"
    )

    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL SUMMARY")
    print(json.dumps(
        report["summary"], indent=2
    ))
    print("Saved:", path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    run_experiment(args.repeats)
