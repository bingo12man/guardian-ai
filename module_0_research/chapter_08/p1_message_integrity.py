
"""
Guardian AI - Chapter 0.8 - P1

Agent-to-agent message security.

Compare:
    naive_receiver
    verified_receiver

Attacks:
    message tampering
    sender spoofing
    replay
    expiration
    unauthorized scope

HMAC is used as a local demonstration.
No real files or external services are accessed.
"""

import argparse
import hashlib
import hmac
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Fixed synthetic time: reproducible offline tests.
NOW = 1050

# Test-only secret. NEVER use a hardcoded key
# like this for production credentials.
TEST_KEYS = {
    "planner": b"guardian-p1-local-fixture-key",
}

DOCUMENTS = {
    "public.txt": "public",
    "confidential.txt": "confidential",
}

CLEARANCE = {
    "visitor": "public",
    "manager": "confidential",
}

LEVELS = {
    "public": 0,
    "confidential": 1,
}

WORKER_CLEARANCE = "confidential"

SCENARIOS = (
    "valid_public",
    "valid_manager",
    "tampered_document",
    "spoofed_sender",
    "replay_message",
    "expired_message",
    "scope_escalation",
)


# -----------------------------------------
# 1. CANONICAL MESSAGE SIGNING
# -----------------------------------------

def canonical(data):
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sign_payload(payload, key):
    return hmac.new(
        key,
        canonical(payload),
        hashlib.sha256,
    ).hexdigest()


def create_envelope(
    requester,
    document,
    nonce,
    sender="planner",
    issued_at=1000,
    expires_at=1100,
):
    payload = {
        "sender": sender,
        "requester": requester,
        "document": document,
        "tool": "read_document",
        "nonce": nonce,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }

    if sender not in TEST_KEYS:
        raise ValueError("Unknown signing identity")

    return {
        "payload": payload,
        "signature": sign_payload(
            payload, TEST_KEYS[sender]
        ),
    }


# -----------------------------------------
# 2. BUILD FIXED ATTACK SCENARIOS
# -----------------------------------------

def make_scenario(name, repeat):
    if name not in SCENARIOS:
        raise ValueError("Unknown scenario")

    nonce = f"{name}-{repeat}"

    if name == "valid_manager":
        requester = "manager"
        document = "confidential.txt"
    elif name == "scope_escalation":
        requester = "visitor"
        document = "confidential.txt"
    else:
        requester = "visitor"
        document = "public.txt"

    envelope = create_envelope(
        requester,
        document,
        nonce,
    )

    # Context is assigned by a trusted controller,
    # not copied from the received message.
    context = {
        "requester": requester,
        "document": document,
        "sender": "planner",
    }

    if name == "tampered_document":
        envelope["payload"]["document"] = (
            "confidential.txt"
        )

    elif name == "spoofed_sender":
        envelope["payload"]["sender"] = (
            "manager_agent"
        )

    elif name == "expired_message":
        envelope["payload"]["expires_at"] = 1020

        # Re-sign to isolate expiry from tampering.
        envelope["signature"] = sign_payload(
            envelope["payload"],
            TEST_KEYS["planner"],
        )

    return envelope, context


# -----------------------------------------
# 3. AUTHORIZATION HELPERS
# -----------------------------------------

def authorized(principal, document):
    clearance = CLEARANCE.get(principal)
    classification = DOCUMENTS.get(document)

    if clearance is None or classification is None:
        return False

    return (
        LEVELS[clearance] >= LEVELS[classification]
    )


def worker_authorized(document):
    classification = DOCUMENTS.get(document)

    if classification is None:
        return False

    return (
        LEVELS[WORKER_CLEARANCE]
        >= LEVELS[classification]
    )


# -----------------------------------------
# 4. VERIFIED MESSAGE RECEIVER
# -----------------------------------------

def verify_envelope(envelope, context, seen, now=NOW):
    """
    Authenticate before trusting message fields.

    'seen' represents a replay cache maintained
    by the trusted receiving service.
    """

    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"payload", "signature"}
        or not isinstance(envelope["payload"], dict)
        or not isinstance(envelope["signature"], str)
    ):
        return False, "INVALID_ENVELOPE"

    payload = envelope["payload"]

    expected_fields = {
        "sender",
        "requester",
        "document",
        "tool",
        "nonce",
        "issued_at",
        "expires_at",
    }

    if set(payload) != expected_fields:
        return False, "INVALID_FIELDS"

    if not all(
        isinstance(payload[k], str)
        for k in (
            "sender",
            "requester",
            "document",
            "tool",
            "nonce",
        )
    ):
        return False, "INVALID_FIELDS"

    if not (
        type(payload["issued_at"]) is int
        and type(payload["expires_at"]) is int
    ):
        return False, "INVALID_TIME"

    # Do not trust claimed sender identity.
    # The key is selected from a trusted mapping,
    # constrained to the expected sender.
    expected_sender = context.get("sender")

    if payload["sender"] != expected_sender:
        return False, "SENDER_MISMATCH"

    key = TEST_KEYS.get(expected_sender)

    if key is None:
        return False, "UNKNOWN_SIGNER"

    expected_signature = sign_payload(
        payload, key
    )

    if not hmac.compare_digest(
        envelope["signature"],
        expected_signature,
    ):
        return False, "BAD_SIGNATURE"

    if not (
        payload["issued_at"] <= now
        < payload["expires_at"]
    ):
        return False, "EXPIRED_OR_NOT_YET_VALID"

    nonce_key = (
        expected_sender,
        payload["nonce"],
    )

    if nonce_key in seen:
        return False, "REPLAY"

    if payload["tool"] != "read_document":
        return False, "INVALID_TOOL"

    if payload["requester"] != context.get("requester"):
        return False, "REQUESTER_MISMATCH"

    if payload["document"] != context.get("document"):
        return False, "TASK_SCOPE_MISMATCH"

    if not authorized(
        context["requester"],
        payload["document"],
    ):
        return False, "REQUESTER_UNAUTHORIZED"

    if not worker_authorized(payload["document"]):
        return False, "WORKER_UNAUTHORIZED"

    # Record nonce only after all checks pass.
    seen.add(nonce_key)

    return True, "VERIFIED"


# -----------------------------------------
# 5. NAIVE VS VERIFIED RECEIVER
# -----------------------------------------

def evaluate(
    envelope,
    context,
    policy,
    seen,
    now=NOW,
):
    """
    Evaluate a proposed read from message fields.

    No document content is ever returned.
    """

    payload = envelope.get("payload", {})
    document = payload.get("document")

    # The naive policy intentionally checks
    # only the executing worker's clearance.
    if policy == "naive_receiver":
        allowed = worker_authorized(document)
        reason = (
            "WORKER_ONLY_ALLOW"
            if allowed else "WORKER_DENIED"
        )

    elif policy == "verified_receiver":
        allowed, reason = verify_envelope(
            envelope,
            context,
            seen,
            now,
        )

    else:
        raise ValueError("Unknown policy")

    return {
        "policy": policy,
        "allowed": allowed,
        "executed": allowed,
        "reason": reason,
        "document": document if allowed else None,
    }


# -----------------------------------------
# 6. PAIRED TRIAL
# -----------------------------------------

def run_trial(name, repeat):
    envelope, context = make_scenario(
        name, repeat
    )

    # Separate replay cache for each paired trial.
    seen = set()

    if name == "replay_message":
        first_allowed, first_reason = verify_envelope(
            envelope,
            context,
            seen,
            NOW,
        )
        assert first_allowed, first_reason

    naive = evaluate(
        envelope,
        context,
        "naive_receiver",
        seen,
    )

    verified = evaluate(
        envelope,
        context,
        "verified_receiver",
        seen,
    )

    # This is an independent intent/reference check,
    # not the rule used by the naive receiver.
    expected_intended_access = (
        document_matches_context(envelope, context)
        and authorized(
            context["requester"],
            envelope["payload"]["document"],
        )
    )

    # Every negative-case message is either
    # forged, stale, replayed, or unauthorized.
    protocol_valid_for_this_trial = name in (
        "valid_public",
        "valid_manager",
    )

    return {
        "scenario": name,
        "repeat": repeat,
        "context": context,
        "received_payload": envelope["payload"],
        "naive": naive,
        "verified": verified,
        "intended_access_authorized":
            expected_intended_access,
        "protocol_valid_for_this_trial":
            protocol_valid_for_this_trial,
        "naive_invalid_message_execution": (
            naive["executed"]
            and not protocol_valid_for_this_trial
        ),
        "verified_invalid_message_execution": (
            verified["executed"]
            and not protocol_valid_for_this_trial
        ),
        "naive_unauthorized_access": (
            naive["executed"]
            and not expected_intended_access
        ),
        "verified_unauthorized_access": (
            verified["executed"]
            and not expected_intended_access
        ),
    }


def document_matches_context(envelope, context):
    return (
        envelope["payload"]["document"]
        == context["document"]
    )


# -----------------------------------------
# 7. SUMMARY
# -----------------------------------------

def summarize(records):
    result = {
        "trials": len(records),
        "conditions": {},
        "policies": {},
    }

    for name in SCENARIOS:
        rows = [
            r for r in records
            if r["scenario"] == name
        ]

        result["conditions"][name] = {
            "trials": len(rows),
            "naive_executions": sum(
                r["naive"]["executed"]
                for r in rows
            ),
            "verified_executions": sum(
                r["verified"]["executed"]
                for r in rows
            ),
            "verified_reasons": sorted(set(
                r["verified"]["reason"]
                for r in rows
            )),
        }

    for policy, prefix in (
        ("naive_receiver", "naive"),
        ("verified_receiver", "verified"),
    ):
        result["policies"][policy] = {
            "executions": sum(
                r[prefix]["executed"]
                for r in records
            ),
            "invalid_message_executions": sum(
                r[f"{prefix}_invalid_message_execution"]
                for r in records
            ),
            "unauthorized_access_executions": sum(
                r[f"{prefix}_unauthorized_access"]
                for r in records
            ),
        }

    return result


# -----------------------------------------
# 8. RUN BENCHMARK
# -----------------------------------------

def run_experiment(repeats=3):
    if repeats < 1:
        raise ValueError(
            "repeats must be >= 1"
        )

    records = []

    for name in SCENARIOS:
        for repeat in range(1, repeats + 1):
            record = run_trial(name, repeat)
            records.append(record)

            print(
                name,
                repeat,
                "| naive:",
                record["naive"]["reason"],
                "| verified:",
                record["verified"]["reason"],
            )

    report = {
        "experiment": "P1",
        "model": "deterministic_protocol_simulation",
        "repeats": repeats,
        "summary": summarize(records),
        "records": records,
    }

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULTS / (
        f"p1_results_r{repeats}.json"
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
