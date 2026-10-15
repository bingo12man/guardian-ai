
"""
Guardian AI — Module 3, Chapter 3.2

HMAC-SHA256 signed, short-lived, single-use
action capabilities.

Security properties:
- Exact task/tool/resource binding
- Workflow-stage binding
- Expiration
- Keyed signature verification
- Single-use replay prevention
- Strict token schema
- Guardian authorization still enforced
- Failed capabilities never execute actions

Synthetic execution only.

The trusted controller must retain exclusive
access to issue_capability().
"""

import base64
import hashlib
import hmac
import json
import math
import secrets
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(
    0, str(HERE.parent / "chapter_01")
)

from guardian_core import (
    ActionProposal,
    GuardianEngine,
    parse_proposal,
)


TOKEN_VERSION = 1

TOKEN_FIELDS = {
    "version",
    "task_id",
    "tool",
    "argument",
    "stage",
    "issued_at",
    "expires_at",
    "nonce",
}

VALID_STAGES = {
    "need_search",
    "need_read",
    "need_finish",
}


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def encode64(data):
    return base64.urlsafe_b64encode(
        data
    ).rstrip(b"=").decode("ascii")


def decode64(value):
    if not isinstance(value, str):
        raise ValueError("Invalid encoding")

    # Reject ambiguous base64 forms.
    if not value or any(
        character not in
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789-_"
        for character in value
    ):
        raise ValueError("Invalid encoding")

    data = base64.urlsafe_b64decode(
        value + "=" * (-len(value) % 4)
    )

    if encode64(data) != value:
        raise ValueError("Noncanonical encoding")

    return data


class CapabilityError(Exception):
    pass


class CapabilityGateway:
    """
    Trusted controller wrapper for GuardianEngine.

    Agents should receive a restricted submit-only
    interface, not this entire object.

    This class does not implement an OS/process
    security boundary.
    """

    def __init__(
        self,
        engine=None,
        signing_key=None,
        clock=None,
        max_ttl=300,
    ):
        self._engine = (
            engine if engine is not None
            else GuardianEngine()
        )

        self._key = (
            signing_key
            if signing_key is not None
            else secrets.token_bytes(32)
        )

        if (
            not isinstance(self._key, bytes)
            or len(self._key) < 32
        ):
            raise ValueError(
                "Signing key must be >=32 bytes"
            )

        if max_ttl < 1:
            raise ValueError("Invalid maximum TTL")

        self._clock = (
            clock if clock is not None
            else time.time
        )
        self._max_ttl = max_ttl
        self._consumed = set()
        self._audit = []

    def register_task(
        self, requester, operation, target
    ):
        return self._engine.register_task(
            requester, operation, target
        )

    def get_stage(self, task_id):
        return self._engine.get_stage(task_id)

    def issue_capability(
        self,
        task_id,
        tool,
        argument,
        ttl_seconds=60,
    ):
        """
        TRUSTED CONTROLLER ONLY.

        Do not expose this method to an agent.
        The controller authorizes issuance.

        Reject invalid stage, target or action
        before signing.
        """

        if (
            not isinstance(ttl_seconds, (int, float))
            or isinstance(ttl_seconds, bool)
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
            or ttl_seconds > self._max_ttl
        ):
            raise CapabilityError("INVALID_TTL")

        task = self._engine.get_task(task_id)
        stage = self._engine.get_stage(task_id)

        if task is None or stage not in VALID_STAGES:
            raise CapabilityError("INVALID_TASK_STATE")

        proposal = parse_proposal({
            "tool": tool,
            "argument": argument,
        })

        if proposal is None:
            raise CapabilityError("INVALID_ACTION")

        expected = {
            "need_search": (
                "search_documents", task.target
            ),
            "need_read": (
                "read_document", task.target
            ),
            "need_finish": (
                "finish", "done"
            ),
        }[stage]

        if (tool, argument) != expected:
            raise CapabilityError("SCOPE_DENIED")

        issued = float(self._clock())

        if not math.isfinite(issued):
            raise CapabilityError("INVALID_CLOCK")

        payload = {
            "version": TOKEN_VERSION,
            "task_id": task_id,
            "tool": tool,
            "argument": argument,
            "stage": stage,
            "issued_at": issued,
            "expires_at": issued + ttl_seconds,
            "nonce": secrets.token_hex(16),
        }

        body = encode64(canonical(payload))
        signature = hmac.new(
            self._key,
            body.encode("ascii"),
            hashlib.sha256,
        ).digest()

        return body + "." + encode64(signature)

    def _verify(
        self, token, task_id, proposal
    ):
        if (
            not isinstance(token, str)
            or len(token) > 4096
            or token.count(".") != 1
        ):
            return None, "INVALID_CAPABILITY"

        try:
            body, signature = token.split(".")

            supplied = decode64(signature)

            expected = hmac.new(
                self._key,
                body.encode("ascii"),
                hashlib.sha256,
            ).digest()

            if not hmac.compare_digest(
                supplied, expected
            ):
                return None, "INVALID_SIGNATURE"

            payload = json.loads(decode64(body))

        except (
            ValueError, TypeError,
            UnicodeError, OverflowError,
        ):
            return None, "INVALID_CAPABILITY"

        if (
            not isinstance(payload, dict)
            or set(payload) != TOKEN_FIELDS
            or payload["version"] != TOKEN_VERSION
        ):
            return None, "INVALID_CAPABILITY"

        if not all(
            isinstance(payload[k], str)
            for k in (
                "task_id", "tool", "argument",
                "stage", "nonce"
            )
        ):
            return None, "INVALID_CAPABILITY"

        if not all(
            isinstance(payload[k], (int, float))
            and not isinstance(payload[k], bool)
            and math.isfinite(payload[k])
            for k in ("issued_at", "expires_at")
        ):
            return None, "INVALID_CAPABILITY"

        now = float(self._clock())

        if not math.isfinite(now):
            return None, "INVALID_CLOCK"

        if (
            payload["expires_at"] <= now
            or payload["issued_at"] > now
            or payload["expires_at"]
            <= payload["issued_at"]
            or payload["expires_at"]
            - payload["issued_at"] > self._max_ttl
        ):
            return None, "EXPIRED_OR_INVALID_TIME"

        if payload["task_id"] != task_id:
            return None, "TASK_BINDING_MISMATCH"

        if (
            payload["tool"] != proposal.tool
            or payload["argument"]
            != proposal.argument
        ):
            return None, "CAPABILITY_SCOPE_MISMATCH"

        key = (
            payload["task_id"],
            payload["nonce"],
        )

        if key in self._consumed:
            return None, "CAPABILITY_REPLAY"

        stage = self._engine.get_stage(task_id)

        if stage is None:
            return None, "UNKNOWN_TASK"

        if stage != payload["stage"]:
            return None, "CAPABILITY_STAGE_MISMATCH"

        return payload, "VALID"

    def submit(
        self, task_id, raw_action, token
    ):
        """
        This is the intended agent-facing entrypoint.

        Capability verification precedes the
        underlying Guardian action decision.
        """

        proposal = parse_proposal(raw_action)

        if proposal is None:
            return self._deny(
                task_id, "INVALID_ACTION"
            )

        payload, reason = self._verify(
            token, task_id, proposal
        )

        if payload is None:
            return self._deny(task_id, reason)

        # Consume before delegating to Guardian.
        # Even an independently denied action
        # cannot reuse its capability.
        self._consumed.add((
            payload["task_id"],
            payload["nonce"],
        ))

        decision = self._engine.propose(
            task_id, proposal
        )

        response = {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "executed": decision.executed,
            "completed": decision.completed,
            "stage_before": decision.stage_before,
            "stage_after": decision.stage_after,
        }

        self._audit.append({
            "task_id": task_id,
            "verified": True,
            "decision": response,
        })

        return response

    def _deny(self, task_id, reason):
        stage = self._engine.get_stage(
            task_id
        ) or "unknown"

        response = {
            "allowed": False,
            "reason": reason,
            "executed": False,
            "completed": False,
            "stage_before": stage,
            "stage_after": stage,
        }

        self._audit.append({
            "task_id": task_id,
            "verified": False,
            "decision": response,
        })

        return response

    def audit_events(self):
        return json.loads(json.dumps(
            self._audit
        ))

    def summary(self):
        return {
            "submissions": len(self._audit),
            "verified_submissions": sum(
                e["verified"] for e in self._audit
            ),
            "capability_denials": sum(
                not e["verified"] for e in self._audit
            ),
            "allowed_actions": sum(
                e["decision"]["allowed"]
                for e in self._audit
            ),
            "blocked_actions": sum(
                not e["decision"]["allowed"]
                for e in self._audit
            ),
            "unauthorized_simulated_executions":
                self._engine.summary()[
                    "unauthorized_simulated_executions"
                ],
        }
