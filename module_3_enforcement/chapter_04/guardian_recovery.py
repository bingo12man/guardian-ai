
"""
Guardian AI — Module 3, Chapter 3.4

Bounded recovery with synthetic pre-effect failures.

Important:
- Uses Chapter 3.2 internals as a research adapter.
- Backend failures occur BEFORE any effect.
- Successful effects are committed to task state.
- Each submission consumes its valid capability.
- Each retry needs a newly issued capability.
- Unknown or ambiguous backend outcomes must
  NOT be automatically retried.
- No real filesystem, network or external tools.

This prototype is sequential and in-process,
not an atomic distributed transaction system.
"""

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chapter_01"))
sys.path.insert(0, str(HERE.parent / "chapter_02"))

from guardian_core import parse_proposal
from guardian_capabilities import CapabilityGateway


GENESIS = hashlib.sha256(
    b"GUARDIAN_M3_C4_RECOVERY_AUDIT_V1"
).hexdigest()

DOCUMENTS = {
    "public.txt": "Synthetic public quarterly overview.",
    "guide.txt": "Synthetic public user guide.",
    "confidential.txt": "Synthetic internal operations report.",
    "finance.txt": "Synthetic internal finance statement.",
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


class KnownPreEffectFailure(Exception):
    """Backend guarantees that no effect occurred."""


class AmbiguousOutcome(Exception):
    """Effect may have occurred: manual reconciliation."""


class SyntheticFailureBackend:
    """
    Deterministic fault-injection backend.

    fail_before:
      Number of initial attempts to fail for each
      task_id + tool pair.

    ambiguous_before:
      Simulated uncertain outcome without recorded
      local effect. Must never trigger auto-retry.

    The ambiguity fixture models uncertainty,
    not proof that the operation did or did not run.
    """

    def __init__(
        self,
        fail_before=0,
        ambiguous_before=0,
    ):
        if (
            not isinstance(fail_before, int)
            or fail_before < 0
            or not isinstance(ambiguous_before, int)
            or ambiguous_before < 0
        ):
            raise ValueError("Invalid fault configuration")

        self.fail_before = fail_before
        self.ambiguous_before = ambiguous_before
        self.attempts = Counter()
        self._effects = []

    def execute(self, task_id, tool, argument):
        key = (task_id, tool)
        self.attempts[key] += 1
        attempt = self.attempts[key]

        if attempt <= self.ambiguous_before:
            raise AmbiguousOutcome(
                "Backend outcome cannot be confirmed"
            )

        if attempt <= (
            self.ambiguous_before + self.fail_before
        ):
            raise KnownPreEffectFailure(
                "Injected failure before effect"
            )

        if tool == "search_documents":
            output = f"Synthetic search results: {argument}"
        elif tool == "read_document":
            output = DOCUMENTS[argument]
        elif tool == "finish" and argument == "done":
            output = "Synthetic task completed."
        else:
            raise ValueError("Unsupported synthetic tool")

        self._effects.append({
            "task_id": task_id,
            "tool": tool,
            "argument": argument,
            "output": output,
        })

        return output

    def effects(self):
        return copy.deepcopy(self._effects)


class AgentSubmitPort:
    """Agent receives submission, not token issuance."""

    def __init__(self, callback):
        self.__callback = callback

    def submit(self, task_id, proposal, capability):
        return self.__callback(
            task_id, proposal, capability
        )


class GuardianRecoveryGateway:
    """
    Sequential experimental gateway.

    Internally uses CapabilityGateway's verifier
    and GuardianEngine's policy state machine.
    The signing authority must remain trusted.
    """

    def __init__(
        self,
        backend=None,
        signing_key=None,
        clock=None,
    ):
        self._caps = CapabilityGateway(
            signing_key=signing_key,
            clock=clock,
        )
        self._backend = (
            backend if backend is not None
            else SyntheticFailureBackend()
        )
        self._events = []
        self._uncertain_tasks = set()

    def register_task(self, requester, operation, target):
        return self._caps.register_task(
            requester, operation, target
        )

    def issue_capability(
        self, task_id, tool, argument,
        ttl_seconds=60,
    ):
        if task_id in self._uncertain_tasks:
            raise RuntimeError("TASK_REQUIRES_RECONCILIATION")

        return self._caps.issue_capability(
            task_id, tool, argument,
            ttl_seconds=ttl_seconds,
        )

    def agent_port(self):
        return AgentSubmitPort(self.submit)

    def get_stage(self, task_id):
        return self._caps.get_stage(task_id)

    def tool_effects(self):
        return self._backend.effects()

    def audit_events(self):
        return copy.deepcopy(self._events)

    def final_audit_hash(self):
        return (
            self._events[-1]["event_hash"]
            if self._events else GENESIS
        )

    def _record(self, task_id, proposal, response):
        previous = self.final_audit_hash()
        event = {
            "sequence": len(self._events) + 1,
            "previous_hash": previous,
            "task_id": task_id,
            "proposal": (
                {
                    "tool": proposal.tool,
                    "argument": proposal.argument,
                }
                if proposal is not None else None
            ),
            "response": copy.deepcopy(response),
        }
        event["event_hash"] = digest(event)
        self._events.append(event)
        return copy.deepcopy(response)

    def _deny(self, task_id, proposal, reason):
        stage = self.get_stage(task_id) or "unknown"
        return self._record(task_id, proposal, {
            "status": "DENIED",
            "reason": reason,
            "authorized": False,
            "effect_executed": False,
            "completed": False,
            "stage_before": stage,
            "stage_after": stage,
            "output": None,
        })

    def submit(self, task_id, raw_action, capability):
        proposal = parse_proposal(raw_action)

        if task_id in self._uncertain_tasks:
            return self._deny(
                task_id, proposal,
                "TASK_REQUIRES_RECONCILIATION",
            )

        if proposal is None:
            return self._deny(
                task_id, None, "INVALID_ACTION"
            )

        # Research adapter: use C2's internal
        # signature, scope, TTL and replay verifier.
        payload, reason = self._caps._verify(
            capability, task_id, proposal
        )

        if payload is None:
            return self._deny(task_id, proposal, reason)

        task = self._caps._engine.get_task(task_id)
        stage = self.get_stage(task_id)

        if task is None:
            return self._deny(
                task_id, proposal, "UNKNOWN_TASK"
            )

        expected = {
            "need_search": (
                "search_documents", task.target
            ),
            "need_read": (
                "read_document", task.target
            ),
            "need_finish": ("finish", "done"),
        }.get(stage)

        if expected != (proposal.tool, proposal.argument):
            return self._deny(
                task_id, proposal, "POLICY_SCOPE_DENIED"
            )

        # Consume a verified capability BEFORE
        # attempting the backend operation.
        # A failed attempt cannot reuse it.
        self._caps._consumed.add((
            payload["task_id"],
            payload["nonce"],
        ))

        before = stage

        try:
            output = self._backend.execute(
                task_id, proposal.tool,
                proposal.argument,
            )

        except KnownPreEffectFailure:
            return self._record(task_id, proposal, {
                "status": "RETRYABLE_FAILURE",
                "reason": "BACKEND_PRE_EFFECT_FAILURE",
                "authorized": True,
                "effect_executed": False,
                "completed": False,
                "stage_before": before,
                "stage_after": before,
                "output": None,
            })

        except AmbiguousOutcome:
            self._uncertain_tasks.add(task_id)
            return self._record(task_id, proposal, {
                "status": "RECONCILIATION_REQUIRED",
                "reason": "AMBIGUOUS_BACKEND_OUTCOME",
                "authorized": True,
                "effect_executed": False,
                "completed": False,
                "stage_before": before,
                "stage_after": before,
                "output": None,
            })

        # The synthetic backend has confirmed
        # success. Only now commit Guardian state.
        decision = self._caps._engine.propose(
            task_id, proposal
        )

        if not decision.allowed:
            # The effect already occurred. Never
            # pretend rollback was successful.
            self._uncertain_tasks.add(task_id)
            return self._record(task_id, proposal, {
                "status": "RECONCILIATION_REQUIRED",
                "reason": "COMMIT_AFTER_EFFECT_FAILED",
                "authorized": True,
                "effect_executed": True,
                "completed": False,
                "stage_before": before,
                "stage_after": self.get_stage(task_id),
                "output": None,
            })

        return self._record(task_id, proposal, {
            "status": "SUCCEEDED",
            "reason": decision.reason,
            "authorized": True,
            "effect_executed": True,
            "completed": decision.completed,
            "stage_before": before,
            "stage_after": decision.stage_after,
            "output": output,
        })

    def summary(self):
        responses = [
            e["response"] for e in self._events
        ]
        return {
            "submissions": len(responses),
            "denied": sum(
                r["status"] == "DENIED"
                for r in responses
            ),
            "retryable_failures": sum(
                r["status"] == "RETRYABLE_FAILURE"
                for r in responses
            ),
            "reconciliation_required": sum(
                r["status"] == "RECONCILIATION_REQUIRED"
                for r in responses
            ),
            "successful_actions": sum(
                r["status"] == "SUCCEEDED"
                for r in responses
            ),
            "tool_effects": len(self.tool_effects()),
            "denied_actions_with_effects": sum(
                r["status"] == "DENIED"
                and r["effect_executed"]
                for r in responses
            ),
            "unauthorized_simulated_executions":
                self._caps._engine.summary()[
                    "unauthorized_simulated_executions"
                ],
        }


def run_bounded_action(
    gateway,
    task_id,
    tool,
    argument,
    max_attempts=3,
):
    """
    Controller-owned retry loop.

    A fresh capability is issued for each try.
    Only known pre-effect failures are retried.
    Other outcomes stop immediately.
    """
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 1
    ):
        raise ValueError("Invalid retry budget")

    results = []

    for _ in range(max_attempts):
        token = gateway.issue_capability(
            task_id, tool, argument
        )

        response = gateway.agent_port().submit(
            task_id,
            {"tool": tool, "argument": argument},
            token,
        )
        results.append(response)

        if response["status"] == "RETRYABLE_FAILURE":
            continue

        return {
            "recovered": (
                len(results) > 1
                and response["status"] == "SUCCEEDED"
            ),
            "attempts": len(results),
            "final": response,
            "history": results,
        }

    return {
        "recovered": False,
        "attempts": len(results),
        "final": results[-1],
        "history": results,
        "budget_exhausted": True,
    }


def verify_audit(events, expected_hash=None):
    previous = GENESIS

    for index, event in enumerate(events, 1):
        if event["sequence"] != index:
            raise ValueError("Invalid event sequence")
        if event["previous_hash"] != previous:
            raise ValueError("Broken audit chain")

        unsigned = {
            key: value for key, value in event.items()
            if key != "event_hash"
        }
        if digest(unsigned) != event["event_hash"]:
            raise ValueError("Corrupted audit event")

        response = event["response"]

        if response["status"] in (
            "DENIED", "RETRYABLE_FAILURE"
        ):
            if (
                response["effect_executed"]
                or response["output"] is not None
                or response["stage_before"]
                != response["stage_after"]
            ):
                raise ValueError(
                    "Invalid denied/failed action record"
                )

        previous = event["event_hash"]

    if (
        expected_hash is not None
        and previous != expected_hash
    ):
        raise ValueError("Audit anchor mismatch")

    return True
