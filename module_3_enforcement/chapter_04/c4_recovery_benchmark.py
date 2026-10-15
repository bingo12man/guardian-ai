
"""
Module 3 Chapter 3.4 deterministic recovery benchmark.

60 scenarios x 4 strategies = 240 episodes.

No real tools or model calls.
"""

import hashlib
import json
import sys
from pathlib import Path

from guardian_recovery import (
    GuardianRecoveryGateway,
    SyntheticFailureBackend,
    run_bounded_action,
    verify_audit,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(
    0, str(ROOT / "module_2_simulation/chapter_01")
)
sys.path.insert(
    0, str(ROOT / "module_2_simulation/chapter_02")
)

import m2_2_attack_dataset as dataset


STRATEGIES = (
    "normal",
    "transient_failure",
    "persistent_failure",
    "attack_then_recover",
)


def run_case(scenario, strategy):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")

    fail_before = {
        "normal": 0,
        "transient_failure": 1,
        "persistent_failure": 10,
        "attack_then_recover": 0,
    }[strategy]

    gateway = GuardianRecoveryGateway(
        backend=SyntheticFailureBackend(
            fail_before=fail_before
        )
    )

    trusted = scenario["trusted_task"]

    try:
        task = gateway.register_task(
            trusted["requester"],
            trusted["operation"],
            trusted["target"],
        )
    except PermissionError:
        return {
            "strategy": strategy,
            "scenario_id": scenario["scenario_id"],
            "original_task_denied": True,
            "completed": False,
            "attack_attempts": 0,
            "retryable_failures": 0,
            "successful_actions": 0,
            "tool_effects": 0,
            "blocked_submissions": 0,
            "denied_actions_with_effects": 0,
            "unauthorized_executions": 0,
            "audit_verified": True,
        }

    attack_attempts = 0

    if (
        strategy == "attack_then_recover"
        and scenario["attack"]["present"]
    ):
        token = gateway.issue_capability(
            task.task_id,
            "search_documents",
            task.target,
        )

        denied = gateway.agent_port().submit(
            task.task_id,
            {
                "tool": "search_documents",
                "argument":
                    dataset.attack_target(scenario),
            },
            token,
        )

        if denied["effect_executed"]:
            raise AssertionError(
                "Attack caused a tool effect"
            )

        attack_attempts = 1

    completed = False

    for tool, argument in (
        ("search_documents", task.target),
        ("read_document", task.target),
        ("finish", "done"),
    ):
        outcome = run_bounded_action(
            gateway,
            task.task_id,
            tool,
            argument,
            max_attempts=3,
        )

        if outcome["final"]["status"] != "SUCCEEDED":
            break

        completed = outcome["final"]["completed"]

    summary = gateway.summary()

    verify_audit(
        gateway.audit_events(),
        gateway.final_audit_hash(),
    )

    return {
        "strategy": strategy,
        "scenario_id": scenario["scenario_id"],
        "original_task_denied": False,
        "completed": completed,
        "attack_attempts": attack_attempts,
        "retryable_failures":
            summary["retryable_failures"],
        "successful_actions":
            summary["successful_actions"],
        "tool_effects": summary["tool_effects"],
        "blocked_submissions": summary["denied"],
        "denied_actions_with_effects":
            summary["denied_actions_with_effects"],
        "unauthorized_executions":
            summary["unauthorized_simulated_executions"],
        "audit_verified": True,
    }


def summarize(records):
    by_strategy = {}

    for strategy in STRATEGIES:
        rows = [
            row for row in records
            if row["strategy"] == strategy
        ]
        by_strategy[strategy] = {
            "episodes": len(rows),
            "completions": sum(
                r["completed"] for r in rows
            ),
            "original_tasks_denied": sum(
                r["original_task_denied"] for r in rows
            ),
            "attack_attempts": sum(
                r["attack_attempts"] for r in rows
            ),
            "retryable_failures": sum(
                r["retryable_failures"] for r in rows
            ),
            "tool_effects": sum(
                r["tool_effects"] for r in rows
            ),
            "blocked_submissions": sum(
                r["blocked_submissions"] for r in rows
            ),
            "unauthorized_executions": sum(
                r["unauthorized_executions"] for r in rows
            ),
            "audits_verified": sum(
                r["audit_verified"] for r in rows
            ),
        }

    return {
        "episodes": len(records),
        "strategies": by_strategy,
        "denied_actions_with_effects": sum(
            r["denied_actions_with_effects"]
            for r in records
        ),
        "unauthorized_executions": sum(
            r["unauthorized_executions"]
            for r in records
        ),
    }


def run_benchmark(output_dir=RESULTS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = dataset.make_dataset()

    records = [
        run_case(scenario, strategy)
        for scenario in scenarios
        for strategy in STRATEGIES
    ]

    summary = summarize(records)

    if (
        summary["unauthorized_executions"] != 0
        or summary["denied_actions_with_effects"] != 0
    ):
        raise AssertionError(
            "Guardian safety invariant failed"
        )

    report = {
        "module": 3,
        "chapter": "3.4",
        "evidence_type":
            "Deterministic injected backend failures",
        "scenario_count": len(scenarios),
        "summary": summary,
        "records": records,
    }

    result_path = output_dir / "c4_results.json"
    result_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    result_hash = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()

    (output_dir / "manifest.json").write_text(
        json.dumps({
            "module": 3,
            "chapter": "3.4",
            "results_sha256": result_hash,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 3 C4 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", result_path)

    return summary


if __name__ == "__main__":
    run_benchmark()
