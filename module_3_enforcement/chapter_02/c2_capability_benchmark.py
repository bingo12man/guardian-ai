
"""
Guardian AI — Module 3, Chapter 3.2 benchmark.

60 synthetic scenarios x 2 strategies.

Strategies:
  trusted_only
  attack_then_recover

A legitimate capability is never expanded by
attacker-controlled action proposals.

No real tool execution or live LLM calls.
"""

import hashlib
import json
import sys
from pathlib import Path

from guardian_capabilities import CapabilityGateway

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

sys.path.insert(
    0,
    str(ROOT / "module_2_simulation/chapter_01"),
)
sys.path.insert(
    0,
    str(ROOT / "module_2_simulation/chapter_02"),
)

import m2_2_attack_dataset as dataset


STRATEGIES = (
    "trusted_only",
    "attack_then_recover",
)


def run_case(scenario, strategy):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")

    task_data = scenario["trusted_task"]

    gateway = CapabilityGateway()

    try:
        task = gateway.register_task(
            task_data["requester"],
            task_data["operation"],
            task_data["target"],
        )
    except PermissionError:
        return {
            "scenario_id": scenario["scenario_id"],
            "strategy": strategy,
            "original_task_denied": True,
            "completed": False,
            "attack_attempted": False,
            "blocked_actions": 0,
            "unauthorized_executions": 0,
            "audit": [],
        }

    attack_attempted = False

    if (
        strategy == "attack_then_recover"
        and scenario["attack"]["present"]
    ):
        token = gateway.issue_capability(
            task.task_id,
            "search_documents",
            task.target,
        )

        wrong = dataset.attack_target(scenario)

        blocked = gateway.submit(
            task.task_id,
            {
                "tool": "search_documents",
                "argument": wrong,
            },
            token,
        )

        if blocked["executed"]:
            raise AssertionError(
                "Wrong-target action was executed"
            )

        attack_attempted = True

    completed = False

    for tool, argument in (
        ("search_documents", task.target),
        ("read_document", task.target),
        ("finish", "done"),
    ):
        # Trusted controller issues a fresh token.
        token = gateway.issue_capability(
            task.task_id,
            tool,
            argument,
        )

        decision = gateway.submit(
            task.task_id,
            {
                "tool": tool,
                "argument": argument,
            },
            token,
        )

        if not decision["allowed"]:
            break

        completed = bool(
            decision["completed"]
        )

    summary = gateway.summary()

    return {
        "scenario_id": scenario["scenario_id"],
        "strategy": strategy,
        "original_task_denied": False,
        "completed": completed,
        "attack_attempted": attack_attempted,
        "blocked_actions": summary[
            "blocked_actions"
        ],
        "unauthorized_executions": summary[
            "unauthorized_simulated_executions"
        ],
        "audit": gateway.audit_events(),
    }


def summarize(records):
    results = {}

    for strategy in STRATEGIES:
        rows = [
            r for r in records
            if r["strategy"] == strategy
        ]

        results[strategy] = {
            "episodes": len(rows),
            "completions": sum(
                r["completed"] for r in rows
            ),
            "original_tasks_denied": sum(
                r["original_task_denied"]
                for r in rows
            ),
            "attack_attempts": sum(
                r["attack_attempted"]
                for r in rows
            ),
            "blocked_actions": sum(
                r["blocked_actions"]
                for r in rows
            ),
            "unauthorized_executions": sum(
                r["unauthorized_executions"]
                for r in rows
            ),
        }

    return {
        "episodes": len(records),
        "strategies": results,
        "unauthorized_executions": sum(
            r["unauthorized_executions"]
            for r in records
        ),
    }


def run_benchmark(output_dir=RESULTS):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    scenarios = dataset.make_dataset()

    records = [
        run_case(scenario, strategy)
        for scenario in scenarios
        for strategy in STRATEGIES
    ]

    summary = summarize(records)

    if summary["unauthorized_executions"] != 0:
        raise AssertionError(
            "Unauthorized execution detected"
        )

    report = {
        "module": 3,
        "chapter": "3.2",
        "evidence_type": (
            "Deterministic synthetic "
            "capability enforcement"
        ),
        "scenario_count": len(scenarios),
        "summary": summary,
        "records": records,
    }

    path = output_dir / "c2_results.json"

    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    (output_dir / "manifest.json").write_text(
        json.dumps({
            "module": 3,
            "chapter": "3.2",
            "results_sha256": digest,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 3 C2 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", path)

    return summary


if __name__ == "__main__":
    run_benchmark()
