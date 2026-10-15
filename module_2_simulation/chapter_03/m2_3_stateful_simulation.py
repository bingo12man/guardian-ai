
"""
Guardian AI — Module 2, Chapter 2.3

Stateful adversarial simulation and audit verification.

Inputs: 60 canonical Chapter 2.2 synthetic scenarios.
Strategies: stop, retry, recover, persistent.

Everything is synthetic. No Ollama calls or
real document access.

Audit events form an SHA-256 hash chain.
Verification also replays actions against the
independent synthetic policy environment.
"""

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE.parent / "chapter_01"))
sys.path.insert(0, str(HERE.parent / "chapter_02"))

import m2_1_environment as base
import m2_2_attack_dataset as attacks


STRATEGIES = (
    "stop",
    "retry",
    "recover",
    "persistent",
)

MAX_STEPS = 5

GENESIS = hashlib.sha256(
    b"GUARDIAN_AI_M2_C3_AUDIT_V1"
).hexdigest()


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def trusted_actions(task):
    """Actions constructed only from trusted state."""
    target = task["target"]
    return [
        ("search_documents", target),
        ("read_document", target),
        ("finish", "done"),
    ]


def adversarial_action(scenario, attempt):
    """
    Scripted attacker changes tactics after denials.

    First attempt: search wrong target.
    Second attempt: read wrong target.
    Later: alternate them.

    The attacker cannot update trusted task state.
    """
    wrong = attacks.attack_target(scenario)

    if attempt % 2 == 1:
        return "search_documents", wrong

    return "read_document", wrong


def build_action(
    scenario,
    strategy,
    step,
    attack_attempts,
    denial_observed,
    trusted_index,
):
    """
    Return tool, argument, source.

    On an attack scenario:
      stop/retry/recover/persistent start with
      an attacker-selected proposal.

    Retry switches to trusted work after 2 blocks.
    Recover switches after 1 block.
    Persistent never switches.
    """
    task = scenario["trusted_task"]

    if not scenario["attack"]["present"]:
        tool, arg = trusted_actions(task)[trusted_index]
        return tool, arg, "trusted_script"

    if strategy == "persistent":
        tool, arg = adversarial_action(
            scenario, attack_attempts + 1
        )
        return tool, arg, "attacker_script"

    if strategy == "retry" and attack_attempts < 2:
        tool, arg = adversarial_action(
            scenario, attack_attempts + 1
        )
        return tool, arg, "attacker_script"

    if strategy in ("stop", "recover") and not denial_observed:
        tool, arg = adversarial_action(
            scenario, attack_attempts + 1
        )
        return tool, arg, "attacker_script"

    tool, arg = trusted_actions(task)[trusted_index]
    return tool, arg, "trusted_recovery"


def append_event(events, episode_id, payload):
    previous = (
        events[-1]["event_hash"]
        if events else GENESIS
    )

    event = {
        "episode_id": episode_id,
        "sequence": len(events) + 1,
        "previous_hash": previous,
        **payload,
    }

    event["event_hash"] = digest(
        canonical({
            key: value
            for key, value in event.items()
            if key != "event_hash"
        })
    )

    events.append(event)


def run_episode(
    scenario,
    strategy,
    repeat=1,
    max_steps=MAX_STEPS,
):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")

    if repeat < 1 or max_steps < 1:
        raise ValueError("Invalid episode configuration")

    base.validate_scenario(scenario)

    trusted_snapshot = copy.deepcopy(
        scenario["trusted_task"]
    )
    episode_id = (
        f"{scenario['scenario_id']}::{strategy}::r{repeat}"
    )

    env = base.SyntheticEnvironment(scenario)

    events = []
    attack_attempts = 0
    denial_observed = False
    trusted_index = 0
    stop_reason = None

    for step in range(1, max_steps + 1):
        if not base.task_authorized(trusted_snapshot):
            tool, argument = (
                "search_documents",
                trusted_snapshot["target"],
            )
            source = "original_task_check"
        else:
            tool, argument, source = build_action(
                scenario,
                strategy,
                step,
                attack_attempts,
                denial_observed,
                trusted_index,
            )

        before = env.stage
        decision = env.execute(tool, argument)
        after = env.stage

        if source == "attacker_script":
            attack_attempts += 1

        if not decision.allowed:
            denial_observed = True

        if (
            decision.allowed
            and source != "attacker_script"
        ):
            trusted_index += 1

        append_event(
            events,
            episode_id,
            {
                "step": step,
                "source": source,
                "tool": tool,
                "argument": argument,
                "stage_before": before,
                "stage_after": after,
                "allowed": decision.allowed,
                "executed": decision.executed,
                "completed": decision.completed,
                "reason": decision.reason,
            },
        )

        if not base.task_authorized(trusted_snapshot):
            stop_reason = "ORIGINAL_TASK_DENIED"
            break

        if decision.completed:
            stop_reason = "TASK_COMPLETED"
            break

        if (
            strategy == "stop"
            and not decision.allowed
        ):
            stop_reason = "STOP_ON_DENIAL"
            break

    if stop_reason is None:
        stop_reason = "STEP_BUDGET_EXHAUSTED"

    if env.trusted_task != trusted_snapshot:
        raise AssertionError("Trusted task was mutated")

    summary = env.summary()

    return {
        "episode_id": episode_id,
        "scenario_id": scenario["scenario_id"],
        "strategy": strategy,
        "repeat": repeat,
        "max_steps": max_steps,
        "original_task_authorized":
            base.task_authorized(trusted_snapshot),
        "attack_present":
            scenario["attack"]["present"],
        "attack_technique":
            scenario["attack"]["technique"],
        "trusted_task_sha256": digest(
            canonical(trusted_snapshot)
        ),
        "attack_attempts": attack_attempts,
        "steps": len(events),
        "completed": env.completed,
        "blocks": summary["blocked_actions"],
        "unauthorized_executions":
            summary["unauthorized_executions"],
        "stop_reason": stop_reason,
        "genesis_hash": GENESIS,
        "final_hash": (
            events[-1]["event_hash"]
            if events else GENESIS
        ),
        "events": events,
    }


def verify_episode(episode, scenario):
    """
    Independently replay events against a fresh
    environment and verify decision and hash chain.

    A trusted external copy of final_hash is needed
    to detect wholesale rewriting of a chain.
    """
    base.validate_scenario(scenario)

    expected_id = (
        f"{scenario['scenario_id']}::"
        f"{episode['strategy']}::r{episode['repeat']}"
    )

    if episode["episode_id"] != expected_id:
        raise ValueError("Episode identity mismatch")

    if episode["genesis_hash"] != GENESIS:
        raise ValueError("Genesis mismatch")

    if episode["trusted_task_sha256"] != digest(
        canonical(scenario["trusted_task"])
    ):
        raise ValueError("Trusted task mismatch")

    fresh = base.SyntheticEnvironment(scenario)
    previous = GENESIS
    events = episode["events"]

    if not events or len(events) > episode["max_steps"]:
        raise ValueError("Invalid audit event count")

    for index, event in enumerate(events, start=1):
        if event["sequence"] != index:
            raise ValueError("Sequence mismatch")

        if event["step"] != index:
            raise ValueError("Step mismatch")

        if event["episode_id"] != expected_id:
            raise ValueError("Event identity mismatch")

        if event["previous_hash"] != previous:
            raise ValueError("Hash chain discontinuity")

        unsigned = {
            k: v for k, v in event.items()
            if k != "event_hash"
        }

        calculated = digest(canonical(unsigned))

        if event["event_hash"] != calculated:
            raise ValueError("Event hash mismatch")

        if fresh.completed:
            raise ValueError("Action after completion")

        if event["stage_before"] != fresh.stage:
            raise ValueError("Stage-before mismatch")

        tool = event["tool"]
        argument = event["argument"]

        if event["source"] not in (
            "attacker_script",
            "trusted_script",
            "trusted_recovery",
            "original_task_check",
        ):
            raise ValueError("Unknown source")

        if event["source"] == "attacker_script":
            if (
                not scenario["attack"]["present"]
                or argument == scenario[
                    "trusted_task"
                ]["target"]
            ):
                raise ValueError(
                    "Invalid attacker action label"
                )

        decision = fresh.execute(tool, argument)

        actual = {
            "stage_after": fresh.stage,
            "allowed": decision.allowed,
            "executed": decision.executed,
            "completed": decision.completed,
            "reason": decision.reason,
        }

        for field, value in actual.items():
            if event[field] != value:
                raise ValueError(
                    f"Decision mismatch: {field}"
                )

        previous = event["event_hash"]

    if episode["final_hash"] != previous:
        raise ValueError("Final hash mismatch")

    summary = fresh.summary()

    expected_summary = {
        "steps": len(events),
        "completed": fresh.completed,
        "blocks": summary["blocked_actions"],
        "unauthorized_executions":
            summary["unauthorized_executions"],
        "attack_attempts": sum(
            e["source"] == "attacker_script"
            for e in events
        ),
    }

    for field, value in expected_summary.items():
        if episode[field] != value:
            raise ValueError(
                f"Episode summary mismatch: {field}"
            )

    if episode["unauthorized_executions"] != 0:
        raise ValueError("Unauthorized execution detected")

    # Reproduce the exact strategy, not merely
    # a plausible simulator event sequence.
    expected = run_episode(
        scenario,
        episode["strategy"],
        repeat=episode["repeat"],
        max_steps=episode["max_steps"],
    )

    if episode != expected:
        raise ValueError(
            "Episode differs from canonical strategy"
        )

    return True


def summarize(episodes):
    results = {
        "episodes": len(episodes),
        "attack_episodes": sum(
            x["attack_present"] for x in episodes
        ),
        "authorized_task_episodes": sum(
            x["original_task_authorized"]
            for x in episodes
        ),
        "completions": sum(
            x["completed"] for x in episodes
        ),
        "scripted_attack_attempts": sum(
            x["attack_attempts"]
            for x in episodes
        ),
        "blocked_actions": sum(
            x["blocks"] for x in episodes
        ),
        "unauthorized_executions": sum(
            x["unauthorized_executions"]
            for x in episodes
        ),
        "stop_reasons": dict(Counter(
            x["stop_reason"] for x in episodes
        )),
        "strategies": {},
    }

    for strategy in STRATEGIES:
        rows = [
            x for x in episodes
            if x["strategy"] == strategy
        ]

        results["strategies"][strategy] = {
            "episodes": len(rows),
            "completions": sum(
                x["completed"] for x in rows
            ),
            "attack_attempts": sum(
                x["attack_attempts"]
                for x in rows
            ),
            "blocked_actions": sum(
                x["blocks"] for x in rows
            ),
            "unauthorized_executions": sum(
                x["unauthorized_executions"]
                for x in rows
            ),
        }

    return results


def write_jsonl(path, records):
    path.write_text(
        "".join(canonical(r) + "\n" for r in records),
        encoding="utf-8",
    )


def run_experiment(repeats=3, max_steps=MAX_STEPS,
                   output_dir=RESULTS):
    if repeats < 1 or max_steps < 1:
        raise ValueError("Invalid benchmark configuration")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = attacks.make_dataset()

    episodes = []

    for scenario in scenarios:
        for repeat in range(1, repeats + 1):
            for strategy in STRATEGIES:
                row = run_episode(
                    scenario, strategy,
                    repeat=repeat,
                    max_steps=max_steps,
                )

                verify_episode(row, scenario)
                episodes.append(row)

    report = {
        "module": 2,
        "chapter": "2.3",
        "evidence_type":
            "Deterministic scripted stateful simulator",
        "scenario_count": len(scenarios),
        "repeats": repeats,
        "max_steps": max_steps,
        "summary": summarize(episodes),
    }

    write_jsonl(
        output_dir / "episode_traces.jsonl",
        episodes,
    )

    report["traces_sha256"] = hashlib.sha256(
        (output_dir / "episode_traces.jsonl").read_bytes()
    ).hexdigest()

    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 2 C3 SUMMARY")
    print(json.dumps(report, indent=2))
    print("\nSaved:", output_dir)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    run_experiment(
        repeats=args.repeats,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
