
"""
Guardian AI — Module 2, Chapter 2.4

Validate:
- Synthetic dataset integrity
- Train/validation/test family isolation
- Independent authorization labels
- Attack variant trust boundaries
- Artifact SHA-256 fingerprints
- Hash-chained stateful episode replay
- Exact and normalized duplicate content

Does not call LLMs.
Does not perform real tool actions.
"""

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
RESULTS = HERE / "results"

sys.path.insert(0, str(MODULE / "chapter_01"))
sys.path.insert(0, str(MODULE / "chapter_02"))
sys.path.insert(0, str(MODULE / "chapter_03"))

import m2_1_environment as c1
import m2_2_attack_dataset as c2
import m2_3_stateful_simulation as c3


INPUTS = {
    "C1": MODULE / "chapter_01/results",
    "C2": MODULE / "chapter_02/results",
    "C3": MODULE / "chapter_03/results",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    require(
        bool(lines),
        f"Empty JSONL: {path}"
    )

    return [json.loads(line) for line in lines]


def compare_file_records(path, expected):
    """
    Compare complete records, including order.
    Reject silent edits even if labels still look
    superficially valid.
    """
    actual = read_jsonl(path)

    require(
        actual == expected,
        f"Content mismatch: {path}"
    )

    return len(actual)


def split_family(scenario_id):
    """
    C2 IDs contain "__attack_type".
    The base scenario ID is everything before it.
    """
    return scenario_id.split("__", 1)[0]


def check_split_integrity(records):
    assignments = defaultdict(set)
    split_ids = defaultdict(set)

    for record in records:
        identifier = record["scenario_id"]
        family = split_family(identifier)
        split = record["split"]

        assignments[family].add(split)
        split_ids[split].add(identifier)

    leaked = {
        family: sorted(splits)
        for family, splits in assignments.items()
        if len(splits) != 1
    }

    require(
        not leaked,
        f"Cross-split base-scenario leakage: {leaked}"
    )

    train = split_ids["train"]
    validation = split_ids["validation"]
    test = split_ids["test"]

    require(
        not (train & validation)
        and not (train & test)
        and not (validation & test),
        "Scenario IDs overlap across splits",
    )

    # Also verify broader scenario families using
    # the canonical Chapter 2.1 family assignment.
    c1.validate_dataset(records)

    return {
        "base_scenarios": len(assignments),
        "cross_split_base_scenario_leaks": 0,
        "split_counts": {
            name: len(split_ids[name])
            for name in ("train", "validation", "test")
        },
    }


def check_ground_truth(records):
    authorized = 0
    unauthorized = 0

    for record in records:
        task = record["trusted_task"]
        truth = record["ground_truth"]

        calculated = c1.task_authorized(task)
        permitted = c1.allowed_resources(
            task["requester"]
        )

        require(
            truth["original_task_authorized"]
            == calculated,
            "Incorrect authorization label: "
            + record["scenario_id"]
        )

        require(
            truth["allowed_resources"]
            == permitted,
            "Incorrect allowed-resource label: "
            + record["scenario_id"]
        )

        require(
            truth["expected_target"]
            == task["target"],
            "Incorrect expected target"
        )

        authorized += int(calculated)
        unauthorized += int(not calculated)

    return {
        "authorized": authorized,
        "unauthorized": unauthorized,
        "label_mismatches": 0,
    }


def check_attack_variants(records):
    originals = {
        row["scenario_id"]: row
        for row in c1.generate_dataset()
    }

    attack_count = 0

    for row in records:
        original_id = split_family(
            row["scenario_id"]
        )

        require(
            original_id in originals,
            "Unknown original scenario"
        )

        original = originals[original_id]

        require(
            row["trusted_task"]
            == original["trusted_task"],
            "Trusted task tampering"
        )

        require(
            row["ground_truth"]
            == original["ground_truth"],
            "Ground truth tampering"
        )

        require(
            row["split"] == original["split"],
            "Variant moved across splits"
        )

        if row["attack"]["present"]:
            attack_count += 1

            technique = row["attack"]["technique"]

            require(
                technique in c2.ATTACK_TYPES,
                "Unknown attack technique"
            )

            require(
                row["untrusted_inputs"][0]["source"]
                == c2.ATTACK_SOURCES[technique],
                "Wrong attack source"
            )

    return {
        "attack_variants": attack_count,
        "trusted_task_mutations": 0,
        "ground_truth_mutations": 0,
    }


def normalized_text(text):
    """
    Simple near-duplicate diagnostic.

    Removes synthetic file names and normalizes
    whitespace. This is not a semantic similarity
    or embedding-based leakage detector.
    """
    text = text.lower()
    text = re.sub(
        r"\b[a-z_]+\.txt\b",
        "<DOCUMENT>",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def duplicate_diagnostics(records):
    exact = defaultdict(set)
    normalized = defaultdict(set)

    for row in records:
        text = "\n".join(
            item["content"]
            for item in row["untrusted_inputs"]
        )

        exact[text].add(row["split"])
        normalized[
            normalized_text(text)
        ].add(row["split"])

    return {
        "cross_split_exact_text_groups": sum(
            len(splits) > 1
            for splits in exact.values()
        ),
        "cross_split_normalized_text_groups": sum(
            len(splits) > 1
            for splits in normalized.values()
        ),
        "template_reuse_warning": (
            "Shared attack templates may appear "
            "across splits even when scenario "
            "families are isolated."
        ),
    }


def check_artifacts(inputs=INPUTS):
    p1 = Path(inputs["C1"])
    p2 = Path(inputs["C2"])
    p3 = Path(inputs["C3"])

    m1 = read_json(p1 / "manifest.json")
    m2 = read_json(p2 / "manifest.json")
    m3 = read_json(p3 / "manifest.json")

    originals = c1.generate_dataset()
    attacks = c2.make_dataset()

    require(
        m1["dataset_sha256"] ==
        sha256(p1 / "scenarios_all.jsonl"),
        "C1 stored dataset hash mismatch"
    )

    require(
        m2["dataset_sha256"] ==
        sha256(p2 / "scenarios_all.jsonl"),
        "C2 stored dataset hash mismatch"
    )

    require(
        m3["traces_sha256"] ==
        sha256(p3 / "episode_traces.jsonl"),
        "C3 stored trace hash mismatch"
    )

    compare_file_records(
        p1 / "scenarios_all.jsonl",
        originals,
    )

    compare_file_records(
        p2 / "scenarios_all.jsonl",
        attacks,
    )

    for prefix, root, records in (
        ("C1", p1, originals),
        ("C2", p2, attacks),
    ):
        for split in (
            "train", "validation", "test"
        ):
            subset = [
                r for r in records
                if r["split"] == split
            ]

            compare_file_records(
                root / f"{split}.jsonl",
                subset,
            )

    c1.validate_dataset(originals)
    c2.validate_attack_dataset(attacks, originals)

    require(
        m1["scenario_count"] == len(originals),
        "C1 manifest scenario count mismatch"
    )

    require(
        m2["scenario_count"] == len(attacks),
        "C2 manifest scenario count mismatch"
    )

    require(
        m3["scenario_count"] == len(attacks),
        "C3 manifest scenario count mismatch"
    )

    return {
        "C1": m1,
        "C2": m2,
        "C3": m3,
        "scenarios": attacks,
    }


def check_traces(
    episodes,
    scenarios,
    repeats=3,
    max_steps=5,
):
    by_id = {
        row["scenario_id"]: row
        for row in scenarios
    }

    expected_keys = {
        (sid, strategy, repeat)
        for sid in by_id
        for strategy in c3.STRATEGIES
        for repeat in range(1, repeats + 1)
    }

    observed_keys = []

    for episode in episodes:
        key = (
            episode["scenario_id"],
            episode["strategy"],
            episode["repeat"],
        )

        observed_keys.append(key)

        require(
            episode["max_steps"] == max_steps,
            "Wrong step budget"
        )

        scenario = by_id.get(
            episode["scenario_id"]
        )

        require(
            scenario is not None,
            "Unknown trace scenario"
        )

        c3.verify_episode(
            episode, scenario
        )

    require(
        len(observed_keys) == len(expected_keys)
        and len(set(observed_keys)) == len(observed_keys)
        and set(observed_keys) == expected_keys,
        "Missing, duplicated or extra episodes"
    )

    return {
        "episodes_verified": len(episodes),
        "unauthorized_executions": sum(
            ep["unauthorized_executions"]
            for ep in episodes
        ),
        "completions": sum(
            ep["completed"] for ep in episodes
        ),
        "attack_attempts": sum(
            ep["attack_attempts"] for ep in episodes
        ),
        "blocked_actions": sum(
            ep["blocks"] for ep in episodes
        ),
    }


def run_analysis(
    inputs=INPUTS,
    output_dir=RESULTS,
):
    data = check_artifacts(inputs)
    scenarios = data["scenarios"]

    p3 = Path(inputs["C3"])
    episodes = read_jsonl(
        p3 / "episode_traces.jsonl"
    )

    m3 = data["C3"]

    trace_results = check_traces(
        episodes,
        scenarios,
        repeats=m3["repeats"],
        max_steps=m3["max_steps"],
    )

    require(
        c3.summarize(episodes) == m3["summary"],
        "C3 manifest summary differs from traces"
    )

    require(
        trace_results["unauthorized_executions"] == 0,
        "Unauthorized synthetic execution found"
    )

    split_results = check_split_integrity(
        scenarios
    )
    labels = check_ground_truth(scenarios)
    variants = check_attack_variants(scenarios)
    duplicates = duplicate_diagnostics(scenarios)

    report = {
        "module": 2,
        "chapter": "2.4",
        "status": "validated",
        "checks": {
            "dataset_hashes": "PASS",
            "canonical_datasets": "PASS",
            "family_isolation": "PASS",
            "ground_truth": "PASS",
            "attack_integrity": "PASS",
            "trace_replay": "PASS",
            "trace_hashes": "PASS",
            "summary_consistency": "PASS",
        },
        "dataset": {
            "scenarios": len(scenarios),
            **split_results,
            **labels,
            **variants,
            **duplicates,
        },
        "traces": trace_results,
        "source_hashes": {
            "C1": data["C1"]["dataset_sha256"],
            "C2": data["C2"]["dataset_sha256"],
            "C3": data["C3"]["traces_sha256"],
        },
        "limitations": [
            "Scripted attacks, not live model attacks",
            "A synthetic document environment",
            "Shared attack templates across splits",
            "No semantic near-duplicate detection",
            "Hash chains are not externally anchored",
            "Current labels rely on one policy implementation",
        ],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    output = output_dir / "m2_4_integrity.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODULE 2 C4 SUMMARY")
    print(json.dumps({
        "status": report["status"],
        "checks": report["checks"],
        "dataset": report["dataset"],
        "traces": report["traces"],
    }, indent=2))

    print("\nSaved:", output)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS),
    )

    args = parser.parse_args()
    run_analysis(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
