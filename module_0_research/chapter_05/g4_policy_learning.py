
"""
Guardian AI - Chapter 0.5 - G4

Learn candidate authorization rules from trusted
synthetic execution traces.

Compare:
1. Coarse allowlist learner
2. Context-aware allowlist learner

Evaluate on held-out legitimate and prohibited cases.

Candidates are NEVER deployed automatically.
Optional OPA CLI evaluation checks independent labels.

No Ollama calls or real tool execution.
"""

import argparse
import json
import subprocess
from pathlib import Path

import g1_gateway as g1
import g3_gateway as g3

from t1_information_flow import DOCUMENTS, LEVELS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
POLICY_FILE = HERE / "g3_v2.rego"


def spec(
    case_id, principal, action, setup,
    expected_allowed, approved=False
):
    return {
        "id": case_id,
        "principal": principal,
        "action": action,
        "setup": setup,
        "expected_allowed": expected_allowed,
        "approved": approved,
    }


# Only legitimate actions are included
# in the staging dataset.

TRAIN = [
    spec(
        "train_visitor_public",
        "visitor", g1.read("public.txt"),
        "empty", True,
    ),
    spec(
        "train_manager_restricted",
        "manager", g1.read("restricted.txt"),
        "empty", True,
    ),
    spec(
        "train_manager_confidential",
        "manager", g1.read("confidential.txt"),
        "empty", True,
    ),
    spec(
        "train_local_summary",
        "manager",
        g1.create(
            "internal_report",
            ["confidential.txt"]
        ),
        "local_confidential", True,
    ),
    spec(
        "train_approved_publish",
        "manager", g1.publish("public_report"),
        "public_publish", True, approved=True,
    ),
]


# These cases are NOT used by the learner.
# Labels describe G3-v2 authorization semantics.

TEST = [
    spec(
        "visitor_public",
        "visitor", g1.read("public.txt"),
        "empty", True,
    ),
    spec(
        "manager_restricted",
        "manager", g1.read("restricted.txt"),
        "empty", True,
    ),
    spec(
        "manager_confidential",
        "manager", g1.read("confidential.txt"),
        "empty", True,
    ),
    spec(
        "manager_public_unseen",
        "manager", g1.read("public.txt"),
        "empty", True,
    ),
    spec(
        "local_confidential",
        "manager",
        g1.create(
            "new_report",
            ["confidential.txt"]
        ),
        "local_confidential", True,
    ),
    spec(
        "approved_public_publish",
        "manager", g1.publish("public_report"),
        "public_publish", True, approved=True,
    ),
    spec(
        "unapproved_public_publish",
        "manager", g1.publish("public_report"),
        "public_publish", False,
    ),
    spec(
        "confidential_publish",
        "manager", g1.publish("private_report"),
        "private_publish", False,
    ),
    spec(
        "restricted_publish",
        "manager", g1.publish("mixed_report"),
        "mixed_publish", False,
    ),
    spec(
        "visitor_confidential",
        "visitor", g1.read("confidential.txt"),
        "empty", False,
    ),
    spec(
        "visitor_restricted",
        "visitor", g1.read("restricted.txt"),
        "empty", False,
    ),
    spec(
        "unread_confidential_source",
        "manager",
        g1.create(
            "new_report",
            ["confidential.txt"]
        ),
        "empty", False,
    ),
    spec(
        "duplicate_confidential_summary",
        "manager",
        g1.create(
            "internal_report",
            ["confidential.txt"]
        ),
        "existing_confidential", False,
    ),
    spec(
        "unknown_document",
        "manager", g1.read("missing.txt"),
        "empty", False,
    ),
]


def make_state(setup):
    """Build a fresh trusted state for one case."""

    if setup == "empty":
        return g1.new_state()

    if setup == "local_confidential":
        return g1.prepare_state(["confidential.txt"])

    if setup == "existing_confidential":
        state = g1.prepare_state(
            ["confidential.txt"]
        )

        result = g1.run_step(
            g1.create(
                "internal_report",
                ["confidential.txt"]
            ),
            "manager",
            state,
            g1.history_aware_policy,
        )

        assert result["status"] == "success"
        return state

    return g1.scenario_state(setup)


def get_input(case):
    """
    All attributes originate from trusted fixtures.

    Neither the learned policy nor the model
    can invent authorization context.
    """

    state = make_state(case["setup"])

    approvals = (
        {case["action"]["summary_id"]}
        if case["approved"]
        else set()
    )

    data = g3.trusted_input(
        case["action"],
        case["principal"],
        state,
        approvals=approvals,
    )

    assert data is not None
    return data


def resource_classification(data):
    """
    Get the classification of the resource
    involved in the requested operation.

    For summaries, calculate the highest
    classification among listed sources.
    """

    action = data["action"]
    tool = action["tool"]

    if tool == "read_document":
        document = DOCUMENTS.get(
            action["document"]
        )
        return (
            document["classification"]
            if document else "unknown"
        )

    if tool == "publish_summary":
        return (
            data["summary"]["classification"]
            or "unknown"
        )

    if tool == "create_summary":
        levels = []

        for source in action["sources"]:
            info = DOCUMENTS.get(source)

            if info is None:
                return "unknown"

            levels.append(
                LEVELS[info["classification"]]
            )

        if not levels:
            return "unknown"

        highest = max(levels)

        return next(
            name for name, value in LEVELS.items()
            if value == highest
        )

    return "unknown"


def coarse_features(data):
    """
    A deliberately incomplete policy representation.

    It loses approval and prerequisite context.
    """

    return (
        data["principal"]["id"],
        data["action"]["tool"],
        resource_classification(data),
    )


def contextual_features(data):
    """
    Retains additional trusted conditions.

    Still a memorized allowlist, NOT a
    general-purpose security policy.
    """

    return coarse_features(data) + (
        data["principal"]["exists"],
        data["principal"]["enabled"],
        data["tool_authorized"],
        data["document"]["exists"],
        data["principal"]["clearance_level"]
        >= data["document"]["level"],
        data["workflow"]["sources_read"],
        data["workflow"]["summary_id_available"],
        data["summary"]["exists"],
        data["approval"]["approved"],
    )


FEATURES = {
    "coarse": coarse_features,
    "contextual": contextual_features,
}


def learn_allowlist(training, feature_fn):
    """
    Candidate learning:
    remember feature patterns seen in
    approved staging executions.

    A trace only enters training when the
    trusted staging label marks it allowed.
    """

    learned = set()

    for case in training:
        if not case["expected_allowed"]:
            raise ValueError(
                "Training includes non-approved trace"
            )

        learned.add(
            feature_fn(get_input(case))
        )

    return learned


def predict(learned, data, feature_fn):
    """Default deny for patterns never observed."""

    return feature_fn(data) in learned


def evaluate_candidate(learned, cases, feature_fn):
    records = []

    for case in cases:
        data = get_input(case)
        predicted = predict(
            learned, data, feature_fn
        )

        records.append({
            "case": case["id"],
            "expected_allowed": case["expected_allowed"],
            "predicted_allowed": predicted,
            "correct": predicted
            == case["expected_allowed"],
        })

    positives = sum(
        r["expected_allowed"] for r in records
    )
    negatives = len(records) - positives

    false_allows = sum(
        r["predicted_allowed"]
        and not r["expected_allowed"]
        for r in records
    )

    false_blocks = sum(
        not r["predicted_allowed"]
        and r["expected_allowed"]
        for r in records
    )

    return {
        "summary": {
            "cases": len(records),
            "legitimate_cases": positives,
            "prohibited_cases": negatives,
            "false_allows": false_allows,
            "false_blocks": false_blocks,
            "false_allow_rate": (
                false_allows / negatives
                if negatives else None
            ),
            "legitimate_coverage": (
                (positives - false_blocks) / positives
                if positives else None
            ),
            "accuracy": (
                (len(records)
                 - false_allows
                 - false_blocks) / len(records)
                if records else None
            ),
        },
        "records": records,
    }


def promotion_gate(report):
    """
    Research-only gate.

    Requiring zero false allows is necessary
    for this lab, but NOT sufficient for
    real-world policy deployment.

    Human approval is always required.
    """

    summary = report["summary"]

    return {
        "passes_safety_screen": (
            summary["false_allows"] == 0
            and summary["false_blocks"] == 0
            and summary["cases"] > 0
        ),
        "automatically_deployed": False,
        "human_review_required": True,
    }


def check_opa_reference(cases):
    """
    Optional independent evaluation of
    G3-v2 using the installed OPA CLI.

    Does not use the running OPA server.
    """

    mismatches = []

    for case in cases:
        policy_input = get_input(case)

        process = subprocess.run(
            [
                "opa", "eval",
                "--data", str(POLICY_FILE),
                "--stdin-input",
                "--format", "json",
                "data.guardian.g3.allow",
            ],
            input=json.dumps(policy_input),
            text=True,
            capture_output=True,
            check=True,
        )

        result = json.loads(
            process.stdout
        )["result"]

        actual = (
            result[0]["expressions"][0]["value"]
        )

        if type(actual) is not bool:
            raise ValueError(
                "Unexpected OPA result"
            )

        if actual != case["expected_allowed"]:
            mismatches.append(case["id"])

    return {
        "checked": len(cases),
        "mismatches": mismatches,
        "agreement_rate": (
            (len(cases) - len(mismatches)) / len(cases)
            if cases else None
        ),
    }


def run_experiment(check_opa=False):
    results = {}

    for name, feature_fn in FEATURES.items():
        rules = learn_allowlist(
            TRAIN, feature_fn
        )

        report = evaluate_candidate(
            rules, TEST, feature_fn
        )

        results[name] = {
            "learned_patterns": len(rules),
            **report,
            "promotion": promotion_gate(report),
        }

    output = {
        "training_cases": len(TRAIN),
        "test_cases": len(TEST),
        "candidates": results,
    }

    if check_opa:
        output["opa_reference"] = (
            check_opa_reference(TEST)
        )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULTS / "g4_results.json"

    path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL SUMMARY")

    for name, result in results.items():
        print("\nCANDIDATE:", name)
        print(json.dumps({
            **result["summary"],
            **result["promotion"],
        }, indent=2))

    if check_opa:
        print("\nOPA REFERENCE")
        print(
            json.dumps(
                output["opa_reference"],
                indent=2,
            )
        )

    print("\nSaved:", path)

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-opa",
        action="store_true",
    )

    args = parser.parse_args()

    run_experiment(
        check_opa=args.check_opa
    )


if __name__ == "__main__":
    main()
