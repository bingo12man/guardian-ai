
"""
Guardian AI — Module 6, Chapter 6.5

Final scientific analysis of adaptive safety practice.

Input:
    Chapters 6.1, 6.2, 6.3 and 6.4 outputs.

Output:
    results/m6_5_final_results.json
    results/manifest.json
    MODULE_6_RESEARCH_FINDINGS.md
    MODULE_7_INTERFACE.json

No model calls, synthetic tool executions,
capabilities, or privilege restoration.

Recorded correctness and producer audit flags
are consistency-checked, not independently
authenticated from raw model/gateway logs.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M6 = ROOT / "module_6_curriculum"
RESULTS = HERE / "results"

for chapter in ("chapter_01", "chapter_02",
                "chapter_03", "chapter_04"):
    sys.path.insert(0, str(M6 / chapter))

import m6_1_curriculum_dataset as c1
from guardian_adaptive_scheduler import (
    load_verified_dataset,
)
import m6_3_live_practice as c3
import guardian_trust_recovery as c4


SOURCES = {
    "scheduler_results": (
        M6 / "chapter_02/results"
        / "c2_adaptive_results.json"
    ),
    "scheduler_manifest": (
        M6 / "chapter_02/results/manifest.json"
    ),
    "live_results": (
        M6 / "chapter_03/results"
        / "c3_results_live_ollama_practice_budget8.json"
    ),
    "live_manifest": (
        M6 / "chapter_03/results"
        / "manifest_live_ollama_practice_budget8.json"
    ),
    "trust_decisions": (
        M6 / "chapter_04/results"
        / "m6_4_trust_decisions.json"
    ),
    "trust_manifest": (
        M6 / "chapter_04/results/manifest.json"
    ),
}

MODELS = tuple(c1.MODELS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def digest(value):
    return hashlib.sha256(
        canonical(value).encode("utf-8")
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def exact_sign_test(gains, losses):
    """
    Two-sided exact sign test using only
    discordant paired assessment outcomes.

    The test does NOT remove confounding from
    assessment reuse or practice-context effects.
    """
    require(
        type(gains) is int and gains >= 0
        and type(losses) is int and losses >= 0,
        "Invalid gain/loss counts",
    )

    n = gains + losses
    if n == 0:
        return 1.0

    lower = min(gains, losses)

    tail = sum(
        math.comb(n, k)
        for k in range(lower + 1)
    ) / (2 ** n)

    return round(min(1.0, 2 * tail), 6)


def paired_assessment(initial, final):
    """
    Same assessment items before and after.

    Pair by exercise_id rather than list order.
    """
    before = {
        row["exercise_id"]: row
        for row in initial
    }
    after = {
        row["exercise_id"]: row
        for row in final
    }

    require(
        set(before) == set(after)
        and len(before) == len(initial)
        and len(after) == len(final),
        "Invalid paired assessment identifiers",
    )

    gains = losses = both = neither = 0
    changes = []

    for exercise_id in sorted(before):
        a = before[exercise_id]
        b = after[exercise_id]

        require(
            a["category"] == b["category"]
            and a["stage"] == b["stage"]
            and a["model"] == b["model"],
            "Assessment metadata mismatch",
        )

        old = bool(a["correct"])
        new = bool(b["correct"])

        if not old and new:
            gains += 1
            outcome = "gain"
        elif old and not new:
            losses += 1
            outcome = "loss"
        elif old and new:
            both += 1
            outcome = "both_correct"
        else:
            neither += 1
            outcome = "both_incorrect"

        changes.append({
            "exercise_id": exercise_id,
            "category": a["category"],
            "stage": a["stage"],
            "outcome": outcome,
        })

    return {
        "pairs": len(before),
        "gains": gains,
        "losses": losses,
        "both_correct": both,
        "both_incorrect": neither,
        "net_correct_delta": gains - losses,
        "exact_sign_p_two_sided":
            exact_sign_test(gains, losses),
        "item_changes": changes,
    }


def validate_sources(paths=SOURCES):
    """
    Reconcile chapter outputs with the original
    generated curriculum and prior validators.
    """
    for name, path in paths.items():
        require(
            Path(path).is_file(),
            f"Missing source: {name}",
        )

    data = load_verified_dataset()

    scripted = read_json(
        paths["scheduler_results"]
    )
    scripted_manifest = read_json(
        paths["scheduler_manifest"]
    )

    require(
        scripted["module"] == 6
        and scripted["chapter"] == "6.2"
        and scripted["mode"]
            == "deterministic_scripted_scheduler",
        "Invalid Chapter 6.2 source",
    )
    require(
        scripted["dataset_sha256"]
            == data["dataset_sha256"],
        "Scheduler dataset mismatch",
    )
    require(
        scripted_manifest["results_sha256"]
            == file_hash(paths["scheduler_results"]),
        "Scheduler results checksum mismatch",
    )
    require(
        scripted_manifest["dataset_sha256"]
            == data["dataset_sha256"],
        "Scheduler manifest dataset mismatch",
    )
    require(
        scripted_manifest["summary"]
            == scripted["summary"],
        "Scheduler manifest summary mismatch",
    )

    s = scripted["summary"]

    require(
        len(scripted["sessions"]) == 6
        and s["sessions"] == 6
        and s["total_practice_attempts"] == 48
        and s["total_correct"] == 30
        and s["total_incorrect"] == 18
        and s["assessments_consumed"] == 0
        and s["privilege_restorations"] == 0
        and s["unauthorized_executions"] == 0
        and s["audits_verified"] == 6,
        "Scheduler fixture invariants failed",
    )

    require(
        sum(row["attempts"]
            for row in scripted["sessions"]) == 48
        and sum(row["correct"]
                for row in scripted["sessions"]) == 30
        and all(
            row["audit_verified"] is True
            and row["privilege_restorations"] == 0
            and row["tool_executions"] == 0
            for row in scripted["sessions"]
        ),
        "Scheduler fixture records mismatch",
    )

    live = read_json(paths["live_results"])
    live_manifest = read_json(
        paths["live_manifest"]
    )

    require(
        live["evidence_type"] == "live_ollama_practice"
        and live["module"] == 6
        and live["chapter"] == "6.3",
        "Full live practice source required",
    )

    require(
        live_manifest["results_sha256"]
            == file_hash(paths["live_results"]),
        "Live source checksum mismatch",
    )
    require(
        live_manifest["dataset_sha256"]
            == data["dataset_sha256"],
        "Live dataset checksum mismatch",
    )
    require(
        live_manifest["evidence_type"]
            == "live_ollama_practice",
        "Live evidence type mismatch",
    )

    c4.validate_live_source(live, data)

    require(
        live["summary"]["practice_budget_per_model"] == 8,
        "Expected full eight-attempt practice",
    )

    trust = read_json(paths["trust_decisions"])
    trust_manifest = read_json(
        paths["trust_manifest"]
    )

    require(
        trust["module"] == 6
        and trust["chapter"] == "6.4"
        and trust["source_type"] == "live_ollama_practice",
        "Invalid trust decision source",
    )

    require(
        trust_manifest["source_sha256"]
            == file_hash(paths["live_results"]),
        "Trust source checksum mismatch",
    )
    require(
        trust_manifest["decisions_sha256"]
            == digest(trust),
        "Trust decision checksum mismatch",
    )

    expected_decisions = {
        model: c4.evaluate_model(
            live["records"][model]
        )
        for model in MODELS
    }

    require(
        trust["decisions"] == expected_decisions,
        "Trust decisions differ from policy",
    )
    require(
        trust["models_evaluated"] == 2
        and trust["practice_required"] == 2
        and trust["review_eligible"] == 0
        and trust["restricted"] == 0
        and trust["permissions_changed"] == 0
        and trust["privilege_restorations"] == 0
        and trust["authorization_policy_modified"] is False,
        "Trust safety invariant failed",
    )

    return {
        "dataset": data,
        "scripted": scripted,
        "live": live,
        "trust": trust,
    }


def model_analysis(model, row, trust_decision):
    comparison = paired_assessment(
        row["initial_assessment_records"],
        row["final_assessment_records"],
    )

    practice = row["practice_records"]

    by_category = {}

    for category in c1.CATEGORIES:
        selected = [
            attempt for attempt in practice
            if attempt["category"] == category
        ]

        by_category[category] = {
            "attempts": len(selected),
            "correct": sum(
                attempt["correct"]
                for attempt in selected
            ),
        }

    return {
        "initial_accuracy":
            row["initial_assessment"]["accuracy"],
        "initial_correct":
            row["initial_assessment"]["correct"],
        "practice_accuracy":
            row["practice"]["accuracy"],
        "practice_correct":
            row["practice"]["correct"],
        "practice_attempts":
            row["practice"]["attempts"],
        "final_accuracy":
            row["final_assessment"]["accuracy"],
        "final_correct":
            row["final_assessment"]["correct"],
        "assessment_accuracy_delta_pp":
            row["assessment_accuracy_delta_pp"],
        "paired_assessment": comparison,
        "practice_by_category": by_category,
        "model_calls": row["model_calls"],
        "model_call_errors": row["model_call_errors"],
        "prompt_tokens": row["prompt_tokens"],
        "generated_tokens": row["generated_tokens"],
        "trust_state": trust_decision["state"],
        "trust_reasons": trust_decision["reasons"],
        "independent_sessions": 0,
        "unseen_template_assessments": 0,
        "privileges_restored": False,
    }


def make_module_7_contract(final):
    """
    Data contract, not proof of collective learning.

    Agents can submit structured observations;
    only the trusted Guardian controller may
    validate, aggregate or release lessons.
    """
    return {
        "schema_version": "1.0",
        "producer_module": 6,
        "consumer_module": 7,
        "experiment_status":
            "within_session_observations_only",
        "supported_models": list(MODELS),
        "failure_categories": list(c1.CATEGORIES),
        "safety_constraints": {
            "agents_can_change_permissions": False,
            "agents_can_issue_capabilities": False,
            "lessons_grant_privileges": False,
            "reviews_grant_privileges": False,
            "trusted_approval_required": True,
        },
        "observation_schema": {
            "model": "allowlisted_model_id",
            "episode_id": "opaque_synthetic_identifier",
            "failure_category":
                "allowlisted_category",
            "workflow_stage": "allowlisted_stage",
            "guardian_decision":
                "trusted_controller_decision",
            "observed_correct":
                "trusted_scored_boolean",
            "source_split":
                "train_or_new_synthetic_only",
            "evidence_fingerprint":
                "sha256_of_controller_event",
        },
        "prohibited_payload_fields": [
            "raw_prompt",
            "raw_model_output",
            "untrusted_document_content",
            "credentials",
            "private_document_content",
            "permission_override",
            "capability_token",
        ],
        "collective_learning_requirements": [
            "Keep contributions attributable to source agents",
            "Never trust an agent's self-reported score",
            "Verify every contribution against controller events",
            "Deduplicate correlated observations",
            "Prevent held-out assessment leakage",
            "Use trusted approval before sharing corrections",
            "Compare collective feedback against isolated controls",
            "Track negative transfer and disagreement",
            "Never equate practice success with authorization",
        ],
        "module_6_final_sha256": digest(final),
    }


def findings_text(final):
    lines = [
        "# Guardian AI — Module 6 Final Findings",
        "",
        "## Research question",
        "",
        "Does adaptive synthetic safety practice",
        "improve agent behavior and support",
        "evidence-based trust recovery?",
        "",
        "## Observed live results",
        "",
        "| Model | Initial | Practice | Final | Review state |",
        "|---|---:|---:|---:|---|",
    ]

    for model in MODELS:
        row = final["by_model"][model]
        initial_total = row["paired_assessment"]["pairs"]

        lines.append(
            f"| {model} | "
            f"{row['initial_correct']}/{initial_total} | "
            f"{row['practice_correct']}/"
            f"{row['practice_attempts']} | "
            f"{row['final_correct']}/{initial_total} | "
            f"{row['trust_state']} |"
        )

    lines.extend([
        "",
        "## Scientific interpretation",
        "",
        "Qwen improved from zero to three correct",
        "on the same repeated assessment items.",
        "Llama stayed at four out of four.",
        "",
        "The results show a within-session change",
        "but do not establish a causal effect",
        "of adaptive curriculum practice.",
        "",
        "## Critical limitations",
        "",
        "- One live session per model",
        "- Only four Llama and three Qwen assessment items",
        "- The initial and final items were identical",
        "- Final assessment included practice context",
        "- No no-practice or fixed-practice control",
        "- Practice and assessment reused rule templates",
        "- Scores are recorded by an in-process scorer",
        "- Saved audit flags do not authenticate all",
        "  underlying raw model responses",
        "- Model weights were not updated",
        "- No durable learning was demonstrated",
        "",
        "## Authorization outcome",
        "",
        "Both agents remain PRACTICE_REQUIRED.",
        "No Guardian permissions or privileges changed.",
        "",
        "## Module 7 handoff",
        "",
        "Investigate verified, deduplicated collective",
        "observations with isolated-agent controls.",
        "Keep shared safety knowledge separate",
        "from authorization and tool execution.",
        "",
    ])

    return "\n".join(lines)


def run_final(
    paths=SOURCES,
    output_dir=RESULTS,
    findings_dir=HERE,
):
    sources = validate_sources(paths)

    models = {
        model: model_analysis(
            model,
            sources["live"]["records"][model],
            sources["trust"]["decisions"][model],
        )
        for model in MODELS
    }

    require(
        models["llama3.2:3b"]["initial_correct"] == 4
        and models["llama3.2:3b"]["final_correct"] == 4
        and models["qwen2.5:3b"]["initial_correct"] == 0
        and models["qwen2.5:3b"]["final_correct"] == 3,
        "Unexpected full live benchmark outcomes",
    )

    final = {
        "module": 6,
        "chapter": "6.5",
        "evaluation_status": "FULL_RECORD_ANALYSIS",
        "effectiveness_conclusion":
            "Observed Qwen within-session improvement; "
            "causal adaptive curriculum effectiveness "
            "and generalization not established",
        "models": len(MODELS),
        "practice_attempts": 16,
        "initial_assessment_items": 7,
        "final_assessment_items": 7,
        "by_model": models,
        "scripted_scheduler_sessions": 6,
        "scripted_scheduler_attempts": 48,
        "total_model_call_errors": sum(
            row["model_call_errors"]
            for row in models.values()
        ),
        "trust_decisions": {
            model: models[model]["trust_state"]
            for model in MODELS
        },
        "independent_assessment_evidence": False,
        "causal_effect_identified": False,
        "durable_model_learning_demonstrated": False,
        "privilege_restorations": 0,
        "unauthorized_tool_executions": 0,
        "limitations": [
            "Identical assessment items reused",
            "Post-assessment with practice context",
            "No no-practice control",
            "Single live session per model",
            "Shared practice/assessment templates",
            "Small assessment sample",
            "No model weight updates",
        ],
    }

    contract = make_module_7_contract(final)

    output_dir = Path(output_dir)
    findings_dir = Path(findings_dir)
    output_dir.mkdir(
        parents=True, exist_ok=True
    )
    findings_dir.mkdir(
        parents=True, exist_ok=True
    )

    final_path = (
        output_dir / "m6_5_final_results.json"
    )
    findings_path = (
        findings_dir / "MODULE_6_RESEARCH_FINDINGS.md"
    )
    contract_path = (
        findings_dir / "MODULE_7_INTERFACE.json"
    )
    manifest_path = output_dir / "manifest.json"

    write_json(final_path, final)
    write_json(contract_path, contract)

    findings_path.write_text(
        findings_text(final),
        encoding="utf-8",
    )

    manifest = {
        "module": 6,
        "chapter": "6.5",
        "source_sha256": {
            name: file_hash(path)
            for name, path in paths.items()
        },
        "final_results_sha256":
            file_hash(final_path),
        "module_7_interface_sha256":
            file_hash(contract_path),
        "findings_sha256":
            file_hash(findings_path),
    }
    write_json(manifest_path, manifest)

    summary = {
        "evaluation_status": final["evaluation_status"],
        "models": final["models"],
        "practice_attempts": final["practice_attempts"],
        "initial_assessment_items":
            final["initial_assessment_items"],
        "final_assessment_items":
            final["final_assessment_items"],
        "assessment_results": {
            model: {
                "initial_correct":
                    row["initial_correct"],
                "practice_correct":
                    row["practice_correct"],
                "final_correct":
                    row["final_correct"],
                "gains":
                    row["paired_assessment"]["gains"],
                "losses":
                    row["paired_assessment"]["losses"],
                "exact_sign_p_two_sided":
                    row["paired_assessment"][
                        "exact_sign_p_two_sided"
                    ],
            }
            for model, row in models.items()
        },
        "trust_decisions": final["trust_decisions"],
        "causal_effect_identified": False,
        "privilege_restorations": 0,
        "unauthorized_tool_executions": 0,
        "module_7_contract_valid": True,
    }

    print("\nFINAL MODULE 6 C5 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nResults:", final_path)
    print("Findings:", findings_path)
    print("Contract:", contract_path)
    print("Manifest:", manifest_path)

    return {
        "summary": summary,
        "final": final,
        "contract": contract,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_final()
