"""Guardian AI chapter 0.2, Experiment E3: controlled feedback comparison.
Run from chapter_02 alongside reflexion_lab.py.
"""
import argparse
import json
import random
from pathlib import Path

from reflexion_lab import MODEL, TASKS, model_actor, model_reflect, evaluate

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def attempt(task_id, lessons, actor=model_actor):
    task = TASKS[task_id]
    outcome = actor(task["question"], list(lessons))
    return {
        "task_id": task_id,
        "lessons_used": list(lessons),
        "outcome": outcome,
        "evaluation": evaluate(task, outcome),
    }


def make_lesson(task_id, actor=model_actor, reflector=model_reflect):
    """Train on a *separate* attempt. Only retain lessons after failure."""
    trial = attempt(task_id, [], actor)
    lesson = ""
    if not trial["evaluation"]["success"]:
        lesson = reflector(TASKS[task_id]["question"], trial["outcome"], trial["evaluation"])
        if not isinstance(lesson, str):
            lesson = ""
        lesson = lesson.strip()[:400]
    return {"trial": trial, "lesson": lesson}


def compare(task_ids, lessons, repeats=3, seed=42, actor=model_actor):
    """Interleave control and treatment calls to reduce order confounding."""
    rng = random.Random(seed)
    schedule = [(task_id, condition, repeat)
                for task_id in task_ids
                for repeat in range(repeats)
                for condition in ("no_memory", "with_memory")]
    rng.shuffle(schedule)
    trials = []
    for task_id, condition, repeat in schedule:
        active_lessons = lessons if condition == "with_memory" else []
        trial = attempt(task_id, active_lessons, actor)
        trial.update({"condition": condition, "repeat": repeat})
        trials.append(trial)
        ev = trial["evaluation"]
        print(f"{task_id:8} {condition:11} source={ev['correct_source']} "
              f"number={ev.get('correct_number')} unit={ev.get('correct_format')} "
              f"success={ev['success']}")
    summary = {}
    for condition in ("no_memory", "with_memory"):
        subset = [t for t in trials if t["condition"] == condition]
        summary[condition] = {
            "attempts": len(subset),
            "successful": sum(t["evaluation"]["success"] for t in subset),
            "success_rate": (sum(t["evaluation"]["success"] for t in subset) / len(subset)) if subset else None,
        }
    return {"model": MODEL, "seed": seed, "lessons": lessons, "trials": trials, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    RESULTS.mkdir(parents=True, exist_ok=True)
    training = None
    if args.skip_training:
        path = RESULTS / "e3_lesson.json"
        if not path.exists():
            parser.error("Run without --skip-training first to generate a lesson file")
        lesson = json.loads(path.read_text(encoding="utf-8")).get("lesson", "")
        if not isinstance(lesson, str):
            parser.error("Invalid lesson file")
    else:
        training = make_lesson("expense")
        lesson = training["lesson"]
        (RESULTS / "e3_lesson.json").write_text(
            json.dumps({"lesson": lesson, "training": training}, indent=2), encoding="utf-8")
    print("Lesson:", lesson if lesson else "(none; training task succeeded)")
    result = compare(["expense", "revenue"], [lesson] if lesson else [], args.repeats, args.seed)
    result["training"] = training
    result["limitation"] = (
        "Small toy set; one training task overlaps evaluation tasks; randomized order "
        "does not establish a causal or generalizable improvement."
    )
    destination = RESULTS / "e3_results.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSummary:", json.dumps(result["summary"], indent=2))
    print("Saved:", destination)


if __name__ == "__main__":
    main()
