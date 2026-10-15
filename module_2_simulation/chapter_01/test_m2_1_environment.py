
import copy
import json

import pytest

import m2_1_environment as lab


def find_scenario(requester, target):
    return next(
        item for item in lab.generate_dataset()
        if item["trusted_task"]["requester"]
        == requester
        and item["trusted_task"]["target"]
        == target
    )


def test_document_catalog():
    assert len(lab.DOCUMENTS) == 4
    assert {
        d["classification"]
        for d in lab.DOCUMENTS.values()
    } == {"public", "confidential"}


def test_visitor_permissions():
    assert lab.can_access("visitor", "public.txt")
    assert lab.can_access("visitor", "guide.txt")
    assert not lab.can_access(
        "visitor", "confidential.txt"
    )
    assert not lab.can_access("visitor", "finance.txt")


def test_manager_permissions():
    for name in lab.DOCUMENTS:
        assert lab.can_access("manager", name)


def test_unknown_identity_denied():
    assert not lab.can_access("unknown", "public.txt")


def test_dataset_count_and_splits():
    records = lab.generate_dataset()

    assert len(records) == 12

    counts = {
        split: sum(
            row["split"] == split
            for row in records
        )
        for split in ("train", "validation", "test")
    }

    assert counts == {
        "train": 6,
        "validation": 2,
        "test": 4,
    }


def test_unique_scenario_ids():
    records = lab.generate_dataset()
    ids = [r["scenario_id"] for r in records]

    assert len(ids) == len(set(ids))


def test_authorization_labels_derived():
    records = lab.generate_dataset()

    for row in records:
        task = row["trusted_task"]
        truth = row["ground_truth"]

        assert truth["original_task_authorized"] == (
            lab.task_authorized(task)
        )
        assert truth["allowed_resources"] == (
            lab.allowed_resources(task["requester"])
        )


def test_family_split_is_stable():
    first = lab.generate_dataset()
    second = lab.generate_dataset()

    assert first == second

    assert (
        lab.canonical_json(first)
        == lab.canonical_json(second)
    )


def test_valid_public_workflow():
    row = find_scenario("visitor", "public.txt")
    env = lab.SyntheticEnvironment(row)

    search = env.execute("search_documents", "public.txt")
    read = env.execute("read_document", "public.txt")
    finish = env.execute("finish", "done")

    assert search.allowed
    assert read.allowed
    assert finish.completed
    assert env.completed
    assert len(env.events) == 3


def test_restricted_original_task_denied():
    row = find_scenario(
        "visitor", "confidential.txt"
    )
    env = lab.SyntheticEnvironment(row)

    decision = env.execute(
        "search_documents", "confidential.txt"
    )

    assert not decision.allowed
    assert not decision.executed
    assert not env.completed


def test_wrong_target_denied():
    row = find_scenario("visitor", "public.txt")
    env = lab.SyntheticEnvironment(row)

    decision = env.execute(
        "search_documents", "confidential.txt"
    )

    assert decision.reason == "TARGET_MISMATCH"
    assert decision.output is None
    assert not decision.executed


def test_manager_task_scope():
    row = find_scenario(
        "manager", "confidential.txt"
    )
    env = lab.SyntheticEnvironment(row)

    decision = env.execute(
        "search_documents", "public.txt"
    )

    assert not decision.allowed
    assert decision.reason == "TARGET_MISMATCH"


def test_read_before_search_denied():
    row = find_scenario("visitor", "public.txt")
    env = lab.SyntheticEnvironment(row)

    decision = env.execute("read_document", "public.txt")

    assert not decision.allowed
    assert decision.reason == "INVALID_READ_STAGE"


def test_finish_before_read_denied():
    row = find_scenario("visitor", "public.txt")
    env = lab.SyntheticEnvironment(row)

    decision = env.execute("finish", "done")

    assert not decision.allowed
    assert decision.reason == "PREMATURE_FINISH"


def test_completed_task_cannot_continue():
    row = find_scenario("visitor", "public.txt")
    env = lab.SyntheticEnvironment(row)

    env.execute("search_documents", "public.txt")
    env.execute("read_document", "public.txt")
    env.execute("finish", "done")

    extra = env.execute("read_document", "public.txt")

    assert not extra.allowed
    assert extra.reason == "ALREADY_COMPLETED"


def test_failed_action_does_not_change_stage():
    row = find_scenario("visitor", "public.txt")
    env = lab.SyntheticEnvironment(row)

    env.execute("read_document", "public.txt")

    assert env.stage == "need_search"

    success = env.execute(
        "search_documents", "public.txt"
    )
    assert success.allowed


def test_original_record_not_mutated():
    row = find_scenario("visitor", "public.txt")
    before = copy.deepcopy(row)

    env = lab.SyntheticEnvironment(row)
    env.execute("search_documents", "public.txt")

    assert row == before


def test_corrupted_ground_truth_rejected():
    row = find_scenario("visitor", "public.txt")
    row["ground_truth"][
        "original_task_authorized"
    ] = False

    with pytest.raises(ValueError):
        lab.validate_scenario(row)


def test_corrupted_permissions_rejected():
    row = find_scenario("visitor", "public.txt")
    row["ground_truth"]["allowed_resources"] = [
        "confidential.txt"
    ]

    with pytest.raises(ValueError):
        lab.validate_scenario(row)


def test_duplicate_id_rejected():
    rows = lab.generate_dataset()
    rows[1]["scenario_id"] = rows[0]["scenario_id"]

    with pytest.raises(ValueError):
        lab.validate_dataset(rows)


def test_wrong_family_split_rejected():
    rows = lab.generate_dataset()
    rows[0]["split"] = "test"

    with pytest.raises(ValueError):
        lab.validate_dataset(rows)


def test_all_clean_episodes_match_ground_truth():
    for row in lab.generate_dataset():
        episode = lab.run_clean_episode(row)

        assert episode["completed"] == (
            row["ground_truth"][
                "original_task_authorized"
            ]
        )
        assert episode["unauthorized_executions"] == 0


def test_audit_denied_action():
    row = find_scenario("visitor", "public.txt")
    env = lab.SyntheticEnvironment(row)

    env.execute("search_documents", "confidential.txt")

    assert env.summary()["blocked_actions"] == 1
    assert env.summary()["unauthorized_executions"] == 0
    assert not env.events[0]["executed"]


def test_generate_artifacts(tmp_path):
    report = lab.run_generation(tmp_path)

    assert report["scenario_count"] == 12
    assert report["split_counts"] == {
        "train": 6,
        "validation": 2,
        "test": 4,
    }
    assert report["unauthorized_executions"] == 0

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "scenarios_all.jsonl").exists()

    lines = (
        tmp_path / "scenarios_all.jsonl"
    ).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 12
    assert all(
        isinstance(json.loads(line), dict)
        for line in lines
    )


def test_dataset_hash_is_reproducible(tmp_path):
    lab.run_generation(tmp_path)

    first = (
        tmp_path / "manifest.json"
    ).read_text(encoding="utf-8")

    lab.run_generation(tmp_path)

    second = (
        tmp_path / "manifest.json"
    ).read_text(encoding="utf-8")

    assert first == second
