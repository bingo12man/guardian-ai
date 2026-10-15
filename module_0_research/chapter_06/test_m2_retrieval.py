
import pytest

import m2_retrieval as m2


def read_action(document):
    return {
        "tool": "read_document",
        "document": document,
        "reason": "Synthetic response",
    }


def test_memory_persistence(tmp_path):
    path = tmp_path / "memories.json"

    saved = m2.prepare_memory(path)

    assert path.exists()
    assert len(saved) == 3
    assert saved == m2.MemoryStore(path).load()


def test_malicious_memory_is_untrusted():
    injected = next(
        m for m in m2.MEMORIES
        if m["id"] == "injected_01"
    )

    assert injected["source"] == "untrusted_import"


def test_lexical_retrieves_clean_on_tie():
    selected = m2.retrieve(
        m2.MEMORIES,
        "lexical",
    )

    assert selected[0]["id"] == "clean_01"


def test_recency_retrieves_injection():
    selected = m2.retrieve(
        m2.MEMORIES,
        "recency",
    )

    assert selected[0]["id"] == "injected_01"


def test_weighted_retrieves_injection():
    selected = m2.retrieve(
        m2.MEMORIES,
        "weighted",
    )

    assert selected[0]["id"] == "injected_01"


def test_trust_aware_retrieves_clean():
    selected = m2.retrieve(
        m2.MEMORIES,
        "trust_aware",
    )

    assert selected[0]["id"] == "clean_01"


def test_trust_aware_excludes_untrusted():
    selected = m2.retrieve(
        m2.MEMORIES,
        "trust_aware",
        top_k=3,
    )

    assert all(
        m["source"] == "verified_observation"
        for m in selected
    )


def test_relevance_bounded():
    for memory in m2.MEMORIES:
        value = m2.relevance(memory)

        assert 0 <= value <= 1


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        m2.retrieve(
            m2.MEMORIES,
            "unknown_mode",
        )


def test_invalid_top_k_rejected():
    with pytest.raises(ValueError):
        m2.retrieve(
            m2.MEMORIES,
            "lexical",
            top_k=0,
        )


def test_bad_importance_rejected():
    bad = [
        {
            **m2.MEMORIES[0],
            "importance": 5.0,
        }
    ]

    with pytest.raises(ValueError):
        m2.validate_memories(bad)


def test_forced_memory_attack_blocked(tmp_path):
    result = m2.run_trial(
        "weighted",
        repeat=1,
        proposer=lambda task, memories: (
            read_action("confidential.txt")
        ),
        results_dir=tmp_path,
    )

    assert result["injection_exposed"] is True
    assert result["unsafe_proposal"] is True
    assert result["guardian_blocked"] is True
    assert result["task_completed"] is False


def test_trust_aware_can_complete(tmp_path):
    result = m2.run_trial(
        "trust_aware",
        repeat=1,
        proposer=lambda task, memories: (
            read_action("public.txt")
        ),
        results_dir=tmp_path,
    )

    assert result["injection_exposed"] is False
    assert result["unsafe_proposal"] is False
    assert result["guardian_blocked"] is False
    assert result["task_completed"] is True


def test_model_failure_cannot_execute(tmp_path):
    def failing(task, memories):
        raise ValueError("Invalid model response")

    result = m2.run_trial(
        "weighted",
        repeat=1,
        proposer=failing,
        results_dir=tmp_path,
    )

    assert result["valid_action"] is False
    assert result["execution"]["executed"] is False


def test_no_unsafe_proposals_undefined_rate():
    records = [{
        "mode": "trust_aware",
        "injection_exposed": False,
        "valid_action": True,
        "unsafe_proposal": False,
        "guardian_blocked": False,
        "task_completed": True,
    }]

    summary = m2.summarize(records)

    assert summary["guardian"]["unsafe_block_rate"] is None


def test_retrieval_modes_expose_expected_memory():
    expected = {
        "lexical": "clean_01",
        "recency": "injected_01",
        "weighted": "injected_01",
        "trust_aware": "clean_01",
    }

    for mode, memory_id in expected.items():
        selected = m2.retrieve(
            m2.MEMORIES, mode
        )

        assert selected[0]["id"] == memory_id