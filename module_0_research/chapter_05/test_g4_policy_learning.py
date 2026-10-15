
import pytest

import g4_policy_learning as g4


def get_case(case_id):
    return next(
        case for case in g4.TEST
        if case["id"] == case_id
    )


def test_training_uses_only_legitimate_traces():
    assert len(g4.TRAIN) == 5

    assert all(
        case["expected_allowed"]
        for case in g4.TRAIN
    )


def test_held_out_cases_are_separate():
    assert len(g4.TEST) == 14

    assert not (
        {c["id"] for c in g4.TRAIN}
        & {c["id"] for c in g4.TEST}
    )


def test_six_legitimate_eight_prohibited():
    allowed = sum(
        case["expected_allowed"]
        for case in g4.TEST
    )

    assert allowed == 6
    assert len(g4.TEST) - allowed == 8


def test_trusted_approval_not_model_supplied():
    approved = g4.get_input(
        get_case("approved_public_publish")
    )

    unapproved = g4.get_input(
        get_case("unapproved_public_publish")
    )

    assert approved["approval"]["approved"] is True
    assert unapproved["approval"]["approved"] is False


def test_coarse_features_lose_approval():
    approved = g4.get_input(
        get_case("approved_public_publish")
    )

    unapproved = g4.get_input(
        get_case("unapproved_public_publish")
    )

    assert (
        g4.coarse_features(approved)
        == g4.coarse_features(unapproved)
    )


def test_contextual_features_keep_approval():
    approved = g4.get_input(
        get_case("approved_public_publish")
    )

    unapproved = g4.get_input(
        get_case("unapproved_public_publish")
    )

    assert (
        g4.contextual_features(approved)
        != g4.contextual_features(unapproved)
    )


def test_coarse_learner_false_allows():
    learned = g4.learn_allowlist(
        g4.TRAIN,
        g4.coarse_features,
    )

    report = g4.evaluate_candidate(
        learned,
        g4.TEST,
        g4.coarse_features,
    )

    assert report["summary"]["false_allows"] == 3

    wrongly_allowed = {
        r["case"]
        for r in report["records"]
        if (
            r["predicted_allowed"]
            and not r["expected_allowed"]
        )
    }

    assert wrongly_allowed == {
        "unapproved_public_publish",
        "unread_confidential_source",
        "duplicate_confidential_summary",
    }


def test_contextual_learner_blocks_unsafe_cases():
    learned = g4.learn_allowlist(
        g4.TRAIN,
        g4.contextual_features,
    )

    report = g4.evaluate_candidate(
        learned,
        g4.TEST,
        g4.contextual_features,
    )

    assert report["summary"]["false_allows"] == 0


def test_both_learners_miss_unseen_legitimate_case():
    case = get_case("manager_public_unseen")
    data = g4.get_input(case)

    for feature_fn in g4.FEATURES.values():
        learned = g4.learn_allowlist(
            g4.TRAIN, feature_fn
        )

        assert (
            g4.predict(
                learned, data, feature_fn
            ) is False
        )


def test_contextual_candidate_not_auto_deployed():
    learned = g4.learn_allowlist(
        g4.TRAIN,
        g4.contextual_features,
    )

    report = g4.evaluate_candidate(
        learned,
        g4.TEST,
        g4.contextual_features,
    )

    gate = g4.promotion_gate(report)

    assert gate["automatically_deployed"] is False
    assert gate["human_review_required"] is True
    assert gate["passes_safety_screen"] is False


def test_training_rejects_unapproved_trace():
    bad = [
        get_case("visitor_confidential")
    ]

    with pytest.raises(ValueError):
        g4.learn_allowlist(
            bad,
            g4.coarse_features,
        )


def test_default_deny_unseen_pattern():
    learned = g4.learn_allowlist(
        g4.TRAIN,
        g4.coarse_features,
    )

    data = g4.get_input(
        get_case("visitor_restricted")
    )

    assert g4.predict(
        learned, data,
        g4.coarse_features
    ) is False


def test_complete_report_without_opa():
    report = g4.run_experiment(
        check_opa=False
    )

    assert report["training_cases"] == 5
    assert report["test_cases"] == 14
    assert "opa_reference" not in report

    coarse = report["candidates"]["coarse"]["summary"]
    contextual = report["candidates"]["contextual"]["summary"]

    assert coarse["false_allows"] == 3
    assert coarse["false_blocks"] == 1

    assert contextual["false_allows"] == 0
    assert contextual["false_blocks"] == 1


def test_no_candidate_auto_promoted():
    report = g4.run_experiment(
        check_opa=False
    )

    for candidate in report["candidates"].values():
        assert (
            candidate["promotion"][
                "automatically_deployed"
            ] is False
        )
