import json

import numpy as np
import pytest

from causalos import evidence_design as evidence


def test_contamination_models_are_observationally_equivalent():
    # Construct actual linear SCM covariances including latent U.
    rho, loading_squared = 0.65, 0.95
    for theta in (-0.65, -0.35, -0.15):
        gamma = (rho - theta) / np.sqrt(loading_squared)
        residual_variance = 1 - theta**2 - gamma**2 - 2 * theta * gamma * np.sqrt(loading_squared)
        assert residual_variance > 0
        a = np.array([np.sqrt(loading_squared), np.sqrt(1 - loading_squared), 0.0])
        y = theta * a + np.array([gamma, 0.0, np.sqrt(residual_variance)])
        np.testing.assert_allclose([[a @ a, a @ y], [y @ a, y @ y]], [[1, rho], [rho, 1]])
        # do(A) se is derived from this same SCM, not arbitrary.
        assert gamma**2 + residual_variance == pytest.approx(1 + theta**2 - 2 * theta * rho)


def test_policy_state_excludes_evaluator_truth():
    s = evidence.initial_state(42, "weak_confounding")
    assert len(s["posterior"]) == 8
    assert sum(s["posterior"]) == pytest.approx(1)
    public = json.dumps(s, allow_nan=False)
    assert "true_effect" not in public
    assert "true_model" not in public
    assert "theta" not in public
    r = evidence.recommend(s)
    evidence.Simulator(999, "unseen_negative").observe(s, r["selected"])
    assert evidence.recommend(s) == r


def test_shadow_evidence_withholds_false_zero_claim():
    state = evidence.initial_state()
    for _ in range(2):
        state = evidence.update(state, "a-high", -0.7)
    assert state["nominal_query_probability"] < 0.05
    assert state["shadow_probability"] > 0.20
    assert state["claim"] == "abstain"
    assert state["model_conflict"]


def test_shared_guard_comparator_changes_acquisition_only():
    state = evidence.initial_state(3011)
    for policy in ("nominal_query_eig_guard", "evidence"):
        ranked = evidence.recommend(state, policy)["ranked"]
        assert ranked
        assert {a["id"] for a in ranked} == {a["id"] for a in state["candidates"]}
    # The same observed evidence updates the same posterior independent of policy.
    a = evidence.update(state, "a-low", -0.2)
    b = evidence.update(state, "a-low", -0.2)
    assert a == b


def test_validity_target_ignores_nuisance_graph():
    state = evidence.initial_state()
    action = next(a for a in state["candidates"] if a["id"] == "b-probe")
    assert action["expected_information_gain"] > 0.5
    assert action["expected_query_information_gain"] < 1e-12
    assert action["expected_decision_gain"] < 1e-12


def test_steps_are_reproducible_and_budgeted():
    state = evidence.initial_state(seed=42)
    action = evidence.recommend(state)["selected"]
    a = evidence.step(state, action, seed=42)
    assert evidence.step(state, action, seed=42) == a
    assert a["spent"] > 0
    assert state["spent"] == 0
    with pytest.raises(ValueError):
        evidence.update(evidence.initial_state(budget=0.1), "a-low", 0.0)
    json.dumps(a, allow_nan=False)


def test_policy_state_is_indistinguishable_across_evaluator_families():
    nominal = evidence.initial_state(42, "nominal")
    confounded = evidence.initial_state(42, "weak_confounding")
    unseen = evidence.initial_state(42, "unseen_negative")
    assert nominal == confounded == unseen
    assert nominal["scenario"] == "validity-study"
