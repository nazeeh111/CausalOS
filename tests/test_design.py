import json

import numpy as np
import pytest

from causalos import design


def test_markov_equivalent_observational_distribution():
    # Both directions of a standardized Gaussian pair imply this covariance.
    rho = 0.7
    forward_transform = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
    reverse_transform = np.array([[np.sqrt(1 - rho**2), rho], [0.0, 1.0]])
    np.testing.assert_allclose(
        forward_transform @ forward_transform.T, reverse_transform @ reverse_transform.T
    )
    s = design.initial_state()
    assert s["posterior"] == [0.25] * 4
    assert s["claim"] == "abstain"


def test_bayes_rule_and_stability():
    p = design.posterior_update([0.25] * 4, [0, 0, 1, 1], [1] * 4, 1)
    assert sum(p) == pytest.approx(1)
    assert p[2] / p[0] == pytest.approx(np.exp(0.5))
    assert np.isfinite(design.posterior_update([0.25] * 4, [0, 0, 1, 1], [1] * 4, 1e4)).all()


def test_no_query_information_from_nuisance_or_passive_measurement():
    s = design.initial_state()
    nuisance = next(a for a in s["candidates"] if a["id"] == "b-probe")
    passive = next(a for a in s["candidates"] if a["id"] == "measurement")
    assert nuisance["expected_information_gain"] > 0.5
    assert nuisance["expected_query_information_gain"] < 1e-12
    assert nuisance["expected_decision_gain"] < 1e-12
    assert passive["expected_information_gain"] < 1e-12
    assert design.recommend(s)["selected"] != "b-probe"


def test_intervention_changes_beliefs_and_accounts_for_budget():
    s = design.initial_state(seed=42)
    recommendation = design.recommend(s)
    a = next(x for x in s["candidates"] if x["id"] == recommendation["selected"])
    result = design.step(s, a["id"], seed=42)
    assert result["posterior"] != s["posterior"]
    assert result["spent"] == a["cost"]
    assert result["budget_remaining"] == s["budget_total"] - a["cost"]
    assert s["history"] == []  # immutable input
    assert result["evidence_status"] == "simulation"
    json.dumps(result, allow_nan=False)


def test_policy_has_no_truth_access_and_is_deterministic():
    s = design.initial_state(seed=123)
    public_json = json.dumps(s)
    assert "true_model" not in public_json
    assert "true_effect" not in public_json
    assert "seed" not in public_json
    before = design.recommend(s)
    design.Simulator(99999).observe(s, "a-low")
    assert design.recommend(s) == before
    assert design.replay(42) == design.replay(42)


def test_infeasible_and_invalid_requests():
    s = design.initial_state(budget=0.25)
    assert design.recommend(s)["selected"] is None
    with pytest.raises(ValueError, match="budget"):
        design.step(s, "a-high")
    with pytest.raises(ValueError):
        design.initial_state(scenario="invented")
    with pytest.raises(ValueError):
        design.update(design.initial_state(), "a-low", float("nan"))


def test_out_of_family_prediction_triggers_abstention():
    s = design.initial_state(scenario="misspecified")
    s = design.update(s, "a-high", -1.6)
    assert s["model_conflict"]
    assert s["claim"] == "abstain"


def test_quadrature_matches_higher_order_reference():
    s = design.initial_state()
    action = next(a for a in s["candidates"] if a["id"] == "a-low")
    low = design._utilities(np.array(s["posterior"]), action, order=21)
    reference = design._utilities(np.array(s["posterior"]), action, order=81)
    assert low["expected_information_gain"] == pytest.approx(
        reference["expected_information_gain"], abs=0.003
    )
    assert low["expected_decision_gain"] == pytest.approx(reference["expected_decision_gain"], abs=0.012)
