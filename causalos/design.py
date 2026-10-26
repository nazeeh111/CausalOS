"""Finite-SCM Bayesian experimental design with an isolated simulator.

The policy receives candidate predictive distributions and posterior weights, never
an oracle graph. Normal likelihoods and Gauss-Hermite expectations are exact for
the sample-mean summaries of the explicitly specified Gaussian SCMs, up to
quadrature error. This is a small-model research testbed, not general discovery.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

POLICIES = ("random", "uncertainty", "information_gain", "query_eig", "decision_value", "causalos", "no_cost", "no_decision")
# Optimized on training seeds and chosen on disjoint validation seeds by benchmark.py.
DEFAULT_PARAMETERS = {"decision_weight": 0.0, "query_weight": 1.0, "cost_power": 1.0, "graph_weight": 0.0}


def _entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))


def _query(p: np.ndarray) -> np.ndarray:
    return np.array([p[:2].sum(), p[2:].sum()])


def _risk(p: np.ndarray) -> float:
    """Normalized regret for binary adoption, cost = half the positive effect."""
    return float(min(_query(p)))


@lru_cache(maxsize=8)
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = hermgauss(order)
    return np.sqrt(2) * x, w / np.sqrt(np.pi)


def posterior_update(prior: list[float] | np.ndarray, means: list[float] | np.ndarray,
                     scales: list[float] | np.ndarray, observation: float) -> list[float]:
    """Bayes' rule in log space; supports tiny likelihoods without underflow."""
    p, mu, sd = np.asarray(prior, float), np.asarray(means, float), np.asarray(scales, float)
    if p.shape != mu.shape or p.shape != sd.shape or p.size != 4:
        raise ValueError("Exactly four matching hypothesis weights, means and scales are required.")
    if not np.isfinite(np.r_[p, mu, sd, observation]).all() or np.any(p < 0) or np.any(sd <= 0) or p.sum() <= 0:
        raise ValueError("Predictive parameters and observation must be finite and valid.")
    logp = np.log(np.maximum(p / p.sum(), 1e-300)) - np.log(sd) - .5 * ((observation - mu) / sd) ** 2
    return np.exp(logp - logsumexp(logp)).tolist()


def _utilities(p: np.ndarray, action: dict, order: int = 21) -> dict[str, float]:
    mu, sd = np.asarray(action["mean_predictions"]), np.asarray(action["standard_errors"])
    nodes, weights = _quadrature(order)
    observations = (mu[:, None] + sd[:, None] * nodes[None, :]).ravel()
    predictive_weights = (p[:, None] * weights[None, :]).ravel()
    logits = np.log(np.maximum(p, 1e-300))[None, :] - np.log(sd)[None, :] - .5 * ((observations[:, None] - mu[None, :]) / sd[None, :]) ** 2
    post = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    h = -np.sum(post * np.log(np.maximum(post, 1e-300)), axis=1)
    query = np.column_stack((post[:, :2].sum(1), post[:, 2:].sum(1)))
    hq = -np.sum(query * np.log(np.maximum(query, 1e-300)), axis=1)
    return {"expected_information_gain": max(0., _entropy(p) - float(predictive_weights @ h)),
            "expected_query_information_gain": max(0., _entropy(_query(p)) - float(predictive_weights @ hq)),
            "expected_decision_gain": max(0., _risk(p) - float(predictive_weights @ np.min(query, axis=1))),
            "predictive_variance": float(p @ (sd ** 2 + mu ** 2) - (p @ mu) ** 2)}


def _validate_state(state: dict) -> None:
    if state.get("schema_version") != 1 or state.get("scenario") not in ("fork", "weak", "expensive", "misspecified"):
        raise ValueError("Unrecognized experimental-design state.")
    p = np.asarray(state.get("posterior"), float)
    if p.shape != (4,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Invalid hypothesis probabilities.")
    if not 0.15 <= float(state.get("rho", 0)) <= .95:
        raise ValueError("Invalid observational correlation.")
    if not 0 <= float(state.get("budget_remaining", -1)) <= float(state.get("budget_total", -1)) <= 100:
        raise ValueError("Invalid experimental budget.")
    if len(state.get("history", [])) > 100:
        raise ValueError("Experiment history is too long.")


def _actions(rho: float, scenario: str) -> list[dict]:
    # h0/h1: A→Y; h2/h3: Y→A. h0/h2: B→Z; h1/h3: Z→B.
    residual = math.sqrt(1 - rho * rho)
    reverse_cost = 3.0 if scenario == "expensive" else 2.2
    specs = [("a-low", "Intervene on A", "A", 1., "Y", 8, 2.),
             ("a-high", "High-dose intervention on A", "A", 2., "Y", 20, 5.),
             ("y-probe", "Reverse-direction probe", "Y", 1., "A", 8, reverse_cost),
             ("b-probe", "Nuisance-graph probe", "B", 2., "Z", 16, .8),
             ("measurement", "Additional observational measurement", "observe", 0., "A", 16, .5)]
    actions = []
    for aid, label, target, value, readout, n, cost in specs:
        causal = np.array([1, 1, 0, 0]) if target == "A" else np.array([0, 0, 1, 1]) if target == "Y" else np.array([1, 0, 1, 0]) if target == "B" else np.zeros(4)
        corr = .8 if target == "B" else rho
        res = math.sqrt(1 - corr ** 2) if target == "B" else residual
        means = causal * corr * value
        scales = np.where(causal == 1, res, 1.) / math.sqrt(n)
        actions.append({"id": aid, "label": label, "target": target, "value": value, "readout": readout,
                        "n": n, "cost": cost, "mean_predictions": means.tolist(), "standard_errors": scales.tolist(),
                        "standard_error": float(scales.mean()), "kind": "measurement" if target == "observe" else "intervention"})
    return actions


def _decorate(state: dict) -> dict:
    p, rho = np.asarray(state["posterior"]), state["rho"]
    forward = float(p[:2].sum())
    state["query_probability"] = forward
    state["effect_mean"] = forward * rho
    state["effect_interval"] = [0. if 1 - forward >= .025 else rho, rho if forward >= .025 else 0.]
    state["interval_type"] = "95% finite-model posterior credible set envelope; not a frequentist confidence interval"
    state["entropy"] = _entropy(p)
    state["confidence"] = max(forward, 1 - forward)
    state["decision"] = "intervene" if forward > .5 else "do not intervene"
    state["claim"] = ("abstain" if state.get("model_conflict") or state["confidence"] < .95 else "positive effect" if forward > .5 else "zero effect")
    for h, prob in zip(state["hypotheses"], p):
        h["posterior"] = float(prob)
    state["candidates"] = [{**a, **_utilities(p, a)} for a in _actions(rho, state["scenario"])]
    return state


def initial_state(seed: int = 42, scenario: str = "fork", budget: float = 12.) -> dict[str, Any]:
    """Create observationally equivalent SCM hypotheses; seed never selects truth.

    Both A→Y and Y→A imply centered unit Gaussians with correlation rho:
    forward: A=eps_A, Y=rho*A+sqrt(1-rho²)*eps_Y;
    reverse: Y=eps_Y, A=rho*Y+sqrt(1-rho²)*eps_A.
    An independent B,Z pair has the same ambiguity and is decision irrelevant.
    """
    if scenario not in ("fork", "weak", "expensive", "misspecified"):
        raise ValueError("scenario must be fork, weak, expensive or misspecified")
    if not math.isfinite(budget) or not 0 < budget <= 100:
        raise ValueError("budget must be in (0,100]")
    rng = np.random.default_rng(int(seed) + 17011)
    rho = float(rng.uniform(.25, .45) if scenario == "weak" else rng.uniform(.45, .85))
    hypotheses = []
    for i, (q, nuisance) in enumerate(((True, True), (True, False), (False, True), (False, False))):
        edges = [["A", "Y"] if q else ["Y", "A"], ["B", "Z"] if nuisance else ["Z", "B"]]
        hypotheses.append({"id": f"h{i}", "label": " / ".join(" → ".join(e) for e in edges), "edges": edges,
                           "effect": rho if q else 0., "posterior": .25})
    return _decorate({"schema_version": 1, "scenario": scenario, "rho": rho, "nodes": ["A", "Y", "B", "Z"],
                      "hypotheses": hypotheses, "posterior": [.25] * 4, "history": [],
                      "budget_total": float(budget), "budget_remaining": float(budget), "spent": 0.,
                      "evidence_status": "simulation", "model_conflict": False,
                      "question": "Does setting A to +1 increase Y enough to justify cost rho/2?",
                      "assumptions": ["One of four specified linear Gaussian SCMs generated the data", "Perfect interventions", "Known likelihood parameters and independent observations", "No unmodeled confounding in the candidate family"],
                      "warnings": ["All intervention results are simulated, not real-world experimental verification.", "Observational likelihood is identical across hypotheses; arrow orientation is unresolved before intervention.", "Model probabilities are conditional on the candidate family, not proof of causal structure."]})


def recommend(state: dict, budget: float | None = None, policy: str = "causalos", parameters: dict | None = None,
              random_seed: int = 0) -> dict[str, Any]:
    """Rank actions using beliefs and predictive distributions only, without truth."""
    _validate_state(state)
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    params = {**DEFAULT_PARAMETERS, **(parameters or {})}
    available = min(state["budget_remaining"], budget) if budget is not None else state["budget_remaining"]
    if not math.isfinite(available) or available < 0:
        raise ValueError("Available budget must be finite and nonnegative.")
    ranked = []
    rng = np.random.default_rng(random_seed)
    for action in state["candidates"]:
        if action["cost"] > available + 1e-9:
            continue
        a = deepcopy(action)
        ig, qi, dg = (a[k] for k in ("expected_information_gain", "expected_query_information_gain", "expected_decision_gain"))
        if policy == "random":
            score = float(rng.random())
        elif policy == "uncertainty":
            score = a["predictive_variance"]
        elif policy == "information_gain":
            score = ig
        elif policy == "query_eig":
            score = qi / a["cost"]
        elif policy == "decision_value":
            score = dg / a["cost"]
        else:
            power = 0. if policy == "no_cost" else params["cost_power"]
            decision_weight = 0. if policy == "no_decision" else params["decision_weight"]
            score = (decision_weight * 2 * dg + params["query_weight"] * qi / math.log(2) + params["graph_weight"] * ig / math.log(4)) / a["cost"] ** power
        a["score"] = float(score)
        a["affordable"] = True
        ranked.append(a)
    ranked.sort(key=lambda a: (-a["score"], a["cost"], a["id"]))
    return {"selected": ranked[0]["id"] if ranked else None, "ranked": ranked, "policy": policy,
            "reason": "Maximize expected reduction in decision regret and query uncertainty per experiment cost." if policy == "causalos" else f"Baseline criterion: {policy}.",
            "parameters": params, "simulated": True}


class Simulator:
    """Evaluator-owned environment. Its true model is never passed to recommend.

    Only benchmark evaluation reads truth. A separate stream generates independent
    sample-mean measurements. Missing-family scenarios deliberately break model
    assumptions to expose calibration and claim limits.
    """
    def __init__(self, seed: int = 42, scenario: str = "fork"):
        self._seed, self.scenario = int(seed), scenario
        self.true_model = int(np.random.default_rng(self._seed + 400009).integers(4))

    def observe(self, state: dict, action_id: str) -> float:
        action = next((a for a in _actions(state["rho"], state["scenario"]) if a["id"] == action_id), None)
        if action is None:
            raise ValueError("Unknown action.")
        # Common random numbers per episode, action, and repeat avoid algorithm-specific lucky draws.
        action_index = [a["id"] for a in _actions(state["rho"], state["scenario"])].index(action_id)
        repeat = sum(h["action_id"] == action_id for h in state["history"])
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, action_index, repeat, 11939]))
        mu = action["mean_predictions"][self.true_model]
        sd = action["standard_errors"][self.true_model]
        if self.scenario == "misspecified" and action["target"] in ("A", "Y"):
            # An omitted negative direct effect is deliberately outside the four-model family.
            mu = -.8 * action["value"]
        return float(rng.normal(mu, sd))

    def true_effect(self, state: dict) -> float:
        return -.8 if self.scenario == "misspecified" else state["hypotheses"][self.true_model]["effect"]


def update(state: dict, action_id: str, observation: float) -> dict:
    """Update beliefs from an externally supplied experimental sample mean."""
    _validate_state(state)
    actions = _actions(state["rho"], state["scenario"])
    action = next((a for a in actions if a["id"] == action_id), None)
    if action is None:
        raise ValueError("Unknown action.")
    if action["cost"] > state["budget_remaining"] + 1e-9:
        raise ValueError("Experiment exceeds the remaining budget.")
    result = deepcopy(state)
    prior = np.asarray(result["posterior"])
    mu, sd = np.asarray(action["mean_predictions"]), np.asarray(action["standard_errors"])
    if not math.isfinite(observation):
        raise ValueError("Experimental observation must be finite.")
    # Conservative conflict alert: all surviving candidates place observation >3.5 SE away.
    plausible = prior > .001
    conflict = bool(np.all(np.abs((observation - mu[plausible]) / sd[plausible]) > 3.5))
    result["model_conflict"] = result["model_conflict"] or conflict
    if conflict and "Predictive conflict: observation is inconsistent with every currently plausible candidate; causal claims withheld." not in result["warnings"]:
        result["warnings"].append("Predictive conflict: observation is inconsistent with every currently plausible candidate; causal claims withheld.")
    result["posterior"] = posterior_update(prior, mu, sd, observation)
    result["spent"] += action["cost"]
    result["budget_remaining"] = max(0., result["budget_total"] - result["spent"])
    result = _decorate(result)
    result["history"].append({"round": len(result["history"]) + 1, "action_id": action_id, "label": action["label"],
                              "observation": observation, "readout": action["readout"], "posterior": result["posterior"].copy(),
                              "entropy": result["entropy"], "effect_mean": result["effect_mean"], "confidence": result["confidence"],
                              "cost": action["cost"], "cumulative_cost": result["spent"], "model_conflict": conflict})
    return result


def step(state: dict, action_id: str, seed: int | None = None) -> dict:
    """Run one explicitly simulated experiment with evaluator-owned truth."""
    env = Simulator(42 if seed is None else seed, state["scenario"])
    return update(state, action_id, env.observe(state, action_id))


def replay(seed: int = 42, policy: str = "causalos", scenario: str = "fork", budget: float = 12.,
           parameters: dict | None = None) -> dict:
    state = initial_state(seed, scenario, budget)
    env = Simulator(seed, scenario)
    while len(state["history"]) < 100:
        ranking = recommend(state, policy=policy, parameters=parameters, random_seed=seed + len(state["history"]) * 79)
        action = ranking["selected"]
        if action is None:
            break
        state = update(state, action, env.observe(state, action))
    state["recommendations"] = recommend(state, policy=policy, parameters=parameters)
    return state
