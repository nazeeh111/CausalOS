"""Validity-aware experimental design with explicit finite candidate contamination.

A separate follow-up study, prompted by the nominal policy's negative result.
Shadow models challenge the nominal family; their posterior mass is a model-risk
indicator conditional on these specified alternatives, never proof of causation.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from . import design

SHADOW_THETA = (-0.15, -0.35)
DEFAULTS = {"contamination_prior": 0.05, "decision_weight": 0.5}
POLICIES = (
    "nominal_query_eig",
    "nominal_query_eig_guard",
    "augmented_model_eig",
    "validity_query_eig",
    "validity_decision_value",
    "evidence_no_guard",
    "evidence",
    "random_guard",
)
GUARD_THRESHOLD = 0.20
ABSTENTION_LOSS = 0.10


def _classes(p: np.ndarray) -> np.ndarray:
    return np.array([p[:2].sum(), p[2:4].sum(), p[4:].sum()])


def _loss(p: np.ndarray) -> float:
    # SURROGATE: three-class classification with reject loss, not the exact
    # operational claim loss (which uses a separate 95% nominal threshold and
    # shadow guard). The third class flags nominal-family inadequacy.
    return min(ABSTENTION_LOSS, float(1 - _classes(p).max()))


def _posterior(p: np.ndarray, means: np.ndarray, scales: np.ndarray, observation: float) -> np.ndarray:
    logits = np.log(np.maximum(p, 1e-300)) - np.log(scales) - 0.5 * ((observation - means) / scales) ** 2
    return np.exp(logits - logsumexp(logits))


def _actions(rho: float) -> list[dict]:
    result = []
    for base in design._actions(rho, "fork"):
        action = deepcopy(base)
        mus, sds = action["mean_predictions"], action["standard_errors"]
        for theta in SHADOW_THETA:
            for nuisance_forward in (True, False):
                if action["target"] == "A":
                    mu = theta * action["value"]
                    sd = math.sqrt(1 + theta * theta - 2 * theta * rho) / math.sqrt(action["n"])
                elif action["target"] == "Y":
                    mu, sd = 0.0, 1 / math.sqrt(action["n"])
                elif action["target"] == "B":
                    mu = 0.8 * action["value"] if nuisance_forward else 0.0
                    sd = (0.6 if nuisance_forward else 1.0) / math.sqrt(action["n"])
                else:
                    mu, sd = 0.0, 1 / math.sqrt(action["n"])
                mus.append(mu)
                sds.append(sd)
        action["mean_predictions"], action["standard_errors"] = mus, sds
        action["standard_error"] = float(np.mean(sds))
        result.append(action)
    return result


def _utilities(p: np.ndarray, action: dict) -> dict:
    mu, sd = np.array(action["mean_predictions"]), np.array(action["standard_errors"])
    nodes, weights = design._quadrature(21)
    obs = (mu[:, None] + sd[:, None] * nodes[None, :]).ravel()
    w = (p[:, None] * weights[None, :]).ravel()
    logits = (
        np.log(np.maximum(p, 1e-300))[None, :]
        - np.log(sd)[None, :]
        - 0.5 * ((obs[:, None] - mu[None, :]) / sd[None, :]) ** 2
    )
    post = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    classes = np.column_stack((post[:, :2].sum(1), post[:, 2:4].sum(1), post[:, 4:].sum(1)))
    h = -np.sum(post * np.log(np.maximum(post, 1e-300)), 1)
    hc = -np.sum(classes * np.log(np.maximum(classes, 1e-300)), 1)
    losses = np.minimum(ABSTENTION_LOSS, 1 - classes.max(1))
    return {
        "expected_information_gain": max(0.0, design._entropy(p) - float(w @ h)),
        "expected_query_information_gain": max(0.0, design._entropy(_classes(p)) - float(w @ hc)),
        "expected_decision_gain": max(0.0, _loss(p) - float(w @ losses)),
        "predictive_variance": float(p @ (sd * sd + mu * mu) - (p @ mu) ** 2),
    }


def _validate(state: dict) -> None:
    p = np.asarray(state.get("posterior"), float)
    if (
        state.get("evidence_schema_version") != 1
        or p.shape != (8,)
        or np.any(p < 0)
        or not np.isfinite(p).all()
        or not np.isclose(p.sum(), 1)
    ):
        raise ValueError("Invalid validity-aware design state.")
    if (
        not 0.15 <= state.get("rho", 0) <= 0.95
        or not 0 <= state.get("budget_remaining", -1) <= state.get("budget_total", -1) <= 100
    ):
        raise ValueError("Invalid model parameter or experimental budget.")


def _decorate(state: dict, guard: bool = True) -> dict:
    p = np.asarray(state["posterior"])
    nominal = p[:4] / max(p[:4].sum(), 1e-300)
    nominal_query = float(nominal[:2].sum())
    state["nominal_posterior"] = nominal.tolist()
    state["query_probability"] = float(p[:2].sum())
    state["nominal_query_probability"] = nominal_query
    state["shadow_probability"] = float(p[4:].sum())
    effects = np.array([h["effect"] for h in state["hypotheses"]])
    state["effect_mean"] = float(p @ effects)
    sorted_idx = np.argsort(effects)
    cdf = np.cumsum(p[sorted_idx])
    state["effect_interval"] = [
        float(effects[sorted_idx[np.searchsorted(cdf, q, side="left")]]) for q in (0.025, 0.975)
    ]
    state["interval_type"] = (
        "Finite contamination-family 95% posterior credible envelope; not a general confidence interval"
    )
    state["entropy"] = design._entropy(p)
    state["confidence"] = float(max(_classes(p)))
    state["decision"] = "intervene" if state["effect_mean"] > state["rho"] / 2 else "do not intervene"
    state["model_conflict"] = bool(state["shadow_probability"] > GUARD_THRESHOLD)
    nominal_confident = max(nominal_query, 1 - nominal_query) >= 0.95
    state["claim"] = (
        "abstain"
        if not nominal_confident or (guard and state["model_conflict"])
        else "positive effect"
        if nominal_query > 0.5
        else "zero effect"
    )
    state["evidence_status"] = "simulation"
    state["risk_is_surrogate"] = True
    state["risk_definition"] = (
        "Three-class classification with reject loss 0.10; operational claim guard is separate"
    )
    for h, prob in zip(state["hypotheses"], p):
        h["posterior"] = float(prob)
    state["candidates"] = [{**a, **_utilities(p, a)} for a in _actions(state["rho"])]
    return state


def initial_state(
    seed: int = 42, scenario: str = "nominal", budget: float = 12.0, contamination_prior: float = 0.05
) -> dict[str, Any]:
    if scenario not in ("nominal", "weak_confounding", "unseen_negative"):
        raise ValueError("Unknown evidence-study scenario.")
    if not 0 < contamination_prior < 0.5:
        raise ValueError("Contamination prior must be between zero and one half.")
    state = design.initial_state(seed=seed, budget=budget)
    state["scenario"] = "validity-study"
    state["evidence_schema_version"] = 1
    state["nodes"] = ["A", "Y", "B", "Z", "U"]
    state["posterior"] = [(1 - contamination_prior) / 4] * 4 + [contamination_prior / 4] * 4
    for theta in SHADOW_THETA:
        for nuisance in (True, False):
            state["hypotheses"].append(
                {
                    "id": f"shadow-{len(state['hypotheses'])}",
                    "label": f"Latent confounding / effect {theta:+.2f}",
                    "edges": [["U", "A"], ["U", "Y"], ["A", "Y"], ["B", "Z"] if nuisance else ["Z", "B"]],
                    "effect": theta,
                    "shadow": True,
                    "posterior": contamination_prior / 4,
                }
            )
    for h in state["hypotheses"][:4]:
        h["shadow"] = False
    state["warnings"] += [
        "Shadow models challenge the nominal causal family. Their posterior mass measures risk relative to the specified alternatives.",
        "A model-risk alert is not proof of a real-world negative treatment effect.",
        "All mechanism coefficients and the finite shadow set are fixed, not estimated from these experiments.",
    ]
    state["assumptions"] = [
        "Specified finite nominal and contamination SCM families",
        "Known mechanism coefficients and Gaussian noise",
        "Perfect interventions",
        "Independent post-intervention samples",
    ]
    return _decorate(state)


def recommend(
    state: dict, policy: str = "evidence", decision_weight: float = 0.5, random_seed: int = 0
) -> dict:
    _validate(state)
    if policy not in POLICIES:
        raise ValueError("Unknown validity-aware policy.")
    if not 0 <= decision_weight <= 1:
        raise ValueError("Decision weight must be in [0,1].")
    rng = np.random.default_rng(random_seed)
    nominal = design.initial_state(0, budget=max(0.01, state["budget_total"]))
    nominal["rho"] = state["rho"]
    nominal["posterior"] = state["nominal_posterior"]
    nominal["budget_remaining"] = state["budget_remaining"]
    nominal = design._decorate(nominal)
    nominal_scores = {a["id"]: a for a in nominal["candidates"]}
    ranked = []
    for action in state["candidates"]:
        if action["cost"] > state["budget_remaining"] + 1e-9:
            continue
        a = deepcopy(action)
        if policy in ("nominal_query_eig", "nominal_query_eig_guard"):
            score = nominal_scores[a["id"]]["expected_query_information_gain"] / a["cost"]
        elif policy == "augmented_model_eig":
            score = a["expected_information_gain"] / a["cost"]
        elif policy == "validity_query_eig":
            score = a["expected_query_information_gain"] / a["cost"]
        elif policy == "validity_decision_value":
            score = a["expected_decision_gain"] / a["cost"]
        elif policy == "random_guard":
            score = float(rng.random())
        else:
            score = (
                decision_weight * a["expected_decision_gain"] / ABSTENTION_LOSS
                + (1 - decision_weight) * a["expected_query_information_gain"] / math.log(3)
            ) / a["cost"]
        a["score"] = float(score)
        ranked.append(a)
    ranked.sort(key=lambda a: (-a["score"], a["cost"], a["id"]))
    return {
        "selected": ranked[0]["id"] if ranked else None,
        "ranked": ranked,
        "policy": policy,
        "reason": "Select evidence that distinguishes positive effect, zero effect and failure of the nominal model family.",
        "simulated": True,
    }


def update(state: dict, action_id: str, observation: float, guard: bool = True) -> dict:
    _validate(state)
    action = next((a for a in _actions(state["rho"]) if a["id"] == action_id), None)
    if action is None or action["cost"] > state["budget_remaining"] + 1e-9:
        raise ValueError("Unknown or unaffordable experiment.")
    if not math.isfinite(observation):
        raise ValueError("Experimental observation must be finite.")
    result = deepcopy(state)
    p = _posterior(
        np.asarray(state["posterior"]),
        np.asarray(action["mean_predictions"]),
        np.asarray(action["standard_errors"]),
        observation,
    )
    result["posterior"] = p.tolist()
    result["spent"] += action["cost"]
    result["budget_remaining"] = max(0.0, state["budget_total"] - result["spent"])
    result = _decorate(result, guard)
    result["history"].append(
        {
            "round": len(result["history"]) + 1,
            "action_id": action_id,
            "label": action["label"],
            "observation": float(observation),
            "readout": action["readout"],
            "posterior": p.tolist(),
            "entropy": result["entropy"],
            "effect_mean": result["effect_mean"],
            "confidence": result["confidence"],
            "cost": action["cost"],
            "cumulative_cost": result["spent"],
            "shadow_probability": result["shadow_probability"],
        }
    )
    return result


class Simulator:
    """Evaluator-only SCM environment; continuous omitted effects never enter policy state."""

    def __init__(self, seed: int, scenario: str):
        if scenario not in ("nominal", "weak_confounding", "unseen_negative"):
            raise ValueError("Unknown evaluator scenario.")
        self.seed, self.scenario = int(seed), scenario
        rng = np.random.default_rng(seed + 718319)
        self.true_model = int(rng.integers(4))
        self.theta = (
            float(rng.uniform(-0.32, -0.08))
            if scenario == "weak_confounding"
            else float(rng.uniform(-0.65, -0.40))
            if scenario == "unseen_negative"
            else None
        )

    def true_effect(self, state: dict) -> float:
        return state["hypotheses"][self.true_model]["effect"] if self.theta is None else self.theta

    def observe(self, state: dict, action_id: str) -> float:
        action = next((a for a in _actions(state["rho"]) if a["id"] == action_id), None)
        if action is None:
            raise ValueError("Unknown experiment.")
        mu, sd = action["mean_predictions"][self.true_model], action["standard_errors"][self.true_model]
        if self.theta is not None:
            if action["target"] == "A":
                mu = self.theta * action["value"]
                sd = math.sqrt(1 + self.theta**2 - 2 * self.theta * state["rho"]) / math.sqrt(action["n"])
            elif action["target"] == "Y":
                mu, sd = 0.0, 1 / math.sqrt(action["n"])
        action_index = [a["id"] for a in _actions(state["rho"])].index(action_id)
        repeat = sum(h["action_id"] == action_id for h in state["history"])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, action_index, repeat, 86431]))
        return float(rng.normal(mu, sd))


def step(state: dict, action_id: str, seed: int | None = None, scenario: str = "nominal") -> dict:
    env = Simulator(42 if seed is None else seed, scenario)
    return update(state, action_id, env.observe(state, action_id))


def episode(
    seed: int, scenario: str, policy: str, contamination_prior: float = 0.05, decision_weight: float = 0.5
) -> tuple[dict, list[dict]]:
    state = initial_state(seed, scenario, contamination_prior=contamination_prior)
    env = Simulator(seed, scenario)
    truth = env.true_effect(state)
    guard = policy not in ("nominal_query_eig", "evidence_no_guard")
    trace, cost_to_valid = [], None
    for round_number in range(100):
        recommendation = recommend(state, policy, decision_weight, seed + round_number * 79)
        if recommendation["selected"] is None:
            break
        state = update(
            state, recommendation["selected"], env.observe(state, recommendation["selected"]), guard
        )
        true_claim = (
            state["claim"] == "positive effect"
            and truth > 0
            or state["claim"] == "zero effect"
            and abs(truth) < 1e-12
        )
        if true_claim and cost_to_valid is None:
            cost_to_valid = state["spent"]
        trace.append(
            {
                "seed": seed,
                "scenario": scenario,
                "policy": policy,
                "round": round_number + 1,
                "cost": state["spent"],
                "claim": state["claim"],
                "shadow_probability": state["shadow_probability"],
                "effect_error": abs(state["effect_mean"] - truth),
            }
        )
    claim = state["claim"] != "abstain"
    true_claim = (
        state["claim"] == "positive effect"
        and truth > 0
        or state["claim"] == "zero effect"
        and abs(truth) < 1e-12
    )
    return {
        "seed": seed,
        "scenario": scenario,
        "policy": policy,
        "true_effect": truth,
        "effect_estimate": state["effect_mean"],
        "effect_squared_error": (state["effect_mean"] - truth) ** 2,
        "false_claim": float(claim and not true_claim),
        "correct_claim": float(true_claim),
        "abstention": float(not claim),
        "cost_to_valid_claim": cost_to_valid if cost_to_valid is not None else 12.5,
        "spent": state["spent"],
        "experiments": len(state["history"]),
        "shadow_probability": state["shadow_probability"],
        "family_alert": float(state["model_conflict"]),
        "alert_truth": float(scenario != "nominal"),
    }, trace


def _objective(rows: list[dict]) -> float:
    false = np.mean([r["false_claim"] for r in rows])
    nominal_abstention = np.mean([r["abstention"] for r in rows if r["scenario"] == "nominal"])
    cost = np.mean([r["cost_to_valid_claim"] for r in rows])
    return float(false + 0.25 * nominal_abstention + 0.02 * cost / 12)


def run_benchmark(
    output: str | Path = "benchmarks/evidence_results", test_count: int = 96, development_count: int = 24
) -> dict:
    import hashlib
    import platform
    import time
    from .benchmark import _save_csv

    if not 2 <= test_count <= 1000 or not 2 <= development_count <= 1000:
        raise ValueError("Seed counts must lie between 2 and 1000.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = ("nominal", "weak_confounding", "unseen_negative")
    candidates = [
        {"contamination_prior": cp, "decision_weight": dw}
        for cp in (0.05, 0.15, 0.25)
        for dw in (0.0, 0.5, 1.0)
    ]
    start = time.time()
    training = []
    for params in candidates:
        rows = [
            episode(seed, scenario, "evidence", **params)[0]
            for seed in range(3000, 3000 + development_count)
            for scenario in scenarios
        ]
        training.append({"parameters": params, "objective": _objective(rows)})
        print("Training", params, training[-1]["objective"], flush=True)
    training.sort(key=lambda x: (x["objective"], json.dumps(x["parameters"], sort_keys=True)))
    validation = []
    for candidate in training[:3]:
        params = candidate["parameters"]
        rows = [
            episode(seed, scenario, "evidence", **params)[0]
            for seed in range(4000, 4000 + development_count)
            for scenario in scenarios
        ]
        validation.append({"parameters": params, "objective": _objective(rows)})
    validation.sort(key=lambda x: (x["objective"], json.dumps(x["parameters"], sort_keys=True)))
    selected = validation[0]["parameters"]
    print("Validation locked selection", selected, flush=True)
    (output / "development.json").write_text(
        json.dumps({"training": training, "validation": validation, "selected": selected}, indent=2) + "\n"
    )
    rows, traces = [], []
    for scenario in scenarios:
        for seed in range(5000, 5000 + test_count):
            for policy in POLICIES:
                row, trace = episode(seed, scenario, policy, **selected)
                rows.append(row)
                traces.extend(trace)
        print("Finished held-out", scenario, flush=True)
    summary = []
    metrics = (
        "false_claim",
        "correct_claim",
        "abstention",
        "cost_to_valid_claim",
        "spent",
        "experiments",
        "family_alert",
        "shadow_probability",
    )
    for scenario in scenarios:
        for policy in POLICIES:
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            item = {"scenario": scenario, "policy": policy, "episodes": len(group)}
            for metric in metrics:
                values = np.array([r[metric] for r in group])
                item[metric] = float(values.mean())
                item[metric + "_se"] = float(values.std(ddof=1) / np.sqrt(len(values)))
            item["effect_rmse"] = float(np.sqrt(np.mean([r["effect_squared_error"] for r in group])))
            summary.append(item)
    # Paired evaluation of the primary acquisition comparison, not a post-hoc
    # independent-sample significance claim. Positive reduction favors evidence.
    paired = []
    for scenario in scenarios:
        indexed = {(r["seed"], r["policy"]): r for r in rows if r["scenario"] == scenario}
        for metric in ("false_claim", "abstention", "cost_to_valid_claim"):
            differences = np.array(
                [
                    indexed[s, "nominal_query_eig_guard"][metric] - indexed[s, "evidence"][metric]
                    for s in range(5000, 5000 + test_count)
                ]
            )
            paired.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "mean_reduction": float(differences.mean()),
                    "paired_se": float(differences.std(ddof=1) / np.sqrt(test_count)),
                }
            )
    config = {
        "study": 2,
        "train_seeds": [3000, 3000 + development_count - 1],
        "validation_seeds": [4000, 4000 + development_count - 1],
        "test_seeds": [5000, 5000 + test_count - 1],
        "parameters": selected,
        "shadow_theta_grid": list(SHADOW_THETA),
        "budget": 12.0,
        "guard_threshold": GUARD_THRESHOLD,
        "abstention_loss": ABSTENTION_LOSS,
        "test_episodes": len(rows),
        "elapsed_seconds": round(time.time() - start, 3),
        "python": platform.python_version(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_comparison": "evidence versus nominal_query_eig_guard, sharing the augmented posterior and guard",
        "limits": [
            "Fixed known mechanism coefficients",
            "Finite two-point shadow set",
            "Synthetic Gaussian SCM evidence only",
            "Mechanism developed after a negative nominal study with a new held-out test split",
            "Alerts are evidence relative to the specified alternative family, not proof of real-world causal direction",
        ],
    }
    _save_csv(output / "episodes.csv", rows)
    _save_csv(output / "summary.csv", summary)
    _save_csv(output / "traces.csv", traces)
    _save_csv(output / "paired_comparison.csv", paired)
    payload = {"config": config, "summary": summary, "paired_comparison": paired}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    _plot(output, summary)
    secondary_alert_analysis(output)
    return payload


def secondary_alert_analysis(output: str | Path) -> dict:
    """Secondary trace-only endpoint added after independent method review.

    This does not alter any episode, selected configuration, or locked objective.
    A cost of 12.5 denotes no alert by budget12. The nominal scenario reports
    false alerts; in negative scenarios an alert is a correct family challenge.
    """
    import csv
    from collections import defaultdict
    from .benchmark import _save_csv

    output = Path(output)
    episodes = list(csv.DictReader((output / "episodes.csv").open()))
    traces = list(csv.DictReader((output / "traces.csv").open()))
    histories = defaultdict(list)
    for trace in traces:
        histories[trace["seed"], trace["scenario"], trace["policy"]].append(trace)
    rows = []
    for episode_row in episodes:
        key = (episode_row["seed"], episode_row["scenario"], episode_row["policy"])
        alerts = [
            float(t["cost"]) for t in histories[key] if float(t["shadow_probability"]) > GUARD_THRESHOLD
        ]
        rows.append(
            {
                "seed": int(key[0]),
                "scenario": key[1],
                "policy": key[2],
                "cost_to_family_alert": min(alerts) if alerts else 12.5,
                "alert_attained": float(bool(alerts)),
                "alert_is_correct": float(key[1] != "nominal"),
            }
        )
    summary = []
    paired = []
    for scenario in ("nominal", "weak_confounding", "unseen_negative"):
        for policy in POLICIES:
            group = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            values = np.array([r["cost_to_family_alert"] for r in group])
            summary.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "episodes": len(group),
                    "cost_to_family_alert": float(values.mean()),
                    "cost_to_family_alert_se": float(values.std(ddof=1) / np.sqrt(len(values))),
                    "alert_attainment": float(np.mean([r["alert_attained"] for r in group])),
                }
            )
        indexed = {(r["seed"], r["policy"]): r for r in rows if r["scenario"] == scenario}
        seeds = sorted({r["seed"] for r in rows if r["scenario"] == scenario})
        for metric in ("cost_to_family_alert", "alert_attained"):
            delta = np.array(
                [
                    indexed[s, "nominal_query_eig_guard"][metric] - indexed[s, "evidence"][metric]
                    for s in seeds
                ]
            )
            rng = np.random.default_rng(83193)
            draws = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
            paired.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "nominal_guard_minus_evidence": float(delta.mean()),
                    "bootstrap_95_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_95_high": float(np.quantile(draws, 0.975)),
                }
            )
    _save_csv(output / "secondary_alert_episodes.csv", rows)
    _save_csv(output / "secondary_alert_summary.csv", summary)
    _save_csv(output / "secondary_alert_paired.csv", paired)
    payload = {
        "endpoint_status": "secondary, added after independent method review; does not change locked optimization objective",
        "summary": summary,
        "paired_comparisons": paired,
    }
    (output / "secondary_alerts.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _plot(output: Path, summary: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = (
        "nominal_query_eig",
        "nominal_query_eig_guard",
        "validity_query_eig",
        "validity_decision_value",
        "evidence",
    )
    colors = ("#9a9a91", "#c39446", "#2b8c86", "#759447", "#dc5d42")
    labels = (
        "nominal query-EIG",
        "nominal + guard",
        "validity query-EIG",
        "validity decision",
        "CAUSALOS evidence",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    fig.set_facecolor("#faf9f5")
    for ax, scenario in zip(axes, ("nominal", "weak_confounding", "unseen_negative")):
        group = {r["policy"]: r for r in summary if r["scenario"] == scenario}
        values = [group[p]["false_claim"] for p in policies]
        errors = [group[p]["false_claim_se"] for p in policies]
        ax.bar(range(len(policies)), values, yerr=errors, capsize=3, color=colors)
        ax.set_xticks(range(len(policies)), labels, rotation=42, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(scenario.replace("_", " "), loc="left", fontsize=11)
        ax.set_ylabel("False causal-claim fraction ↓")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("#faf9f5")
    fig.suptitle(
        "CAUSALOS · candidate-family challenge on new held-out SCMs", x=0.01, ha="left", fontweight="bold"
    )
    fig.savefig(output / "evidence_comparison.svg")
    fig.savefig(output / "evidence_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/evidence_results")
    parser.add_argument("--seeds", type=int, default=96)
    args = parser.parse_args()
    run_benchmark(args.output, test_count=args.seeds)
