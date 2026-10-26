"""Reproducible train/validation/test evaluation for finite-SCM design policies."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from . import design

SCENARIOS = ("fork", "weak", "expensive")
BASELINES = (
    "random",
    "uncertainty",
    "information_gain",
    "query_eig",
    "decision_value",
    "causalos",
    "no_cost",
    "no_decision",
)


def episode(
    seed: int, scenario: str, policy: str, parameters: dict | None = None, budget: float = 12.0
) -> tuple[dict, list[dict]]:
    state = design.initial_state(seed, scenario, budget)
    env = design.Simulator(seed, scenario)
    true_effect = env.true_effect(state)
    cost_to_confidence = None
    trace = []
    for round_number in range(100):
        r = design.recommend(
            state, policy=policy, parameters=parameters, random_seed=seed + round_number * 79
        )
        if r["selected"] is None:
            break
        state = design.update(state, r["selected"], env.observe(state, r["selected"]))
        if state["confidence"] >= 0.95 and not state["model_conflict"] and cost_to_confidence is None:
            cost_to_confidence = state["spent"]
        trace.append(
            {
                "seed": seed,
                "scenario": scenario,
                "policy": policy,
                "round": round_number + 1,
                "cost": state["spent"],
                "effect_error": abs(state["effect_mean"] - true_effect),
                "query_entropy": design._entropy(design._query(np.array(state["posterior"]))),
                "regret": _regret(state, true_effect),
                "confidence": state["confidence"],
            }
        )
        if state["model_conflict"]:
            break
    forward_truth = float(true_effect > state["rho"] / 2)
    prediction = float(state["query_probability"])
    shd = _graph_shd(int(np.argmax(state["posterior"])), env.true_model)
    claim_made = state["claim"] != "abstain"
    claim_correct = (state["claim"] == "positive effect" and true_effect > 0) or (
        state["claim"] == "zero effect" and abs(true_effect) < 1e-12
    )
    row = {
        "seed": seed,
        "scenario": scenario,
        "policy": policy,
        "true_model": env.true_model,
        "true_effect": true_effect,
        "effect_estimate": state["effect_mean"],
        "effect_squared_error": (state["effect_mean"] - true_effect) ** 2,
        "decision_regret": _regret(state, true_effect),
        "query_brier": (prediction - forward_truth) ** 2,
        "credible_coverage": float(
            state["effect_interval"][0] - 1e-9 <= true_effect <= state["effect_interval"][1] + 1e-9
        ),
        "graph_shd": shd if scenario != "misspecified" else None,
        "graph_exact": float(shd == 0) if scenario != "misspecified" else None,
        "posterior_truth": state["posterior"][env.true_model] if scenario != "misspecified" else None,
        "spent": state["spent"],
        "experiments": len(state["history"]),
        "cost_to_95": cost_to_confidence if cost_to_confidence is not None else budget + 0.5,
        "reached_95": float(cost_to_confidence is not None),
        "claim_made": float(claim_made),
        "false_causal_claim": float(claim_made and not claim_correct),
        "abstained": float(not claim_made),
        "model_conflict": float(state["model_conflict"]),
        "query_probability": prediction,
        "query_truth": forward_truth,
    }
    return row, trace


def _regret(state: dict, truth: float) -> float:
    cost = state["rho"] / 2
    chosen_utility = truth - cost if state["decision"] == "intervene" else 0.0
    return max(0.0, truth - cost) - chosen_utility


def _graph_shd(predicted: int, truth: int) -> int:
    # A reversal counts as one edge error, rather than two adjacency entries.
    return int(predicted // 2 != truth // 2) + int(predicted % 2 != truth % 2)


def _objective(rows: list[dict]) -> float:
    # Prespecified dimensionless development objective. Ground truth is used by
    # the offline optimizer only, never the action-selection policy at test time.
    return float(
        np.mean(
            [
                r["decision_regret"] / max(abs(r["true_effect"]), 0.25)
                + 0.20 * r["query_brier"]
                + 0.04 * r["cost_to_95"] / 12
                for r in rows
            ]
        )
    )


def optimize(train_count: int = 24, validation_count: int = 24) -> tuple[dict, dict]:
    candidates = [
        {"decision_weight": dw, "query_weight": 1 - dw, "cost_power": cp, "graph_weight": 0.0}
        for dw in (0.0, 0.5, 1.0)
        for cp in (0.0, 1.0, 1.5)
    ]
    training = []
    for p in candidates:
        rows = [episode(s, scene, "causalos", p)[0] for s in range(train_count) for scene in SCENARIOS]
        training.append({"parameters": p, "objective": _objective(rows)})
    training.sort(key=lambda x: (x["objective"], json.dumps(x["parameters"], sort_keys=True)))
    validation = []
    for candidate in training[:3]:
        p = candidate["parameters"]
        rows = [
            episode(s, scene, "causalos", p)[0]
            for s in range(1000, 1000 + validation_count)
            for scene in SCENARIOS
        ]
        validation.append({"parameters": p, "objective": _objective(rows)})
    validation.sort(key=lambda x: (x["objective"], json.dumps(x["parameters"], sort_keys=True)))
    return validation[0]["parameters"], {
        "training": training,
        "validation": validation,
        "objective": "mean(normalized regret + 0.20 query Brier + 0.04 capped cost-to-95 / budget)",
        "selection": "Train nine parameter settings; choose among training top three using validation only.",
    }


def _summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario"], row["policy"]].append(row)
    results = []
    metrics = (
        "decision_regret",
        "query_brier",
        "credible_coverage",
        "graph_shd",
        "graph_exact",
        "cost_to_95",
        "reached_95",
        "spent",
        "experiments",
        "false_causal_claim",
        "abstained",
        "model_conflict",
    )
    for (scenario, policy), group in sorted(groups.items()):
        item = {"scenario": scenario, "policy": policy, "episodes": len(group)}
        for metric in metrics:
            a = np.array([r[metric] for r in group if r[metric] is not None])
            item[metric] = float(a.mean()) if len(a) else None
            item[metric + "_se"] = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        item["effect_rmse"] = float(np.sqrt(np.mean([r["effect_squared_error"] for r in group])))
        results.append(item)
    return results


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _plots(output: Path, summary: list[dict], traces: list[dict], rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#faf9f5",
            "axes.facecolor": "#faf9f5",
            "text.color": "#1b3434",
            "axes.labelcolor": "#1b3434",
        }
    )
    colors = {
        "random": "#9a9a91",
        "uncertainty": "#8c6c9c",
        "information_gain": "#d5964a",
        "query_eig": "#2b8c86",
        "decision_value": "#759447",
        "causalos": "#dc5d42",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("cost_to_95", "decision_regret", "query_brier"),
        ("Cost to 95% query confidence ↓", "Decision regret at budget ↓", "Query Brier score ↓"),
    ):
        policies = list(colors)
        values, errors = [], []
        for policy in policies:
            by_seed = defaultdict(list)
            for row in rows:
                if row["policy"] == policy and row["scenario"] in SCENARIOS:
                    by_seed[row["seed"]].append(float(row[metric]))
            a = np.array([np.mean(values) for values in by_seed.values()])
            values.append(a.mean())
            errors.append(a.std(ddof=1) / math_sqrt(len(a)))
        ax.bar(
            range(len(policies)),
            values,
            yerr=errors,
            color=[colors[p] for p in policies],
            width=0.7,
            capsize=3,
        )
        ax.set_xticks(
            range(len(policies)), [p.replace("_", " ") for p in policies], rotation=38, ha="right", fontsize=8
        )
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("CAUSALOS · held-out finite-SCM design benchmark", x=0.01, ha="left", fontweight="bold")
    fig.savefig(output / "policy_comparison.svg")
    fig.savefig(output / "policy_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for policy in colors:
        group = [r for r in rows if r["policy"] == policy and r["scenario"] in SCENARIOS]
        # Query probability calibration bins, no truth used in policies.
        xs, ys = [], []
        for lo in np.linspace(0, 0.8, 5):
            b = [r for r in group if lo <= r["query_probability"] <= lo + 0.2]
            if b:
                xs.append(np.mean([r["query_probability"] for r in b]))
                ys.append(np.mean([r["query_truth"] for r in b]))
        axes[0].plot(xs, ys, "o-", color=colors[policy], label=policy.replace("_", " "), alpha=0.8)
        costs = np.linspace(0, 12, 25)
        episodes = defaultdict(list)
        for t in traces:
            if t["policy"] == policy and t["scenario"] in SCENARIOS:
                episodes[t["seed"], t["scenario"]].append(t)
        errors = []
        for cost in costs:
            per = []
            for values in episodes.values():
                eligible = [v for v in values if v["cost"] <= cost + 1e-9]
                per.append(eligible[-1]["effect_error"] if eligible else values[0]["initial_effect_error"])
            errors.append(np.mean(per))
        axes[1].plot(costs, errors, color=colors[policy], label=policy.replace("_", " "), linewidth=2)
    axes[0].plot([0, 1], [0, 1], ":", color="#667777")
    axes[0].set(
        xlabel="Predicted probability of positive effect",
        ylabel="Observed positive-effect frequency",
        title="Posterior calibration · 5 bins",
    )
    axes[1].set(
        xlabel="Cumulative experiment cost",
        ylabel="Mean absolute effect error",
        title="Learning progress on held-out SCMs",
    )
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(output / "calibration_learning.svg")
    fig.savefig(output / "calibration_learning.png", dpi=180)
    plt.close(fig)


def math_sqrt(x: int) -> float:
    return float(np.sqrt(x))


def run_benchmark(
    output: str | Path = "benchmarks/results",
    test_count: int = 64,
    train_count: int = 24,
    validation_count: int = 24,
) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    parameters, optimization = optimize(train_count, validation_count)
    rows, traces = [], []
    for scenario in (*SCENARIOS, "misspecified"):
        for seed in range(2000, 2000 + test_count):
            for policy in BASELINES:
                row, trace = episode(seed, scenario, policy, parameters)
                initial = design.initial_state(seed, scenario)
                truth = design.Simulator(seed, scenario).true_effect(initial)
                for t in trace:
                    t["initial_effect_error"] = abs(initial["effect_mean"] - truth)
                rows.append(row)
                traces.extend(trace)
    summary = _summary(rows)
    config = {
        "schema_version": 1,
        "train_seeds": [0, train_count - 1],
        "validation_seeds": [1000, 1000 + validation_count - 1],
        "test_seeds": [2000, 2000 + test_count - 1],
        "scenarios": [*SCENARIOS, "misspecified"],
        "budget": 12,
        "policies": list(BASELINES),
        "parameters": parameters,
        "optimization": optimization,
        "quadrature_order": 21,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": round(time.time() - start, 3),
        "test_episodes": len(rows),
        "design_source_sha256": hashlib.sha256(Path(design.__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Four known-parameter linear-Gaussian candidate SCMs, not unrestricted graph discovery",
            "95% intervals are finite-model Bayesian credible envelopes, not general frequentist confidence intervals",
            "Query-EIG is a myopic finite-model adaptation of goal-oriented Bayesian design, not a reproduction of GO-CBED's transformer policy",
            "No claim of state-of-the-art performance or universal scientific novelty",
            "Misspecification is deliberately outside the model family and reports abstention/calibration failures",
        ],
    }
    _save_csv(output / "episodes.csv", rows)
    _save_csv(output / "summary.csv", summary)
    _save_csv(output / "traces.csv", traces)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    payload = {"config": config, "summary": summary}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    _plots(output, summary, traces, rows)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results")
    parser.add_argument("--seeds", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output, args.seeds), indent=2))
