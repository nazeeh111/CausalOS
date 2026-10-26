"""Execute scientific analysis studies. Truth is evaluation-only, never a feature.

Run from the repository root: python examples/evaluate.py --ihdp /tmp/ihdp.csv
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.stats.proportion import proportion_confint
import importlib.metadata
import duckdb

# Running the script from the source tree also works without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causalos.core import analyze, identify, ASSUMPTIONS, _ps
from causalos.datasets import example
from causalos.profiling import profile
from fetch_ihdp import SHA256


def synthetic(seeds: int = 12, n: int = 1200) -> dict:
    records = []
    for seed in range(seeds):
        for name in ["confounded", "randomized"]:
            data, meta = example(name, seed=1000 + seed, n=n)
            q = meta["question"]
            q["bootstrap"] = 40
            r = analyze(data, q)
            truth = meta["simulation_truth"]["ate"]
            for e in r["estimates"]:
                records.append({"case": name, "seed": 1000 + seed, "method": e["method"], "true_ate": truth, "estimate": e["estimate"], "absolute_error": abs(e["estimate"] - truth), "ci": e["ci"], "covered": bool(e["ci"][0] <= truth <= e["ci"][1])})
        print(f"Synthetic seed {seed + 1}/{seeds}", flush=True)
    summary = []
    for case in ["confounded", "randomized"]:
        for method in ["regression", "ipw", "aipw", "dml"]:
            rows = [r for r in records if r["case"] == case and r["method"] == method]
            errors = np.array([r["estimate"] - r["true_ate"] for r in rows])
            coverage = np.mean([r["covered"] for r in rows])
            summary.append({"case": case, "method": method, "repetitions": len(rows), "bias": float(errors.mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.abs(errors).mean()), "coverage_95": float(coverage), "coverage_mc_se": float(np.sqrt(coverage * (1 - coverage) / len(rows))), "coverage_binomial_wilson_95": [float(v) for v in proportion_confint(sum(r["covered"] for r in rows), len(rows), method="wilson")] })
    abstention = []
    for name in ["collider", "hidden_confounding", "nonidentifiable"]:
        decisions = []
        for seed in range(seeds):
            d, m = example(name, seed=1000 + seed, n=n)
            r = identify(m["question"], list(d))
            decisions.append(r["status"] == "blocked")
        abstention.append({"case": name, "n": seeds, "abstention_rate": float(np.mean(decisions)), "unsupported_estimate_rate": float(1 - np.mean(decisions))})
    # Negative finding: untestable incorrect confirmations can fool any purely
    # observational backdoor gate. Here the actual effect is zero and U hidden.
    false_claims = []
    for seed in range(seeds):
        rng = np.random.default_rng(9000 + seed)
        u = rng.normal(size=n)
        t = rng.binomial(1, 1 / (1 + np.exp(-u)))
        y = 2 * u + rng.normal(size=n)
        d = pd.DataFrame({"T": t, "Y": y})
        q = {"treatment": "T", "outcome": "Y", "covariates": [], "graph": {"nodes": ["T", "Y"], "edges": [["T", "Y"]]}, "assumptions": {k: True for k in ASSUMPTIONS}, "bootstrap": 40, "seed": seed}
        r = analyze(d, q)
        e = r["estimates"][0]
        false_claims.append({"seed": 9000 + seed, "true_ate": 0.0, "conditional_estimate": e["estimate"], "ci": e["ci"], "false_nonzero": bool(not e["ci"][0] <= 0 <= e["ci"][1])})
    return {"configuration": {"seeds": seeds, "n": n, "estimand": "known constant ATE", "nuisance_training": "Cross-fitting internal to each independent SCM replicate for DML; no simulator truth is supplied."}, "summary": summary, "records": records, "abstention": abstention, "incorrect_assumption_stress_test": {"description": "An unrecorded confounder is deliberately omitted from the supplied graph and exchangeability is incorrectly confirmed. This cannot be detected from observational data by the gate.", "false_nonzero_rate": float(np.mean([r["false_nonzero"] for r in false_claims])), "records": false_claims}}


def ihdp(path: Path, seeds: int = 10) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("IHDP replication 1 checksum mismatch")
    array = np.loadtxt(path, delimiter=",")
    # Explicitly separate observed inputs from evaluator-only counterfactuals.
    observed = pd.DataFrame(array[:, [0, 1, *range(5, 30)]], columns=["T", "Y", *[f"X{i}" for i in range(25)]])
    truth = pd.DataFrame({"mu0": array[:, 3], "mu1": array[:, 4]})
    assert not set(truth).intersection(observed)
    x = observed.filter(regex="^X").to_numpy()
    t, y = observed["T"].to_numpy(), observed["Y"].to_numpy()
    records = []
    for seed in range(seeds):
        train, rest = train_test_split(np.arange(len(y)), test_size=0.4, random_state=seed + 2700, stratify=t)
        valid, test = train_test_split(rest, test_size=0.5, random_state=seed + 3700, stratify=t[rest])
        candidates = []
        for leaf in [5, 15, 30]:
            models = []
            prediction = np.empty(len(valid))
            for a in [0, 1]:
                ix = train[t[train] == a]
                model = RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf, random_state=seed + a, n_jobs=1).fit(x[ix], y[ix])
                mask = t[valid] == a
                prediction[mask] = model.predict(x[valid[mask]])
                models.append(model)
            candidates.append((mean_squared_error(y[valid], prediction), leaf, models))
        mse, leaf, models = min(candidates, key=lambda row: row[0])
        # Validation selects leaf size; validation rows never join training and
        # test rows never choose nuisance models or hyperparameters.
        mu0, mu1 = [model.predict(x[test]) for model in models]
        p = np.clip(_ps(x[train], t[train], x[test]), 0.01, 0.99)
        scores = mu1 - mu0 + t[test] * (y[test] - mu1) / p - (1 - t[test]) * (y[test] - mu0) / (1 - p)
        ate = float(scores.mean())
        se = float(scores.std(ddof=1) / np.sqrt(len(test)))
        # The truth is read only after the predictions/score are computed.
        test_truth = float((truth.iloc[test].mu1 - truth.iloc[test].mu0).mean())
        effect_prediction = mu1 - mu0
        true_effects = truth.iloc[test].mu1.to_numpy() - truth.iloc[test].mu0.to_numpy()
        ci = [ate - 1.96 * se, ate + 1.96 * se]
        records.append({"seed": seed, "n_train": len(train), "n_validation": len(valid), "n_test": len(test), "partition_disjoint": bool(not set(train) & set(valid) and not set(train) & set(test) and not set(valid) & set(test)), "selected_min_leaf": leaf, "validation_mse": float(mse), "test_ate_truth": test_truth, "heldout_aipw_ate": ate, "absolute_ate_error": abs(ate - test_truth), "ci": ci, "covered": bool(ci[0] <= test_truth <= ci[1]), "pehe": float(np.sqrt(np.mean((effect_prediction - true_effects) ** 2))), "propensity_clipped_fraction": float(np.mean((p == 0.01) | (p == 0.99))), "test_index_sha256": hashlib.sha256(np.array(test, dtype=np.int64).tobytes()).hexdigest()})
        print(f"IHDP split {seed + 1}/{seeds}", flush=True)
    return {"source": "https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP", "source_sha256": SHA256, "benchmark": "IHDP NPCI replication 1, 747 units, 25 covariates; semi-synthetic outcomes", "source_loader_reference": "https://github.com/AMLab-Amsterdam/CEVAE/blob/master/datasets.py", "license_note": "No upstream license declaration found; source data are not redistributed. Local analysis uses separately downloaded benchmark bytes.", "method": "Independent train/validation/test nuisance fitting with validation-selected RF outcome models and logistic propensity, heldout orthogonal AIPW score", "summary": {"splits": seeds, "mean_absolute_ate_error": float(np.mean([r["absolute_ate_error"] for r in records])), "mean_pehe": float(np.mean([r["pehe"] for r in records])), "coverage_95": float(np.mean([r["covered"] for r in records]))}, "records": records, "limitations": ["Ten resplits of one fixed outcome replication are dependent and are not ten independent data-generating replications.", "Semi-synthetic conditional means are known only to the evaluator; no guarantee for genuine observational causal validity follows.", "Small heldout samples, propensity clipping and nuisance misspecification can invalidate nominal coverage."]}


def nsw_sql_summary(data):
    db = duckdb.connect(":memory:")
    db.register("observations", data)
    query = "SELECT treat, COUNT(*) AS n, AVG(re78) AS mean_1978_earnings, STDDEV_SAMP(re78) / SQRT(COUNT(*)) AS standard_error FROM observations GROUP BY treat ORDER BY treat"
    rows = db.execute(query).df().to_dict("records")
    db.close()
    return {"query": query, "results": rows, "interpretation": "Descriptive group summaries; the difference alone is causal only under the study design assumptions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ihdp", type=Path)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("examples/analysis-results.json"))
    args = parser.parse_args()
    start = time.time()
    result = {"versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "duckdb"]}, "synthetic": synthetic(args.seeds)}
    d, m = example("nsw")
    result["nsw"] = {"metadata": m, "profile": profile(d, "treat", m["question"]["covariates"]), "analysis": analyze(d, m["question"]), "sql_summary": nsw_sql_summary(d)}
    if args.ihdp:
        result["ihdp"] = ihdp(args.ihdp)
    result["elapsed_seconds"] = time.time() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    colors = ["#2d665c", "#b5853a", "#65768c", "#ab5e53"]
    summary = result["synthetic"]["summary"]
    for ax, case in zip(axes, ["confounded", "randomized"]):
        rows = [r for r in summary if r["case"] == case]
        ax.bar([r["method"].upper() for r in rows], [r["rmse"] for r in rows], color=colors)
        ax.set(title=case.replace("_", " ").title(), ylabel="ATE root mean squared error")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"Executed analysis benchmark · {args.seeds} independent SCM seeds per case", fontsize=14)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(f"Saved {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
