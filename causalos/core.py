"""Identification precedes estimation. All causal results remain assumption-conditional.

Only binary-treatment, continuous-outcome, observed-covariate backdoor ATEs are
supported. This module deliberately does not implement the general ID algorithm.
"""
from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ASSUMPTIONS = {
    "exchangeability": "All common causes of treatment and outcome are measured and correctly represented; conditional exchangeability is assumed, not proven from the data.",
    "consistency": "Observed outcomes equal the potential outcomes for the treatment actually received; treatment versions are well defined.",
    "positivity": "Both treatment levels are possible at every covariate value in the target population. Sample overlap can diagnose failures but cannot prove this.",
    "temporal_order": "Adjustment variables precede treatment and the outcome follows treatment; timing is established from study knowledge.",
    "no_interference": "One unit's treatment does not change another unit's outcome.",
    "measurement_validity": "Treatment, covariates and outcome are measured sufficiently well for the stated model.",
    "selection_ignorability": "Selection and missingness do not induce unaccounted bias for the stated sample estimand.",
}
TRUTH_COLUMNS = {"mu0", "mu1", "mu_0", "mu_1", "y_cf", "y0", "y1", "true_effect", "ground_truth", "counterfactual", "potential_y0", "potential_y1"}


def assumption_definitions() -> dict[str, str]:
    return dict(ASSUMPTIONS)


def _blocked(result: dict, reason: str) -> dict:
    result["status"] = "blocked"
    result["identification"] = {"identified": False, "reason": reason, "estimand": "ATE", "adjustment_set": [], "formula": None}
    return result


def _d_separated(graph: nx.DiGraph, x: str, y: str, z: list[str]) -> bool:
    """D-separation by ancestral moralization (works for collider adjustment too)."""
    ancestors = {x, y, *z}
    for node in list(ancestors):
        ancestors.update(nx.ancestors(graph, node))
    ancestral = graph.subgraph(ancestors)
    moral = nx.Graph(ancestral)
    for node in ancestral:
        parents = list(ancestral.predecessors(node))
        for i, a in enumerate(parents):
            for b in parents[i + 1:]:
                moral.add_edge(a, b)
    moral.remove_nodes_from(z)
    return not nx.has_path(moral, x, y)


def identify(question: dict, columns: list[str]) -> dict:
    """Validate an explicit DAG and the supplied backdoor set, conservatively.

    Unresolved/latent graphs abstain even when a more general ID algorithm could
    identify an effect. Success is conditional on all explicitly accepted assumptions.
    """
    if not isinstance(question, dict):
        return _blocked({"estimates": [], "diagnostics": {}, "refutation": {}, "sensitivity": {}, "assumptions": {}, "limitations": []}, "The causal question must be an object.")
    base = {"status": "blocked", "estimates": [], "diagnostics": {}, "refutation": {}, "sensitivity": {}, "assumptions": question.get("assumptions", {}), "limitations": [
        "Identification is conditional on the supplied graph and accepted scientific assumptions, not established by observational fit.",
        "Only complete-data binary-treatment backdoor ATEs are supported; latent-variable ID and partial-identification bounds are not implemented.",
        "Population transport, measurement error, selection bias and interference cannot be ruled out by these diagnostics.",
    ]}
    if not isinstance(question.get("assumptions", {}), dict):
        return _blocked(base, "Assumptions must be an object of explicit boolean confirmations.")
    if not isinstance(question.get("graph_source", "user"), str) or question.get("graph_source", "user") not in {"user", "learned"}:
        return _blocked(base, "Graph source must be user or learned.")
    if not isinstance(question.get("post_treatment", []), list) or not all(isinstance(v, str) for v in question.get("post_treatment", [])):
        return _blocked(base, "Post-treatment variables must be a list of column names.")
    t, y = question.get("treatment"), question.get("outcome")
    covars = question.get("covariates", question.get("adjustment", []))
    if not isinstance(t, str) or not isinstance(y, str) or t == y:
        return _blocked(base, "Choose distinct treatment and outcome columns.")
    if not isinstance(covars, list) or not all(isinstance(v, str) for v in covars) or len(covars) != len(set(covars)):
        return _blocked(base, "Adjustment variables must be a list of unique column names.")
    if any(str(c).lower() in TRUTH_COLUMNS or str(c).lower().startswith("truth_") for c in [t, y, *covars]):
        return _blocked(base, "Simulator truth or potential-outcome columns cannot enter an estimator.")
    if any(c not in columns for c in [t, y, *covars]):
        return _blocked(base, "Treatment, outcome or adjustment column is absent from the observed data.")
    if t in covars or y in covars:
        return _blocked(base, "Treatment and outcome cannot be adjustment variables.")
    missing = [a for a in ASSUMPTIONS if question.get("assumptions", {}).get(a) is not True]
    if missing:
        return _blocked(base, "Explicit scientific confirmation is required for: " + ", ".join(missing) + ".")
    if question.get("graph_source") == "learned" and question.get("graph_confirmed") is not True:
        return _blocked(base, "A learned graph alone cannot authorize identification. Confirm its directions using study knowledge or interventions.")
    spec = question.get("graph", {})
    if not isinstance(spec, dict):
        return _blocked(base, "Provide a graph object with nodes, directed edges and unresolved edges.")
    if spec.get("unresolved"):
        return _blocked(base, "Unresolved orientations define competing graph hypotheses. This backdoor engine abstains until the relevant directions are justified.")
    if spec.get("latent") or spec.get("bidirected"):
        return _blocked(base, "Unresolved latent confounding is reported. This engine does not identify effects in latent-variable graphs.")
    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    if not isinstance(nodes, list) or not all(isinstance(v, str) for v in nodes) or len(nodes) != len(set(nodes)):
        return _blocked(base, "Graph nodes must be unique strings.")
    if any(c not in nodes for c in [t, y, *covars]):
        return _blocked(base, "The graph must contain treatment, outcome and every adjustment variable.")
    if not isinstance(edges, list) or any(not isinstance(e, (list, tuple)) or len(e) != 2 or e[0] not in nodes or e[1] not in nodes for e in edges):
        return _blocked(base, "Every directed edge must connect two declared graph nodes.")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        return _blocked(base, "Causal DAGs cannot contain a directed cycle or a self-loop.")
    missing_observed = [v for v in nodes if v not in columns]
    if missing_observed:
        return _blocked(base, "Graph includes unobserved variables: " + ", ".join(missing_observed) + ". Mark confounding explicitly; this engine abstains.")
    post = set(question.get("post_treatment", [])) | nx.descendants(graph, t)
    if set(covars) & post:
        return _blocked(base, "Post-treatment adjustment is unsupported for the total effect: " + ", ".join(sorted(set(covars) & post)) + ".")
    backdoor = graph.copy()
    backdoor.remove_edges_from(list(backdoor.out_edges(t)))
    if not _d_separated(backdoor, t, y, covars):
        return _blocked(base, "The adjustment set does not block every backdoor path in the proposed graph; collider conditioning can also open a path.")
    base["status"] = "identified"
    base["identification"] = {"identified": True, "reason": "The supplied adjustment set satisfies the backdoor criterion in the user-confirmed DAG, conditional on the accepted assumptions.", "estimand": "Sample-population average treatment effect E[Y(1)-Y(0)]", "adjustment_set": covars, "formula": "E_X[E[Y | T=1,X] - E[Y | T=0,X]]", "graph_source": question.get("graph_source", "user")}
    return base


def _ps(x: np.ndarray, t: np.ndarray, xtest: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.full(len(xtest), t.mean())
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1000, max_iter=1500, solver="lbfgs"))
    model.fit(x, t)
    return model.predict_proba(xtest)[:, 1]


def _summary(method: str, value: float, se: float, *, note: str, target: str = "ATE") -> dict:
    return {"method": method, "estimate": float(value), "se": float(se), "ci": [float(value - 1.96 * se), float(value + 1.96 * se)], "confidence_level": 0.95, "target": target, "assumptions": note}


def _estimate_all(x: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int, bootstrap: int) -> tuple[list[dict], dict, Any]:
    n = len(y)
    design = sm.add_constant(np.column_stack([t, x]), has_constant="add")
    fitted = sm.OLS(y, design).fit(cov_type="HC3")
    regression = _summary("regression", fitted.params[1], fitted.bse[1], note="Additive linear conditional mean; HC3 heteroskedasticity-robust normal interval. A constant effect or correct additive outcome specification is required.")
    raw_p = _ps(x, t, x)
    p = np.clip(raw_p, 0.01, 0.99)
    ipw_scores = t * y / p - (1 - t) * y / (1 - p)
    ipw = float(ipw_scores.mean())
    # Stratified pairs bootstrap re-fits propensity, including nuisance uncertainty.
    rng = np.random.default_rng(seed)
    samples = []
    groups = [np.flatnonzero(t == a) for a in [0, 1]]
    for _ in range(bootstrap):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        pb = np.clip(_ps(x[ix], t[ix], x[ix]), 0.01, 0.99)
        samples.append(float(np.mean(t[ix] * y[ix] / pb - (1 - t[ix]) * y[ix] / (1 - pb))))
    ipw_se = float(np.std(samples, ddof=1))
    ipw_result = _summary("ipw", ipw, ipw_se, note=f"Horvitz-Thompson ATE; correctly specified logistic propensity; propensity clipped at [0.01,0.99]. Stratified pairs bootstrap ({bootstrap} refits), percentile interval.")
    ipw_result["ci"] = [float(q) for q in np.quantile(samples, [0.025, 0.975])]
    # Classical parametric doubly-robust score, using separate group regressions.
    xx = x if x.shape[1] else np.zeros((n, 1))
    mu = []
    for a in [0, 1]:
        fit = LinearRegression().fit(xx[t == a], y[t == a])
        mu.append(fit.predict(xx))
    scores = mu[1] - mu[0] + t * (y - mu[1]) / p - (1 - t) * (y - mu[0]) / (1 - p)
    aipw = _summary("aipw", scores.mean(), scores.std(ddof=1) / math.sqrt(n), note="Parametric augmented IPW with logistic treatment model and separate linear outcome models. The point estimate is doubly robust under a correct propensity or outcome model; the displayed simple influence-score interval additionally needs both parametric nuisance models correct (or negligible nuisance bias) and regularity.")
    # Double ML for the interactive regression (IRM) ATE, not a heterogeneous-
    # effect partially-linear coefficient masquerading as the ATE.
    m0, m1, phat = np.empty(n), np.empty(n), np.empty(n)
    folds = min(5, int(np.bincount(t.astype(int)).min()))
    for fold, (train, test) in enumerate(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, t)):
        phat[test] = _ps(x[train], t[train], x[test])
        for a, destination in [(0, m0), (1, m1)]:
            selected = train[t[train] == a]
            model = RandomForestRegressor(n_estimators=80, min_samples_leaf=max(5, len(selected) // 50), max_features=1.0, random_state=(seed + fold + a) % (2 ** 32), n_jobs=1)
            model.fit(xx[selected], y[selected])
            destination[test] = model.predict(xx[test])
    phat = np.clip(phat, 0.01, 0.99)
    dml_scores = m1 - m0 + t * (y - m1) / phat - (1 - t) * (y - m0) / (1 - phat)
    dml = _summary("dml", dml_scores.mean(), dml_scores.std(ddof=1) / math.sqrt(n), note=f"Cross-fitted interactive-regression-model ATE / orthogonal AIPW score; {folds} stratified folds, random-forest outcomes and logistic propensity. Asymptotic influence-score interval needs nuisance-rate and overlap assumptions; finite-sample coverage is not guaranteed.")
    overlap = {"min": float(raw_p.min()), "max": float(raw_p.max()), "outside_005_095": float(np.mean((raw_p < 0.05) | (raw_p > 0.95))), "clipped_fraction": float(np.mean((raw_p < 0.01) | (raw_p > 0.99))), "effective_sample_size_treated": float((t / p).sum() ** 2 / ((t / p) ** 2).sum()), "effective_sample_size_control": float(((1 - t) / (1 - p)).sum() ** 2 / (((1 - t) / (1 - p)) ** 2).sum()), "proves_positivity": False}
    return [regression, ipw_result, aipw, dml], overlap, fitted


def analyze(data: pd.DataFrame, question: dict) -> dict:
    """Analyze observed columns only; no estimator has access to simulator truth."""
    if not isinstance(data, pd.DataFrame) or data.columns.duplicated().any():
        return _blocked({"estimates": [], "diagnostics": {}, "refutation": {}, "sensitivity": {}, "limitations": []}, "Supply a dataframe with unique column names.")
    result = identify(question, list(data.columns))
    if result["status"] == "blocked":
        return result
    tname, yname = question["treatment"], question["outcome"]
    covars = question.get("covariates", question.get("adjustment", []))
    subset = data[[tname, yname, *covars]]
    if len(subset) < 50:
        return _blocked(result, "At least 50 complete observations are required by this implementation.")
    if not all(pd.api.types.is_numeric_dtype(subset[c]) for c in subset):
        return _blocked(result, "Estimator inputs must be numeric. Encode categorical covariates with an audited transformation first.")
    if not np.isfinite(subset.to_numpy(dtype=float)).all():
        return _blocked(result, "Missing or non-finite estimator inputs require an explicit missing-data plan. No rows were silently dropped.")
    t = subset[tname].to_numpy(float)
    y = subset[yname].to_numpy(float)
    x = subset[covars].to_numpy(float)
    if set(np.unique(t)) != {0.0, 1.0} or min(np.bincount(t.astype(int))) < 15:
        return _blocked(result, "Treatment must be coded 0/1 with at least 15 observations in each arm.")
    design = sm.add_constant(np.column_stack([t, x]), has_constant="add")
    if np.linalg.matrix_rank(design) < design.shape[1] or len(y) < design.shape[1] + 20:
        return _blocked(result, "The regression design is rank deficient or has too few observations for its adjustment dimension.")
    if np.std(y) == 0:
        return _blocked(result, "A constant outcome cannot support the requested uncertainty analysis.")
    seed_value = question.get("seed", 42)
    if isinstance(seed_value, bool) or not isinstance(seed_value, (int, np.integer)) or not 0 <= seed_value < 2 ** 32:
        return _blocked(result, "Seed must be an integer from 0 to 4294967295.")
    bootstrap_value = question.get("bootstrap", 80)
    if isinstance(bootstrap_value, bool) or not isinstance(bootstrap_value, (int, np.integer)) or bootstrap_value < 1:
        return _blocked(result, "Bootstrap refits must be a positive integer; computation is bounded to 40–400 refits.")
    seed = int(seed_value)
    bootstrap = min(400, max(40, int(bootstrap_value)))
    rawp = _ps(x, t, x)
    if np.mean((rawp < 0.01) | (rawp > 0.99)) > 0.05:
        result["diagnostics"]["overlap"] = {"min": float(rawp.min()), "max": float(rawp.max()), "clipped_fraction": float(np.mean((rawp < 0.01) | (rawp > 0.99))), "proves_positivity": False}
        return _blocked(result, "Severe empirical overlap failure: more than 5% of fitted propensities fall outside [0.01,0.99]. No effect estimate is reported.")
    estimates, overlap, fitted = _estimate_all(x, t, y, seed, bootstrap)
    result["estimates"] = estimates
    association = _summary("unadjusted_association", y[t == 1].mean() - y[t == 0].mean(), math.sqrt(y[t == 1].var(ddof=1) / sum(t == 1) + y[t == 0].var(ddof=1) / sum(t == 0)), note="Unadjusted difference in sample means; association alone does not establish causality.", target="association")
    balance = []
    p = np.clip(rawp, 0.01, 0.99)
    for i, c in enumerate(covars):
        sd = math.sqrt((np.var(x[t == 1, i], ddof=1) + np.var(x[t == 0, i], ddof=1)) / 2)
        before = (x[t == 1, i].mean() - x[t == 0, i].mean()) / sd if sd else 0.0
        after = (np.average(x[t == 1, i], weights=1 / p[t == 1]) - np.average(x[t == 0, i], weights=1 / (1 - p[t == 0]))) / sd if sd else 0.0
        balance.append({"variable": c, "smd_before": float(before), "smd_after_ipw": float(after)})
    result["diagnostics"] = {"n": len(y), "treated": int(t.sum()), "control": int((1 - t).sum()), "association": association, "overlap": overlap, "balance": balance, "rows_dropped": 0}
    # Negative control: random treatment permutation, regression only.
    rng = np.random.default_rng(seed + 9001)
    permuted = rng.permutation(t)
    placebo_fit = sm.OLS(y, sm.add_constant(np.column_stack([permuted, x]), has_constant="add")).fit(cov_type="HC3")
    place = _summary("permuted_treatment_regression", placebo_fit.params[1], placebo_fit.bse[1], note="One seeded treatment permutation; detects gross estimator artifacts, cannot validate exchangeability.")
    place["includes_zero"] = bool(place["ci"][0] <= 0 <= place["ci"][1])
    ix = rng.choice(len(y), int(0.8 * len(y)), replace=False)
    subset_fit = sm.OLS(y[ix], design[ix]).fit(cov_type="HC3")
    stable = _summary("random_80_percent_regression", subset_fit.params[1], subset_fit.bse[1], note="Seeded 80% subset stability check; not an independent replication.")
    result["refutation"] = {"placebo": place, "subset": stable, "interpretation": "Passing a refutation check does not prove the causal assumptions. A failed check prompts investigation."}
    txfit = sm.OLS(t, sm.add_constant(x, has_constant="add")).fit() if x.shape[1] else sm.OLS(t, np.ones((len(t), 1))).fit()
    scale = math.sqrt(float(np.dot(fitted.resid, fitted.resid) / np.dot(txfit.resid, txfit.resid)))
    grid = []
    for rt in [0.01, 0.05, 0.1, 0.2, 0.3]:
        for ry in [0.01, 0.05, 0.1, 0.2, 0.3]:
            bias = math.sqrt(ry * rt / (1 - rt)) * scale
            grid.append({"r2_treatment": rt, "r2_outcome": ry, "bias": bias, "adjusted_effect": float(fitted.params[1] - bias), "effect_lower": float(fitted.params[1] - bias), "effect_upper": float(fitted.params[1] + bias)})
    result["sensitivity"] = {"model": "Single omitted-variable linear-model bias envelope using specified partial R-squared values (Cinelli & Hazlett, 2020).", "grid": grid, "interpretation": "User-specified confounder strengths, not measured hidden confounding. Bounds cover point-estimate bias in either sign; they are not confidence or causal identification intervals."}
    result["limitations"].extend(["Estimator agreement does not establish causal validity. Confidence intervals quantify sampling uncertainty under estimator assumptions, not uncertainty in the graph.", "Individual counterfactuals are not identified by an ATE analysis; no individual counterfactual outcomes are generated."])
    if overlap["outside_005_095"] > 0:
        result["limitations"].append("Some estimated propensities lie outside [0.05,0.95]; inspect overlap and sensitivity to nuisance models.")
    return result
