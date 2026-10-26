"""Observed examples. Synthetic truth remains metadata, never an estimator input."""
from __future__ import annotations
import hashlib
from importlib.resources import files
import numpy as np
import pandas as pd
from .core import ASSUMPTIONS


def list_examples() -> list[dict]:
    return [
        {"name": "confounded", "title": "Measured confounding", "kind": "synthetic", "description": "A known synthetic SCM with observed common cause X and effect 2."},
        {"name": "randomized", "title": "Randomized treatment", "kind": "synthetic", "description": "A known synthetic randomized SCM with effect 1."},
        {"name": "nonidentifiable", "title": "Unresolved causal direction", "kind": "synthetic", "description": "Observational association with unresolved orientation. Estimation abstains."},
        {"name": "collider", "title": "Post-treatment collider trap", "kind": "synthetic", "description": "Adjustment for T → C ← Y is rejected."},
        {"name": "hidden_confounding", "title": "Unobserved common cause", "kind": "synthetic", "description": "The common cause U is hidden. Backdoor identification abstains."},
        {"name": "nsw", "title": "NSW job-training experiment", "kind": "real_experimental", "description": "445 observed records from the Dehejia–Wahba NSW randomized sample; historical 1978 earnings."},
    ]


def example(name: str = "confounded", seed: int = 42, n: int = 800) -> tuple[pd.DataFrame, dict]:
    if name not in {e["name"] for e in list_examples()}:
        raise ValueError("Unknown example: " + str(name))
    rng = np.random.default_rng(seed)
    accepted = {k: True for k in ASSUMPTIONS}
    if name == "nsw":
        path = files("causalos").joinpath("resources/data/nsw.csv")
        raw = path.read_bytes()
        import io
        data = pd.read_csv(io.BytesIO(raw)).drop(columns=["rownames", "data_id"])
        covars = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
        edges = [[c, "re78"] for c in covars] + [["treat", "re78"]]
        question = {"treatment": "treat", "outcome": "re78", "covariates": covars, "graph": {"nodes": [*covars, "treat", "re78"], "edges": edges, "unresolved": [], "latent": []}, "assumptions": accepted, "graph_source": "user", "seed": seed}
        meta = {"name": name, "title": "NSW job-training randomized experiment", "kind": "real_experimental", "simulated": False, "question": question, "provenance": {"source": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/causaldata/nsw_mixtape.csv", "documentation": "https://nickch-k.r-universe.dev/causaldata/doc/manual.html#nsw_mixtape", "sha256": hashlib.sha256(raw).hexdigest(), "license": "causaldata: MIT + LICENSE; copyright 2021 Nick Huntington-Klein", "original_study": "LaLonde (1986), Evaluating the Econometric Evaluations of Training Programs with Experimental Data; Dehejia and Wahba (1999) restricted sample.", "transformations": ["Remove rownames and constant data_id labels; keep all 445 observations and all measured variables."]}, "limitations": ["Assumptions in this reproducible case are configured from the documented randomized study design; users must assess their relevance.", "The estimand is the average effect in this selected randomized study sample, not a current labor-market policy forecast.", "No ground-truth individual treatment effects or true population ATE are observed.", "Earnings are in historical 1978 dollars. Heavy tails make finite-sample uncertainty important."]}
        return data, meta
    if n < 50 or n > 20000:
        raise ValueError("Synthetic examples support 50–20,000 rows.")
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-0.7 * x))
    t = rng.binomial(1, p)
    y = 2 * t + 1.5 * x + rng.normal(size=n)
    data = pd.DataFrame({"X": x, "T": t, "Y": y})
    graph = {"nodes": ["X", "T", "Y"], "edges": [["X", "T"], ["X", "Y"], ["T", "Y"]], "unresolved": [], "latent": []}
    covars, truth = ["X"], 2.0
    if name == "randomized":
        t = rng.binomial(1, 0.5, n)
        data["T"] = t
        data["Y"] = t + 1.5 * x + rng.normal(size=n)
        graph["edges"] = [["X", "Y"], ["T", "Y"]]
        truth = 1.0
    elif name == "nonidentifiable":
        graph["edges"] = [["X", "T"], ["X", "Y"]]
        graph["unresolved"] = [["T", "Y"]]
        truth = None
    elif name == "collider":
        data["C"] = data["T"] + data["Y"] + rng.normal(size=n)
        graph["nodes"].append("C")
        graph["edges"].extend([["T", "C"], ["Y", "C"]])
        covars.append("C")
    elif name == "hidden_confounding":
        data = data.drop(columns="X")
        graph = {"nodes": ["U", "T", "Y"], "edges": [["U", "T"], ["U", "Y"], ["T", "Y"]], "latent": ["U"], "unresolved": []}
        covars = []
    question = {"treatment": "T", "outcome": "Y", "covariates": covars, "graph": graph, "assumptions": accepted, "graph_source": "user", "seed": seed}
    return data, {"name": name, "title": next(e["title"] for e in list_examples() if e["name"] == name), "kind": "synthetic", "simulated": True, "question": question, "seed": seed, "n": n, "simulation_truth": {"ate": truth, "access": "Evaluation-only metadata, never passed to estimators or design policies."}, "limitations": ["Synthetic structural causal model. Assumptions are known by construction within this simulation; this is not real-world experimental verification."]}
