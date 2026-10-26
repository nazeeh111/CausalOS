from copy import deepcopy
import numpy as np
import pytest
from causalos.core import analyze, identify, _d_separated
from causalos.datasets import example
import networkx as nx


def test_assumptions_required_not_inferred_from_graph():
    d, m = example()
    q = deepcopy(m["question"])
    q["assumptions"] = {}
    r = analyze(d, q)
    assert r["status"] == "blocked" and not r["estimates"]
    q = deepcopy(m["question"])
    q["graph_source"] = "learned"
    assert identify(q, list(d))["status"] == "blocked"
    q["graph_confirmed"] = True
    assert identify(q, list(d))["status"] == "identified"


@pytest.mark.parametrize("name", ["collider", "nonidentifiable", "hidden_confounding"])
def test_invalid_cases_abstain(name):
    d, m = example(name)
    r = analyze(d, m["question"])
    assert r["status"] == "blocked" and not r["estimates"]


def test_unblocked_backdoor_and_cycle_abstain():
    d, m = example()
    q = deepcopy(m["question"])
    q["covariates"] = []
    assert identify(q, list(d))["status"] == "blocked"
    q["graph"]["edges"].append(["Y", "X"])
    assert "cycle" in identify(q, list(d))["identification"]["reason"]


def test_moralization_detects_opening_a_collider():
    graph = nx.DiGraph([("A", "T"), ("A", "C"), ("B", "C"), ("B", "Y")])
    assert _d_separated(graph, "T", "Y", [])
    assert not _d_separated(graph, "T", "Y", ["C"])
    q = {"treatment": "T", "outcome": "Y", "covariates": ["C"], "graph": {"nodes": list(graph), "edges": [list(e) for e in graph.edges]}, "assumptions": example()[1]["question"]["assumptions"]}
    assert identify(q, list(graph))["status"] == "blocked"


def test_ground_truth_is_never_an_adjustment_variable():
    d, m = example()
    d["mu0"] = d["Y"] - 2 * d["T"]
    q = deepcopy(m["question"])
    q["covariates"].append("mu0")
    q["graph"]["nodes"].append("mu0")
    r = analyze(d, q)
    assert r["status"] == "blocked" and "truth" in r["identification"]["reason"]


def test_no_silent_missing_data_drop():
    d, m = example()
    d.loc[0, "X"] = np.nan
    r = analyze(d, m["question"])
    assert r["status"] == "blocked" and "missing" in r["identification"]["reason"].lower()


def test_numerical_effect_recovery():
    d, m = example(seed=17, n=1600)
    m["question"]["bootstrap"] = 40
    r = analyze(d, m["question"])
    assert r["status"] == "identified"
    assert {e["method"] for e in r["estimates"]} == {"regression", "ipw", "aipw", "dml"}
    for estimator in r["estimates"]:
        assert abs(estimator["estimate"] - 2) < 0.3
        assert estimator["ci"][0] < estimator["ci"][1]
    assert r["diagnostics"]["rows_dropped"] == 0
    assert r["diagnostics"]["balance"][0]["smd_after_ipw"] < 0.1


def test_empirical_overlap_blocks_extrapolative_estimation():
    d, m = example(n=400)
    d["T"] = (d.X > 0).astype(int)
    r = analyze(d, m["question"])
    assert r["status"] == "blocked" and not r["estimates"]
    assert "overlap" in r["identification"]["reason"].lower()


def test_packaged_real_data_has_provenance_and_no_truth():
    d, m = example("nsw")
    assert len(d) == 445 and int(d.treat.sum()) == 185
    assert not m["simulated"] and "simulation_truth" not in m
    assert len(m["provenance"]["sha256"]) == 64
    assert "rownames" not in d


@pytest.mark.parametrize("bad", [None, [], "confirm", 1])
def test_malformed_assumptions_block_without_exceptions(bad):
    d, m = example()
    q = deepcopy(m["question"])
    q["assumptions"] = bad
    r = analyze(d, q)
    assert r["status"] == "blocked" and not r["estimates"]


@pytest.mark.parametrize("bad", [None, [], "42", -1, float("nan"), float("inf"), 1.5, True, 2 ** 32])
def test_malformed_seed_blocks_without_exceptions(bad):
    d, m = example()
    q = deepcopy(m["question"])
    q["seed"] = bad
    r = analyze(d, q)
    assert r["status"] == "blocked" and "Seed" in r["identification"]["reason"]


@pytest.mark.parametrize("bad", [None, [], "DAG"])
def test_malformed_graph_blocks_without_exceptions(bad):
    d, m = example()
    q = deepcopy(m["question"])
    q["graph"] = bad
    assert analyze(d, q)["status"] == "blocked"


def test_malformed_question_blocks_without_exception():
    assert identify(None, ["T", "Y"])["status"] == "blocked"
