"""Optional PC search: return an equivalence-class hypothesis, never identification."""
from __future__ import annotations

import numpy as np
import pandas as pd


def discover(data: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Fisher-Z PC for small continuous, complete tables under strong assumptions.

    Matrix interpretation follows causal-learn's PC documentation. Ambiguous
    and bidirected edges are preserved rather than silently forced into a DAG.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:
        raise ValueError("PC search needs the optional dependency: pip install 'causalos-lab[discovery]'.") from exc
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    if not 2 <= len(data.columns) <= 10 or not 30 <= len(data) <= 5000:
        raise ValueError("PC search supports 2–10 columns and 30–5000 rows in this CPU example.")
    if not all(pd.api.types.is_numeric_dtype(data[c]) for c in data.columns):
        raise ValueError("PC Fisher-Z search requires numeric continuous columns.")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Handle missing or non-finite data explicitly before structure search.")
    if any(data[c].nunique() < 5 for c in data.columns):
        raise ValueError("Fisher-Z is a continuous Gaussian test; do not apply it to binary/discrete columns.")
    if np.linalg.matrix_rank(np.cov(values, rowvar=False)) < len(data.columns):
        raise ValueError("Columns have a singular covariance matrix; remove deterministic duplicates.")
    model = pc(values, alpha=alpha, indep_test="fisherz", stable=True, show_progress=False)
    matrix = model.G.graph
    names = list(map(str, data.columns))
    edges, unresolved, bidirected = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left, right = int(matrix[i, j]), int(matrix[j, i])
            if (left, right) == (-1, 1):
                edges.append([names[i], names[j]])
            elif (left, right) == (1, -1):
                edges.append([names[j], names[i]])
            elif (left, right) == (1, 1):
                bidirected.append([names[i], names[j]])
                unresolved.append([names[i], names[j]])
            elif left != 0 or right != 0:
                unresolved.append([names[i], names[j]])
    return {
        "status": "hypothesis", "graph_source": "learned", "algorithm": "stable-PC / Fisher-Z",
        "alpha": alpha, "graph": {"nodes": names, "edges": edges, "unresolved": unresolved,
                                    "latent": [], "bidirected": bidirected},
        "identifiable": False,
        "assumptions": ["Causal sufficiency (no latent common causes)", "Acyclicity", "Causal Markov and faithfulness",
                        "Independent observations", "Linear Gaussian conditional independence tests"],
        "limitations": ["Finite-sample test errors can create or orient spurious edges.",
                        "Undirected edges represent unresolved orientation in an equivalence class.",
                        "Learned structure alone never authorizes a causal estimate; supply substantive assumptions and an eligible graph."],
        "source": "https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html",
    }
