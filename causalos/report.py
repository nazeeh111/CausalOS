"""Portable, escaped research reports with exact configurations and data fingerprints."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from causalos import __version__


def clean(value: Any) -> Any:
    """Turn numerical results into strict JSON, retaining missing values as null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def fingerprint(data: pd.DataFrame) -> str:
    return hashlib.sha256(data.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def build_report(data: pd.DataFrame | None = None, question: dict | None = None,
                 analysis: dict | None = None, design: dict | None = None,
                 profile: dict | None = None, metadata: dict | None = None,
                 counterfactual: dict | None = None, evidence_design: dict | None = None,
                 reproducibility: dict | None = None) -> dict:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "networkx", "duckdb"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return clean({
        "format": "causalos-evidence-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"causalos": __version__, "python": platform.python_version(), "packages": packages},
        "dataset": {"rows": len(data), "columns": list(data.columns), "sha256": fingerprint(data)}
        if data is not None else None,
        "source_metadata": metadata,
        "question": question, "analysis": analysis, "profile": profile, "experiment": design,
        "counterfactual": counterfactual,
        "validity_experiment": evidence_design,
        "reproducibility_configuration": reproducibility,
        "interpretation": {
            "association": "An observed relationship does not establish a causal effect.",
            "identification": "Identification is conditional on the supplied graph and explicit, partly untestable assumptions.",
            "simulation": "Intervention replay is simulated evidence within a specified finite model family, not real-world experimental verification.",
            "counterfactuals": "Individual counterfactual claims require an appropriate structural model; no individual counterfactual is inferred from an average treatment effect.",
        },
    })


def render_html(report: dict) -> str:
    def block(title: str, value: Any) -> str:
        if value is None:
            return ""
        text = html.escape(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False))
        return f"<section><h2>{html.escape(title)}</h2><pre>{text}</pre></section>"

    content = "".join(block(title, report.get(key)) for title, key in (
        ("Interpretation boundaries", "interpretation"), ("Dataset provenance", "dataset"),
        ("Source and study metadata", "source_metadata"),
        ("Causal question and identifying assumptions", "question"), ("Analysis evidence", "analysis"),
        ("Data validation and SQL exploration", "profile"), ("Simulated sequential experiments", "experiment"),
        ("Structural-model counterfactual comparison", "counterfactual"),
        ("Candidate-family challenge", "validity_experiment"),
        ("Exact simulation configuration", "reproducibility_configuration"),
        ("Reproducibility environment", "software")))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CAUSALOS · Research evidence</title>
<style>body{{background:#f5f3eb;color:#192c34;font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:48px auto;padding:24px}}
h1{{font:64px Georgia,serif;margin:0}}h2{{font-size:22px;border-bottom:1px solid #b7c0bc;padding-bottom:12px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:24px;border:1px solid #d5dad2;font:13px/1.6 ui-monospace,monospace}}
header p{{max-width:700px}}section{{break-inside:avoid}}@media print{{body{{margin:0;background:white}}pre{{font-size:10px}}}}</style>
<header><p>CAUSALOS LAB / EVIDENCE RECORD</p><h1>Research evidence.</h1>
<p>A portable record of assumptions, diagnostics, estimates and simulation. Conclusions apply only within the documented scope.</p>
<p>Generated {html.escape(report.get('created_utc', ''))}</p></header>{content}</html>"""
