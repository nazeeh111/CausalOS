"""Local, typed HTTP boundary. Uploaded data is processed in memory and never persisted."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from causalos import __version__, core, datasets, design, profiling
from causalos.report import build_report, clean, render_html

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="CAUSALOS LAB", version=__version__, description="Local causal evidence laboratory")


class DataRequest(BaseModel):
    data: list[dict[str, Any]] = Field(min_length=1, max_length=20000)

    @field_validator("data")
    @classmethod
    def scalar_table(cls, value):
        columns = set().union(*(row.keys() for row in value))
        if len(columns) > 40 or any(len(str(c)) > 100 for c in columns):
            raise ValueError("Use at most 40 columns with names at most 100 characters.")
        for row in value:
            for cell in row.values():
                if cell is not None and not isinstance(cell, (str, int, float, bool)):
                    raise ValueError("Cells must be text, numeric, boolean or null.")
                if isinstance(cell, str) and len(cell) > 1000:
                    raise ValueError("Text cells must be at most 1000 characters.")
        return value


class AnalysisRequest(DataRequest):
    question: dict[str, Any]


class DiscoveryRequest(DataRequest):
    alpha: float = Field(default=0.05, gt=0, lt=1)


class DesignRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    scenario: str = "fork"
    budget: float = Field(default=12, ge=0.1, le=100)


class StepRequest(BaseModel):
    state: dict[str, Any]
    action_id: str = Field(max_length=80)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class ReportRequest(BaseModel):
    data: list[dict[str, Any]] | None = None
    question: dict | None = None
    analysis: dict | None = None
    design: dict | None = None
    profile: dict | None = None
    metadata: dict | None = None
    counterfactual: dict | None = None
    evidence_design: dict | None = None
    reproducibility: dict | None = None


class CounterfactualRequest(BaseModel):
    state: dict[str, Any]
    observed: dict[str, float]
    target: str
    value: float


class EvidenceRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    scenario: Literal['nominal', 'weak_confounding', 'unseen_negative'] = 'weak_confounding'


class EvidenceStepRequest(EvidenceRequest):
    state: dict[str, Any]
    action_id: str = Field(max_length=80)


def safe(call, *args, **kwargs):
    try:
        return clean(call(*args, **kwargs))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def with_recommendations(state: dict) -> dict:
    state = dict(state)
    state["recommendations"] = safe(design.recommend, state)
    return clean(state)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "mode": "local", "external_model_api": False}


@app.get("/api/examples")
def list_examples():
    return {"examples": datasets.list_examples()}


@app.get("/api/examples/{name}")
def get_example(name: str, seed: int = 42):
    data, metadata = safe(datasets.example, name=name, seed=seed)
    question = metadata.get("question", {})
    return clean({"data": data.to_dict(orient="records"), "metadata": metadata,
                  "question": question, "profile": safe(profiling.profile, data,
                  treatment=question.get("treatment"), covariates=question.get("covariates", []))})


@app.post("/api/profile")
def profile_data(body: DataRequest):
    return safe(profiling.profile, pd.DataFrame(body.data))


@app.post("/api/analyze")
def analyze(body: AnalysisRequest):
    return safe(core.analyze, pd.DataFrame(body.data), body.question)


@app.post("/api/discover")
def discover(body: DiscoveryRequest):
    from causalos.discovery import discover as search
    return safe(search, pd.DataFrame(body.data), body.alpha)


@app.get("/api/design")
def initial_design(seed: int = 42):
    return with_recommendations(safe(design.initial_state, seed=seed))


@app.post("/api/design/reset")
def reset_design(body: DesignRequest):
    return with_recommendations(safe(design.initial_state, seed=body.seed, scenario=body.scenario, budget=body.budget))


@app.post("/api/design/step")
def step_design(body: StepRequest):
    if len(json.dumps(body.state)) > 200000 or len(body.state.get("history", [])) > 100:
        raise HTTPException(422, "Experiment state is too large; reset the simulation.")
    return with_recommendations(safe(design.step, body.state, body.action_id, seed=body.seed))


@app.get("/api/benchmarks")
def benchmarks():
    paths = [ROOT / "resources" / "benchmark_summary.json", ROOT.parent / "benchmarks" / "results" / "summary.json"]
    for path in paths:
        if path.exists():
            return json.loads(path.read_text())
    return {"status": "pending", "message": "Run causalos benchmark to generate measured results."}


@app.post("/api/counterfactual")
def counterfactual(body: CounterfactualRequest):
    from causalos.counterfactual import compare
    return safe(compare, body.state, body.observed, body.target, body.value)


def evidence_recommendations(state: dict) -> dict:
    from causalos import evidence_design
    result = dict(state)
    result['recommendations'] = safe(evidence_design.recommend, state, decision_weight=0.5)
    return clean(result)


@app.get('/api/evidence')
def initial_evidence(seed: int = 42):
    from causalos import evidence_design
    return evidence_recommendations(safe(evidence_design.initial_state, seed=seed, contamination_prior=0.05))


@app.post('/api/evidence/reset')
def reset_evidence(body: EvidenceRequest):
    from causalos import evidence_design
    return evidence_recommendations(safe(evidence_design.initial_state, seed=body.seed,
                                         scenario=body.scenario, contamination_prior=0.05))


@app.post('/api/evidence/step')
def step_evidence(body: EvidenceStepRequest):
    from causalos import evidence_design
    if len(json.dumps(body.state)) > 250000 or len(body.state.get('history', [])) > 100:
        raise HTTPException(422, 'Evidence state is too large; reset the simulation.')
    return evidence_recommendations(safe(evidence_design.step, body.state, body.action_id,
                                         seed=body.seed, scenario=body.scenario))


@app.post("/api/report")
def report(body: ReportRequest, format: str = "html"):
    data = None
    if body.data is not None:
        data = pd.DataFrame(DataRequest(data=body.data).data)
    result = build_report(data=data, question=body.question, analysis=body.analysis,
                          design=body.design, profile=body.profile, metadata=body.metadata,
                          counterfactual=body.counterfactual, evidence_design=body.evidence_design,
                          reproducibility=body.reproducibility)
    if format == "json":
        return result
    return HTMLResponse(render_html(result), headers={"Content-Disposition": 'attachment; filename="causalos-evidence.html"'})


class VersionCheckedStaticFiles(StaticFiles):
    """Revalidate local assets so an upgraded backend cannot retain an older UI."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers['Cache-Control'] = 'no-cache'
        return response


demo_path = ROOT.parent / 'docs' / 'demo'
if demo_path.exists():
    app.mount('/public-demo', VersionCheckedStaticFiles(directory=demo_path, html=True), name='public-demo')
app.mount("/", VersionCheckedStaticFiles(directory=ROOT / "web", html=True), name="lab")
