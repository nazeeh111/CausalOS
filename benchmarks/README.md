# Executed experimental-design evidence

| Study | Development | Held-out evaluation | Result |
|---|---|---|---|
| Nominal policy | Train 0–23; validate 1000–1023 | Seeds 2000–2063; 4 scenarios; 8 policies; 2,048 episodes | Optimized nominal blend equals query-EIG exactly. No original-policy superiority. |
| Candidate-family challenge | Train 3000–3023; validate 4000–4023 | Seeds 5000–5095; 3 scenarios; 8 policies; 2,304 episodes | Weak-confounding false claims 21/96→6/96 versus nominal query-EIG with the same augmented posterior and guard. Ties a strong decision-value baseline; no unseen-negative improvement. |

`results/` and `evidence_results/` retain every episode, trace, configuration, summary, and executed chart. Ground truth fields occur in evaluator outputs only. `study1-sha256.json` protects the original CSVs. `plots_provenance.json` records the correction to seed-cluster uncertainty bars without changing raw episodes. `source_equivalence.json` in the second study records nonnumerical state-isolation/default synchronizations and replay equivalence after source hashes changed.

The original `misspecified` scenario is an adversarial response-table stressor, not a realizable documented DAG. The second study uses a coherent latent-confounded Gaussian SCM. Its negative cases intentionally abstain rather than emit a negative-effect claim. Consequently `cost_to_valid_claim` is a capped endpoint for nominal positive/zero claims; the separate secondary family-alert endpoint measures model-family detection.

```bash
causalos benchmark --out artifacts/nominal-benchmark
python -m causalos.evidence_design --output artifacts/evidence-benchmark
```

For exact flags use `python -m causalos.evidence_design --help`. Development optimization precedes held-out evaluation. Do not tune on these published test seeds. [The research report](../docs/research.md) defines the acquisition, surrogate loss, baselines, paired comparisons, uncertainty and scientific limits.
