# Spend on the causal question

This is the existing built-in `fork` scenario, seed 42, default `causalos` policy and budget 12, executed from current source `1f25fabdf066358cd56963acddaa647f5349bc5c` (actual tree `221bde7ca7eccd6d9ae510ea021dc75ab0105e4b`). It is one finite-model simulation, not a new experiment-design benchmark. The simulator, likelihood parameters and priors are supplied, not identified from real intervention data. [Provenance](provenance.json) records the source hashes, policy weights and runtime versions.

## Run the scenario

After the [local setup](../../README.md#run-the-laboratory), run from the repository root:

```sh
causalos replay --seed 42 --steps 6 --out completed.json
```

The actual six-step episode spends all 12 simulated cost units. Read the saved final state and history with any JSON reader:

```sh
python -c 'import json; s=json.load(open("completed.json")); print(s["claim"], s["spent"]); print(s["posterior"]); print(s["history"])'
```

The equivalent existing library route explicitly loops until no candidate is affordable:

```python
from causalos import design
state = design.replay(seed=42)
assert state["spent"] == 12.0
assert state["claim"] == "zero effect"
```

These assertions check this recorded scenario, not a real-world causal effect. Output paths must be fresh. The public [initial state](initial-state.json), [initial action ranking](initial-ranking.json), [completed replay](completed-replay.json) and [nuisance-probe control](nuisance-control.json) are the actual records used by the figure. Their seeds reproduce within the recorded implementation and runtime; future libraries/platforms are not bitwise guarantees. No raw personal data, external model or provider was involved.

## Why the graph is not the target

The four candidate linear Gaussian SCMs combine two A–Y orientations and two unrelated B–Z orientations. Each model starts with prior 1/4; the figure groups the two A → Y models against the two Y → A models. The A–Y pairs have centered unit-Gaussian observational distributions with the same `rho=0.5739524441142261`. The A→Y model has effect rho and the Y→A model has effect 0 by construction. B–Z has its own independent ambiguity.

Two of the five initial candidate actions:

| Action | Samples | Simulated cost | Expected whole-model gain / nats | Expected query gain / nats |
| --- | ---: | ---: | ---: | ---: |
| Set B=2, read Z | 16 | 0.8 | 0.6930709032 | 0 |
| Set A=1, read Y | 8 | 2.0 | 0.2890013992 | 0.2890013992 |

The gains are model-based expectations computed with 21-point Gaussian quadrature, not realized information from one observation. The common bar scale begins at zero and ends at 0.70 nats. The default nominal policy's selected weights reduce to query information per cost (with a constant normalization), matching the retained negative finding against query-EIG. It does not use the evaluator's true graph. The actual B-probe control changes the nuisance-model weights while keeping the A→Y query probability at 0.5. A high whole-model information gain alone can spend budget on an irrelevant question in this supplied model family.

## Follow the actual evidence

| Step | Selected action | Simulated sample mean | Spent cost | P(A → Y), conditional on family |
| --- | --- | ---: | ---: | ---: |
| Prior | none | not observed | 0 | 0.5 |
| 1 | A=1, read Y | 0.1925450655 | 2 | 0.3729373180 |
| 2 | A=1, read Y | 0.2903323800 | 4 | 0.3863967446 |
| 3 | A=1, read Y | -0.1627222729 | 6 | 0.0324864867 |
| 4 | A=2, read Y | -0.3033414843 | 11 | approximately 2.36e-15 |
| 5–6 | observational measurement | see raw history | 11.5,12 | unchanged within numerical precision |

The plot shows these exact posterior states with a step function between observations, not a fitted learning curve. The y-axis is a model probability on 0–1, not empirical accuracy. The dotted 0.05 line corresponds to the conditional zero-effect claim rule: confidence at least 95% with no model-conflict alarm. At step 3 this particular state crosses that rule; the unrelated B–Z direction is still unresolved at episode end. The final two affordable actions are passive observations with zero expected query information; this implementation spends the remaining budget rather than asserting optimal stopping. Cost units are supplied synthetic costs, not money or elapsed time.

The assumption list includes one of the supplied four linear Gaussian models being correct, no omitted confounding, known likelihood parameters, independent observations and perfect interventions. Posterior concentration cannot verify those assumptions or establish that the real mechanism is represented. A predictive-conflict alarm is a heuristic rather than a calibrated sequential test. The public research also retains adversarial and hidden-confounding cases, false claims and negative results: [research report](../research.md), [scientific boundaries](../../README.md#scientific-boundaries), [analysis methods](../analysis.md). This one episode establishes no comparative policy efficacy.

## Actual laboratory and saved analyses

Run `causalos serve` and open the printed loopback address. In Sequential experiment design, choose the recommended A actions and run each simulated intervention; inspect the updated hypotheses, posterior, budget and experiment record. [The current capture](laboratory-current.png) is this actual workflow. Export report produces a readable HTML record of the state, its assumptions and software; reopen it in a browser. Simulation reports do not have a report-import control, so use the replay command to regenerate the state.

The newer current-main saved-analysis workflow is preserved separately: `analyze` exports a versioned record; `run-open` displays it unverified until `run-replay` matches its original input and recorded software. This task also exercised that route on a synthetic confounded dataset, including export, fresh-process open and matched replay. Its bounded per-analysis nuisance-estimator fitting is ordinary product estimation, not a new training study. See the complete [saved-record contract](../../README.md#save-reopen-and-replay-an-analysis). The numerical analysis, scientific assumptions and saved question are distinct from this finite-SCM episode.

## Figure regeneration

`python docs/presentation/render.py` reads these raw JSON records, checks bound numerical source hashes, and exports desktop/phone PNGs and SVGs with Matplotlib. It does not run a benchmark, fit a model or call a service. Graph arrows are schematic model orientations, not estimated spatial coordinates. Important numbers and assumptions are available in this Markdown and raw JSON, including all five actions and all six observations. No claim of general causal discovery, real-world validity or policy superiority follows from the figure.
