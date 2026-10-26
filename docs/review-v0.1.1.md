# v0.1.1 interface and public-release review

A fresh reviewer used the published v0.1.0 demo, repository and release, then independently inspected the refined application. The review was read-only and separate from implementation. It found no demonstrated blocking estimation or publication defect.

## Findings and fixes

| Finding | Change |
| --- | --- |
| Scientific working text was too small, and experiment selection was difficult to find. | Larger body, form, table and caution text; a causal-question summary; visible probability and round count; a direct experiment shortcut; aligned experiment rows. |
| Full-model information gain appeared beside a recommendation based on query information. | Query gain, whole-model gain and cost are separately labeled, with the nuisance-pair distinction and information units explained. |
| The canonical CSV control could not be reached with the keyboard. | A focusable native file input with a visible focus ring. The public replay instead links to local installation for custom data. |
| Synthetic EDA used the generic user-supplied provenance fallback. | A synthetic-specific description identifies the recorded structural model and its limits. |
| Latent alternatives had duplicate visible names. | H1–H8 identifiers and available directed-edge descriptions distinguish the rows. |
| A long raw table buried the research figures. | Scoped positive and negative findings and executed figures lead. All 12 second-study and 32 nominal comparisons remain in expandable tables; full JSON remains available. |
| List and graph updates discarded keyboard focus. | Focus stays on the selected experiment or graph node; deleting an edge moves focus to another relevant control. |
| Browser caching could mix assets from different releases. | Public script/style URLs include the release version; fixture requests share that key and revalidate. The build check rejects missing asset keys. |

The cream-and-cobalt identity, local Python application, scientific boundaries, datasets, estimators and experimental policies are retained. No benchmark episode or statistical result was changed.

## Executed verification

- 61 Python tests and Ruff passed locally. Python 3.11/3.12 CI and public-replay checks remain required in the published workflow.
- Seven JavaScript/Python scenario sequences agree within 3.70e-16; all six built-in analyses, 18 modified-analysis blocks and report escaping pass the generated-demo check.
- Independent inspection found all four views fit 1280px desktop and 390px mobile widths. Tables scroll within their containers. This is bounded browser verification, not full accessibility certification.
- Nominal interventions update posterior, rounds and budget. Selecting the nuisance probe changes B–Z beliefs without changing P(A → Y). Query and whole-model gains remain visible at narrow widths.
- Keyboard Enter opens the canonical CSV chooser. A separate successful IAB upload imported the task-owned 12-row CSV, produced its profile and cleared every identifying-assumption confirmation. The reviewer's Chrome file-setting automation stalled; that route is not counted as a successful upload.
- Public custom-data controls identify the local-only boundary. Edited assumptions clear previous estimates and block unsupported reuse of precomputed results.
- A real browser export downloaded the HTML report. Its parsed evidence record contained the three completed rounds, six remaining budget units and no stale fitted estimate. The reviewer's separate Chrome download observation stalled and is not counted.
- A separate final keyboard regression check ran a valid four-estimator analysis, selected X and then T with Enter/Tab/Enter, and verified that changing X → T removed all old estimates while retaining focus on T. Selecting the nuisance experiment with Enter retained its focus and pressed state; Tab then reached Run intervention.
- Reduced-motion emulation disables the inspected posterior and button transitions. Text, selected-state and focus cues remain available without motion.
- Genuine current-source screenshots show the [desktop overview](../assets/lab-v0.1.1-desktop.png), [experiment table](../assets/lab-v0.1.1-experiments.png) and [mobile layout](../assets/lab-v0.1.1-mobile.png).

The [original research limitations](research.md) still apply. The original v0.1.0 tag, release, evidence exports and launch film remain preserved. This patch improves the interface and release coherence; it adds no claim of broader causal validity or scientific superiority.
