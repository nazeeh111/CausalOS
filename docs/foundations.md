# Research foundations and design choices

CAUSALOS separates scientific stages because each establishes a different kind of evidence: an association is an observation, a graph is a hypothesis, identification is a mathematical result conditional on assumptions, an estimator fits an identified quantity, and an experiment provides evidence under a particular intervention. A robustness check can contradict a model; passing it does not prove that all assumptions hold.

## Existing work considered before implementation

| Foundation | Useful established capability | Architectural consequence and boundary |
|---|---|---|
| [DoWhy](https://github.com/py-why/dowhy), [Sharma & Kiciman 2020](https://arxiv.org/abs/2011.04216) | Models assumptions, identifies effects, estimates and refutes | Keep these stages explicit. The application implements a deliberately narrow backdoor workflow, rather than claiming all of DoWhy's identification algorithms. |
| [EconML](https://github.com/py-why/EconML) | Orthogonal/double machine learning and heterogeneous treatment effects | Reuse the established cross-fitting and orthogonal-score approach. No new effect estimator or general heterogeneous-effect claim. |
| [causal-learn](https://github.com/py-why/causal-learn) | Constraint-based and score-based structure learning | Optional stable-PC returns partially oriented hypotheses. Fisher-Z requires appropriate continuous-data assumptions; it cannot establish causal sufficiency or direction from correlation alone. |
| [Causica](https://github.com/microsoft/causica) | Generative structural causal modeling and probabilistic causal inference | Finite structural models make the demo's assumptions and posterior auditable on CPU. This release does not claim neural-SCM scale or reuse Causica's trained models. |
| [BoTorch](https://github.com/meta-pytorch/botorch) | Bayesian acquisition optimization | A small discrete intervention menu can be scored directly with numerical integration. This avoids a heavyweight optimization stack without claiming a replacement for continuous-space Bayesian optimization. |
| [Zhang et al. 2023](https://www.nature.com/articles/s42256-023-00719-0) | Causally informed acquisition for target-mean optimization with known causal graphs | Distinguish outcome optimization from learning among competing graphs. Do not label a finite-model heuristic as a reproduction of CIV/CIV-OW. |
| [GO-CBED, Zhang et al. 2025](https://arxiv.org/abs/2507.07359) | Goal-oriented information gain and sequential causal experimental design | Goal-directed information gain is prior art. Originality here must rest on the specific implemented budgeted policy, its optimization and measured tradeoffs, with no claim to inventing the general idea. |

The narrow statistical scope is intentional: complete finite DAG hypotheses and binary-treatment average effects are easier to audit than automatically estimating every effect on an inferred graph. A partially oriented graph is preserved and estimation abstains. This is conservative: some effects can be identifiable across an equivalence class, but that more general procedure is outside this release.

## Licenses

CAUSALOS code is MIT licensed. Runtime packages remain under their own licenses; no vendored library source is relabeled. See the distribution's installed dependency metadata for transitive license texts and [data provenance](../examples/README.md) for dataset-specific rights and restrictions.

Sources were inspected for this release on 2026-10-25. The linked repositories evolve; reproducible local dependency versions are recorded separately.
