# CAUSALOS: decision-aware experimental design with auditable causal evidence

This release is a reproducible research prototype. Its contribution is an implemented and optimized **cost-aware blend of expected decision-regret reduction and causal-query information**, joined to explicit evidence states, posterior-predictive conflict alerts, and a causal-analysis laboratory. It is not a claim to have invented goal-oriented experimental design or solved general causal discovery. All benchmark graphs and intervention outcomes in this document are synthetic unless explicitly identified otherwise in the dataset case studies.

## Why this architecture

We examined the requested foundations before choosing a small CPU-reproducible experiment-design testbed:

| Foundation | What it establishes or provides | Consequence for CAUSALOS |
|---|---|---|
| [DoWhy paper](https://arxiv.org/abs/2011.04216) and [source](https://github.com/py-why/dowhy) | A workflow separating modeling, identification, estimation and refutation. | Keep identifying assumptions visible; estimation cannot establish them. |
| [EconML](https://github.com/py-why/EconML) | Orthogonal machine-learning estimators for treatment effects under identifying conditions. | Compare established adjustment and orthogonal estimators only after the identification gate. Flexible nuisance prediction cannot remove hidden confounding. |
| [causal-learn](https://github.com/py-why/causal-learn) | Causal-discovery algorithms and conditional-independence tools. | Unresolved orientations belong in equivalence classes. A fitted graph is evidence conditional on assumptions, not a causal certificate. |
| [Causica](https://github.com/microsoft/causica) | DECI learns a variational graph distribution and additive-noise structural mechanisms; simulated interventions use those fitted models. | Model-derived intervention predictions remain conditional on the family and fit. A model posterior need not contain the real mechanism. |
| [BoTorch](https://github.com/meta-pytorch/botorch) | General differentiable Monte Carlo acquisition-function infrastructure. | Its machinery is useful for larger continuous action spaces; exact finite actions and Gaussian quadrature avoid a large dependency here. |
| [Zhang et al., 2023](https://doi.org/10.1038/s42256-023-00719-0), [author preprint](https://arxiv.org/abs/2209.04744), [code](https://github.com/uhlerlab/actlearn_optint) | Causal integrated variance for target-mean optimization with a known linear Gaussian graph. | This is an important precedent, but its known-graph optimization problem differs from our unknown-orientation question. We do not label a different formula a CIV reproduction. |
| [GO-CBED, 2025](https://arxiv.org/html/2507.07359v1) | Goal-oriented, non-myopic Bayesian causal design with a learned transformer policy and variational posterior. | Query-specific information gain is established prior art. Our `query_eig` is a myopic finite-model adaptation of that goal-oriented criterion, not a reproduction of its trained architecture or published results. |

All five software foundations use MIT licenses in their repositories, with additional notices applying to some EconML components. CAUSALOS does not vendor their implementation code. Bibliographic entries are in [references.bib](references.bib); dependency licensing remains the upstream authors' property.

## Controlled structural causal models

Let the standardized observational pair `(A,Y)` have correlation `rho`. Two candidate mechanisms are:

```
Forward: A = eps_A; Y = rho A + sqrt(1-rho²) eps_Y
Reverse: Y = eps_Y; A = rho Y + sqrt(1-rho²) eps_A
```

The independent noises are standard Gaussian. Both models imply exactly the same observational covariance `[[1,rho],[rho,1]]`. Neither observational fit nor sample size orients this pair. The forward model gives `E[Y | do(A=a)] = rho*a`; the reverse model gives zero. An independent `(B,Z)` pair with correlation `.8` has the same directional ambiguity. Combining the two orientations gives four candidate SCMs. Cheap experiments on B can identify a nuisance arrow while leaving the A-to-Y question unchanged.

Every action observes the mean of `n` independent post-intervention samples. Its likelihood is Gaussian with the SCM-implied mean and standard error, including the residual-variance difference between the two directions. The simulator samples this sample-mean summary directly, which is distributionally equivalent to generating all `n` observations and averaging them. Because variances differ between hypotheses, the mean is not sufficient for discrimination using the full individual-level sample; all compared policies receive the same reduced observation channel. No hidden potential outcome or oracle graph is included in public policy state.

The `fork`, `weak` and `expensive` scenarios vary observational correlation or reverse-probe cost. Action cost is an abstract budget unit, not a measured monetary expense. Candidate doses, sample sizes and costs are a controlled experimental menu. An additional observational measurement is a zero-information negative control for arrow orientation.

## Policy and offline optimization

For posterior model weights `p`, let `Q` indicate a positive A-to-Y effect. The deployment decision chooses whether to adopt `do(A=+1)`, with utility `effect - rho/2`. The normalized posterior Bayes regret is `R(p)=min(P(Q=1),P(Q=0))`.

For each feasible action, integrate over its posterior predictive distribution to compute:

- Model information: `H(p) - E[H(p_after)]`.
- Query information: `H(Q) - E[H(Q_after)]`.
- Decision value: `R(p) - E[R(p_after)]`.

Gaussian Gauss-Hermite quadrature uses 21 nodes per generating hypothesis. Likelihood updates use logarithms and log-sum-exp for numerical stability. The optimized acquisition is:

```
score(a) = [w_D * 2 * decision_value(a)
            + w_Q * query_information(a)/log(2)
            + w_G * model_information(a)/log(4)] / cost(a)^eta
```

The current research sweep fixes `w_G=0` and `w_Q=1-w_D`. It evaluates nine settings from `w_D in {0,.5,1}` and `eta in {0,1,1.5}` on training seeds 0–23. It chooses among the training top three on validation seeds 1000–1023. Evaluation uses untouched test seeds 2000–2063. The full selected configuration and sweep losses are retained in `benchmarks/results/config.json`. This is offline parameter optimization rather than neural policy training. Simulator truth is used for the development objective and held-out evaluation only; action selection receives predictive models and current posterior weights.

The optimization objective is mean normalized decision regret plus `0.20*query_Brier + 0.04*cost_to_95/budget`. Seeds are disjoint. Matched test episodes use common random numbers indexed by episode, action and its repeat count, so a different policy does not receive an arbitrary algorithm-specific noise stream. Train/validation selection is complete before test evaluation.

## Baselines, ablations and measured outcomes

The benchmark compares random affordable actions, maximum predictive variance (`uncertainty`), full-model information gain, query information per cost, expected decision value per cost, the optimized blend, and its no-cost and no-decision ablations. All share the same candidate family, posterior updates, menu, budget and simulator. The strongest relevant comparison is query-EIG and decision-value per cost, not only random selection.

The reported measurements include:

- Decision regret and treatment-effect RMSE against evaluator-only truth.
- Cost to first 95% query confidence, capped at budget + `.5` when never attained, plus the attainment rate. Confidence is conditional on the candidate family.
- Query Brier score, a proper probabilistic scoring rule; lower is better.
- Empirical coverage of 95% **finite-model Bayesian credible envelopes**. These are not general frequentist confidence intervals for an estimated arbitrary treatment effect.
- Graph structural Hamming distance, counting an orientation reversal as one error, and exact graph recovery within the four-model family.
- False causal claims, abstention, and predictive-conflict alerts. A confident zero-effect claim is false when the omitted true effect is negative.

`episodes.csv` preserves every evaluated episode; `traces.csv` preserves learning progress. `summary.csv` reports means and standard errors across episodes. These standard errors describe this benchmark distribution, not uncertainty over all real causal problems. Cost-to-confidence and regret are distinct: a highly confident wrong posterior can be cheap and scientifically harmful.

Plots are generated from executed CSV/array results using Matplotlib, never image generation. See `policy_comparison.svg` and `calibration_learning.svg` in the results folder.

## Executed first-study result: the new blend did not beat the strong baseline

The training/validation procedure selected `decision_weight=0`, `query_weight=1`, `cost_power=1`. Thus the optimized nominal policy **is exactly the query-EIG-per-cost baseline**. Removing the decision component changes nothing. This is a negative finding about the proposed mixture, not evidence of a new acquisition-method advantage.

Across the 64 held-out seeds and three in-family scenarios (192 episodes per policy), the measured means are:

| Policy | Cost to 95% query confidence ↓ | Decision regret ↓ | Query Brier ↓ | Exact whole-graph recovery ↑ |
|---|---:|---:|---:|---:|
| Random | 6.615 | 0.00883 | 0.03061 | 84.4% |
| Predictive uncertainty | 6.608 | 0.00410 | 0.01577 | 97.4% |
| Whole-model information gain | 6.224 | 0.00085 | 0.00523 | 99.5% |
| Query-EIG / optimized nominal CAUSALOS | 4.744 | 0.00085 | 0.00277 | 54.7% |
| Decision value per cost | 5.695 | 0.00085 | 0.00290 | 49.0% |

Query-focused design uses less cost to resolve its question, while often leaving the independent nuisance edge unresolved. This is an intended task tradeoff, not superior full-graph discovery. Cost-to-confidence is capped at 12.5 for failures; query-EIG attained the threshold in 98.4% of these episodes. No policy made a false causal claim in these 192 in-family episodes. This finite observation does not establish zero future error. Credible-envelope coverage was 100%, reflecting conservative finite-atom intervals with known parameters; it does not validate generic causal confidence intervals.

[Executed policy chart](../benchmarks/results/policy_comparison.svg) uses within-seed means across the three scenarios before calculating standard errors over 64 independent seeds. [Raw episode CSV](../benchmarks/results/episodes.csv), [configuration and full sweep](../benchmarks/results/config.json), [learning/calibration chart](../benchmarks/results/calibration_learning.svg), and [plot provenance](../benchmarks/results/plots_provenance.json) are retained. Formatting and plot-error-bar corrections do not rerun or replace the first study's test episodes.

## Model misspecification, abstention and limits

The first study also contains a fourth, deliberately adversarial **response-table stress test**, labeled `misspecified` in its raw files. It assigns negative responses in both intervention directions. This is not a documented acyclic SCM and must not be interpreted as a realizable extra DAG or a causal graph-recovery case. It tests the software response to anomalous measurements only. The second study below uses a coherent latent-confounded SCM instead. An observation more than 3.5 predictive standard errors from every currently plausible candidate raises a conflict alert and withholds causal claims. This heuristic is an alarm, not a calibrated sequential hypothesis test; repeated inspection changes its false-alert rate. Failure to trigger an alarm does not validate the family. An apparently certain posterior can be wrong when every candidate is wrong.

The study is deliberately small. It conditions on known mechanism coefficients, a four-element hypothesis set and perfect interventions with known targets. It does not infer a general nonlinear SCM from raw observational data, learn arbitrary latent confounding, or establish real-world interventional validity. The experiment-design benchmark is separate from effect-estimation case studies. It is neither a direct replication nor a superiority test against published GO-CBED or CIV implementations. The real-world dataset analysis cannot validate the simulated experimental-design policy.

A general-purpose discovery policy, unknown mechanisms, nonlinear families, real active interventions and publication-level novelty assessment remain future research. An optimized blend that fails to improve on query-EIG is a useful negative finding, and must be reported as such.


## Second study: challenge the candidate model family

The first result motivated a materially different experiment-selection question: when should the laboratory collect evidence that could invalidate its own candidate family? The second study was specified before its execution in [evidence-study-preregistration.json](../benchmarks/evidence-study-preregistration.json), using new development seeds 3000–3023, validation seeds 4000–4023 and a locked test set 5000–5095. It does not repurpose the first test set to tune a claimed improvement.

`evidence_design.py` adds a fixed contamination family with negative treatment-effect points `theta=-0.15,-0.35`. These shadow models include an unobserved common cause. With independent standard Gaussian `U,eps_A,eps_Y`, define:

```
A = sqrt(c) U + sqrt(1-c) eps_A, c=0.95
Y = theta A + gamma U + sigma eps_Y
gamma = (rho-theta)/sqrt(c)
sigma² = 1-theta²-gamma²-2theta*gamma*sqrt(c)
```

The generated parameter ranges ensure positive residual variance. Every such model has the same unit-variance observational covariance and correlation `rho` as the nominal forward/reverse pair. Nevertheless, `E[Y|do(A=a)]=theta*a` with variance `1+theta²-2theta*rho`, while `E[A|do(Y=y)]=0` with variance one. Unit tests verify these formulas directly. The shadow grid is fixed and supplied to every augmented-family policy. Evaluator effects are drawn independently and continuously from `[-.32,-.08]`; a stronger stress distribution `[-.65,-.40]` lies outside that grid. Neither true theta nor the generating model is in public policy state.

The policy's information target has three classes: nominal positive effect, nominal zero effect, and nominal-family inadequacy. It combines expected information about this target with expected reduction of a **surrogate** three-class classification risk that allows rejection at loss 0.10. This risk is not identical to the operational claim rule. Operational positive/zero claims still require 95% conditional nominal confidence, and a shadow posterior over 0.20 withholds either claim. A candidate-family alert means the specified alternatives explain the simulated evidence better; it does not prove a real-world negative effect or identify arbitrary omitted mechanisms.

Contamination prior and risk-mixture weight are selected on the new development/validation seeds only. All comparators share the same augmented posterior, action menu, noise, budget, and evaluator. The critical acquisition comparison is `evidence` against **nominal query-EIG with the identical shadow guard**. Additional controls include information gain over the augmented full model, augmented validity-query EIG, validity decision-value, a no-guard ablation and random actions with the guard. Any difference attributable only to adding support or withholding claims must be credited to that safeguard, not to improved selection.

The locked cost-to-valid-claim metric concerns positive/zero claims. It is necessarily censored in the negative-effect stress scenarios, where abstention is the desired behavior and the system does not issue a negative-effect causal claim. A secondary cost-to-first-correct-family-alert endpoint is computed from preserved experiment traces to assess alert efficiency. This secondary endpoint was added after independent methodological review and does not change the locked development objective.

### Executed second-study result

The selected development configuration is contamination prior `0.05` and surrogate decision weight `0.5`. The held-out evaluation contains **2,304 episodes**, 96 independent seeds for each of three scenarios and eight policies. This retains a nonzero validity-risk component, unlike the first study. The primary interpretation is a supported improvement over a nominal-query acquisition **with the same model-risk safeguard**, with important limits against stronger augmented-family baselines.

| Policy | False claims, nominal | False claims, weak hidden confounding | False claims, stronger unseen negative mechanism |
|---|---:|---:|---:|
| Nominal query-EIG, no guard | 0/96 | 88/96 | 95/96 |
| Nominal query-EIG, same shadow guard | 0/96 | 21/96 | 1/96 |
| Validity-query information per cost | 0/96 | 7/96 | 2/96 |
| Validity decision-value per cost | 0/96 | 6/96 | 1/96 |
| CAUSALOS evidence blend | 0/96 | 6/96 | 1/96 |

In weak hidden confounding, CAUSALOS reduces false zero-effect claims from **21.9% to 6.25%**, a paired reduction of **15.6 percentage points**, compared with nominal query-EIG plus the identical guard. A paired percentile bootstrap over the 96 seeds gives a 95% interval of **7.3 to 24.0 percentage points** for this reduction. This interval describes this finite simulation distribution, not arbitrary real scientific problems.

The blend **does not beat validity decision-value** on the false-claim endpoint, and its 1/96 improvement over validity-query EIG is inconclusive (paired bootstrap interval −2.1 to +4.2 percentage points). There is no demonstrated improvement over the shared-guard baseline for the stronger unseen negative mechanism. These results support testing validity-directed experimentation, not a claim of universally superior acquisition.

On nominal data, CAUSALOS issues 96/96 correct final claims and the shared-guard nominal acquisition issues 95/96. Mean cost to a valid claim is essentially unchanged: **4.508 versus 4.511** budget units. The secondary family-alert endpoint improves under weak confounding: mean capped cost to first alert **5.893 versus 6.784**, with paired reduction 0.892 and bootstrap interval `[0.297,1.475]`. Alert attainment is 95/96 versus 79/96. Under stronger unseen negative responses, both alert in 96/96 episodes, and the blend is slightly slower on average (4.513 versus 4.288). In nominal episodes the policy raises at least one transient model-risk alert in 4/96 cases, compared with 5/96 for the same-guard baseline; an alert can subsequently resolve as more evidence arrives.

The no-guard blend makes false claims in 94/96 weak-confounded episodes. Thus the safeguard is essential, and selection alone does not justify causal statements. Final abstention under the full blend is 90/96 weak-confounded episodes and 95/96 unseen-negative episodes. The six weak-confounded false claims remain a material failure: this finite contamination set does not prevent every unsupported conclusion.

See the [executed comparison chart](../benchmarks/evidence_results/evidence_comparison.svg), [episode data](../benchmarks/evidence_results/episodes.csv), [full development sweep](../benchmarks/evidence_results/development.json), [paired endpoint results](../benchmarks/evidence_results/paired_comparison.csv), [paired bootstrap](../benchmarks/evidence_results/paired_claim_bootstrap.json), and [secondary alert analysis](../benchmarks/evidence_results/secondary_alerts.json). The posterior-risk surrogate and deployed claim rule differ; this study provides empirical performance evidence rather than a decision-optimality theorem.

After execution, a source-only isolation fix removed the unused evaluator scenario name from policy state. All scenario families now produce identical initial policy state at a fixed seed; the simulator receives its scenario externally. A fixed equivalence check reproduces all eight policies across three scenarios at seed 5000 within `1e-12` on numerical endpoints. [Source-equivalence provenance](../benchmarks/evidence_results/source_equivalence.json) records this change, the validation-selected default synchronization, and unchanged raw measurements. The first study's legacy state retains an unused stress-scenario label; its recommendation function never inspects that label for truth, but this is a weaker architectural boundary than the second study's indistinguishable policy states.
