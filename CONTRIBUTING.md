# Contributing

Use Python 3.11 or newer. Create an isolated environment, then `pip install -e '.[dev,discovery]'` and run `ruff check causalos tests` and `pytest -q`.

Changes to causal claims need evidence. State the estimand, identification assumptions, data-generating process and applicable uncertainty interpretation. A learned edge is not experimental verification. Include invalid and nonidentifiable examples when changing identification or estimation.

Keep simulator truth in the evaluator. Never expose truth labels, test-seed performance, potential outcomes or post-treatment features to the policy, hyperparameter search or estimators. Do not select configurations on test results. Record negative results and denominator changes. If a benchmark changes, regenerate its tables, figures, metadata and research report together.

UI changes should remain keyboard accessible, preserve reduced-motion behavior and label simulation clearly. Include a current screenshot for a substantial visual change. Reports must retain assumptions and limitations, including blocked analyses.

Open an issue to discuss large methodological changes. Contributions are MIT licensed; retain attribution and obey dataset-specific terms. Do not commit private datasets, credentials or participant identifiers. This project runs locally and needs no model API key.
