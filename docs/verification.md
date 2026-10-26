# Verification record

This is a research-preview release. Verification checks bounded mathematical and software properties; it does not establish general causal validity.

## Independent scientific review

A separate reviewer inspected source and raw results, rather than relying on builder summaries. Checks included 640 random DAG/query triples against NetworkX d-separation, estimator equations and interval assumptions, simulator/policy input separation, disjoint development/test seeds, and first-study baseline equality. No remaining demonstrated causal-identification or estimator-arithmetic defect was found after fixes.

The reviewer independently checked all 2,304 second-study rows and replayed 18 seed/scenario/policy combinations, matching stored outcomes within 1e-13. A separate 100,000 resample paired bootstrap gave a weak-confounding false-claim reduction interval of 7.29–23.96 percentage points, while also confirming ties and no unseen-negative improvement. The reviewed evidence-policy source hash is `1c3ad2802fe037fb8466276be56e3a2b6b7be8319b5ec7b91272812de7bcb31b`; raw episodes hash is `9071f82ec6203d6c8edb3b880f1c7b779681be05db83f1038277f7de2c61458d`. Subsequent API/UI integration does not alter that policy.

Review found and prompted fixes for malformed assumption objects causing HTTP500, dropped study metadata in reports, inappropriate pooling of correlated scenario seeds in plot error bars, and misleading interpretation of the first response-table stressor. The source and research report preserve these distinctions.

## Executed software checks

The local suite covers identification, invalid cases, numerical recovery, SQL/provenance, audited transformations, actual packaged data, Bayesian updates and quadrature, state isolation, counterfactual consistency, API workflows and report escaping. Tests and CI are included in the source. The CLI has been exercised from dataset export through profiling, analysis, report generation and sequential replay.

The current browser interface has been inspected and used to run the simulation, withdraw an assumption, receive a blocked result, and render valid analysis outputs. Screenshots come from the functioning application. The launch film has a current-source HyperFrames check, decoded-frame inspection and a full FFmpeg decode; it uses original locally synthesized audio. Rendering-audit limits are preserved in its verification files.

## Material limits

The estimator study has 12 easy independent linear-SCM replications per eligible case and one IHDP outcome replication with dependent row splits. This is numerical/statistical validation rather than broad calibration. The design comparisons are explicit finite-model acquisition criteria, including adaptations of published information-gain principles; they do not reproduce large neural published systems. No real experiment was selected and performed, and no scientific result is claimed for an arbitrary real-world intervention.

Release installation, CI and public-demo verification are recorded with the final release evidence.

The final local source suite passes 61 tests. A clean wheel installation also passes the same suite; the source archive installs independently and exports the bundled NSW case and experiment replay. The released package's 26 application/resource files were compared byte-for-byte against source. Live checks additionally verified that imported bidirected confounding is preserved and blocks estimation, exports include an edited draft question without stale estimates, and counterfactual results clear when inputs or model beliefs change. The v0.1.0 bounded four-view text-contrast check measured a minimum 4.66:1; this is not a claim of complete accessibility certification.

## v0.1.1 independent public review and interface refinement

A fresh reviewer checked the public release and live application, then rechecked the refined desktop and mobile workflows. The findings, fixes, browser evidence and limits are recorded in [the v0.1.1 review](review-v0.1.1.md). Working text, acquisition metrics, keyboard access, provenance labels, model identities and the research presentation were improved without changing statistical policies or measured results. The original launch film records the v0.1.0 interface and remains labeled accordingly.
