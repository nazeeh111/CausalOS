"""Build the API-free public replay from executed Python analyses.

Run from the repository: python examples/build_demo.py. The public demo only
replays bundled examples. It never analyzes user uploads or altered questions.
Evaluator observation banks are stored separately from policy input state.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causalos import __version__, core, datasets, design, evidence_design, profiling  # noqa: E402
from causalos.report import clean  # noqa: E402


def build() -> None:
    target = ROOT / "docs/demo"
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / "causalos/web"
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    html = (target / "index.html").read_text()
    html = html.replace('href="/styles.css"', 'href="./styles.css"')
    for name in ("app.js", "counterfactual.js", "evidence.js"):
        html = html.replace(f'src="/{name}"', f'src="./{name}"')
    html = html.replace('<script defer src="./app.js">', '<script defer src="./demo-api.js"></script><script defer src="./app.js">')
    for asset in ('styles.css', 'demo-api.js', 'app.js', 'counterfactual.js', 'evidence.js'):
        html = html.replace(f'./{asset}"', f'./{asset}?v={__version__}"')
    banner = '<div class="public-demo-banner" role="note"><strong>PUBLIC REPRODUCIBLE REPLAY</strong><span>Built-in analyses are precomputed. Run locally for new datasets and edited causal questions.</span><a href="https://github.com/nazeeh111/CausalOS#run-the-laboratory">Run locally ↗</a></div>'
    html = html.replace('<body>', '<body>'+banner)
    html = html.replace('LOCAL RESEARCH ENVIRONMENT', 'PUBLIC REPRODUCIBLE REPLAY')
    html = html.replace('Processed by your local CAUSALOS server.', 'New CSV analyses require the local application.')
    html = html.replace('Use non-sensitive research data.', 'This public replay uses bundled examples only.')
    importer = '<label class="button button-primary file-button">Import CSV <span>↑</span><input id="csv-file" type="file" accept=".csv,text/csv" aria-label="Import CSV"></label>'
    local_link = '<a class="button button-primary" href="https://github.com/nazeeh111/CausalOS#run-the-laboratory">Import your data locally ↗</a><input id="csv-file" type="file" hidden disabled>'
    if importer not in html:
        raise ValueError('Canonical CSV control changed; review the public replay adaptation.')
    html = html.replace(importer, local_link)
    html = html.replace('<title>CAUSALOS LAB', '<title>CAUSALOS Public Replay')
    (target / "index.html").write_text(html)
    with (target / "styles.css").open('a') as f:
        f.write('\n.public-demo-banner{position:sticky;top:0;z-index:100;background:#253dc1;color:#fff;padding:10px 24px;display:flex;justify-content:center;align-items:center;gap:18px;font-size:11px;line-height:1.6}.public-demo-banner strong{font:9px var(--mono);letter-spacing:.6px}.public-demo-banner a{color:#e1eaa8;white-space:nowrap}.sidebar{top:var(--demo-banner-height,38px)}.public-demo-banner+.shell{min-height:calc(100vh - 38px)}@media(max-width:1100px){.sidebar{top:var(--demo-banner-height,56px)}.public-demo-banner{padding:10px 20px;gap:12px}.public-demo-banner strong{max-width:140px}}@media(max-width:760px){.public-demo-banner{padding:12px 16px;gap:5px 12px;flex-wrap:wrap;justify-content:flex-start;font-size:10px}.public-demo-banner strong{font-size:9px;max-width:none}.public-demo-banner span{flex-basis:100%;order:2}.public-demo-banner a{margin-left:auto}.sidebar{top:auto}}\n')

    examples = {}
    for item in datasets.list_examples():
        name = item['name']
        data, meta = datasets.example(name, seed=42)
        q = meta['question']
        prof = profiling.profile(data, q.get('treatment'), q.get('covariates', []))
        analysis = core.analyze(data, deepcopy(q))
        examples[name] = clean({'data':data.to_dict(orient='records'), 'metadata':meta, 'question':q, 'profile':prof, 'analysis':analysis})
        print('Analyzed built-in example:', name, analysis['status'], flush=True)
    nominal, evidence, outcomes = {}, {}, {'design':{}, 'evidence':{}}
    for scenario in ('fork','weak','expensive','misspecified'):
        state = design.initial_state(seed=42, scenario=scenario)
        state['recommendations'] = design.recommend(state)
        nominal[scenario] = clean(state)
        env = design.Simulator(42, scenario)
        outcomes['design'][scenario] = {}
        for action in state['candidates']:
            values=[]
            for repeat in range(25):
                mock=deepcopy(state)
                mock['history']=[{'action_id':action['id']}]*repeat
                values.append(env.observe(mock, action['id']))
            outcomes['design'][scenario][action['id']] = values
    for scenario in ('nominal','weak_confounding','unseen_negative'):
        state=evidence_design.initial_state(seed=42,scenario=scenario,contamination_prior=.05)
        state['recommendations']=evidence_design.recommend(state,decision_weight=.5)
        evidence[scenario]=clean(state)
        env=evidence_design.Simulator(42,scenario)
        outcomes['evidence'][scenario]={}
        for action in state['candidates']:
            values=[]
            for repeat in range(25):
                mock=deepcopy(state)
                mock['history']=[{'action_id':action['id']}]*repeat
                values.append(env.observe(mock,action['id']))
            outcomes['evidence'][scenario][action['id']]=values
    benchmark_path = ROOT / 'causalos/resources/benchmark_summary.json'
    if not benchmark_path.exists():
        benchmark_path=ROOT/'benchmarks/results/summary.json'
    benchmarks=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {'status':'pending'}
    nodes,weights=design._quadrature(21)
    build_sources = sorted(set([
        *ROOT.joinpath('causalos').glob('*.py'),
        *(p for p in source.iterdir() if p.is_file()),
        *(p for p in ROOT.joinpath('causalos/resources/data').rglob('*') if p.is_file()),
        Path(__file__), target/'demo-api.js', target/'check_parity.cjs',
        *([benchmark_path] if benchmark_path.exists() else []),
    ]))
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in build_sources}
    environment = {'python':platform.python_version(),
                   'packages':{name:importlib.metadata.version(name) for name in
                               ('numpy','scipy','pandas','scikit-learn','statsmodels','duckdb')}}
    payload={'version':__version__,'seed':42,'examples_list':datasets.list_examples(),'examples':examples,
             'design':nominal,'evidence':evidence,'quadrature':{'nodes':nodes.tolist(),'weights':weights.tolist()},
             'design_parameters':design.DEFAULT_PARAMETERS,'benchmarks':benchmarks,
             'reproducibility':{'build_source_sha256':source_hashes,'environment':environment,'note':'Python-executed built-in fixtures and deterministic evaluator observation banks. Policy never receives evaluator ground truth.',
                                'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'causalos/design.py',ROOT/'causalos/evidence_design.py',ROOT/'causalos/core.py']}}}
    # Cross-language numerical contract checks, not policy inputs.
    parity=[]
    for evidence_mode, scenarios in ((False, ('fork','weak','expensive','misspecified')), (True, ('nominal','weak_confounding','unseen_negative'))):
        for scenario in scenarios:
            module=evidence_design if evidence_mode else design
            state=deepcopy((evidence if evidence_mode else nominal)[scenario])
            env=module.Simulator(42,scenario)
            actions=['a-low','y-probe','b-probe','a-high']
            for aid in actions:
                state=module.update(state,aid,env.observe(state,aid))
            recommendation=module.recommend(state,**({'decision_weight':.5} if evidence_mode else {}))
            parity.append({'kind':'evidence' if evidence_mode else 'design','scenario':scenario,'actions':actions,
                           'posterior':state['posterior'],'effect_mean':state['effect_mean'],'claim':state['claim'],
                           'model_conflict':state['model_conflict'],'selected':recommendation['selected'],
                           'candidate_utilities':[{k:a[k] for k in ('id','expected_information_gain','expected_query_information_gain','expected_decision_gain')} for a in state['candidates']]})
    (target/'parity.json').write_text(json.dumps(clean(parity),separators=(',',':'),allow_nan=False)+'\n')
    (target/'fixtures.json').write_text(json.dumps(clean(payload),separators=(',',':'),allow_nan=False)+'\n')
    (target/'observations.json').write_text(json.dumps(clean(outcomes),separators=(',',':'),allow_nan=False)+'\n')
    (ROOT/'docs/index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=./demo/"><title>CAUSALOS LAB</title><p><a href="./demo/">Open the CAUSALOS public reproducible replay.</a></p></html>\n')
    (ROOT/'docs/.nojekyll').write_text('')
    manifest = {'format':'causalos-public-replay-build-v1', 'source_sha256':source_hashes,
                'environment':environment,
                'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(target.iterdir())
                                   if p.is_file() and p.name != 'build-manifest.json'}}
    (target/'build-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Public replay built:',target,flush=True)


if __name__=='__main__':
    build()
