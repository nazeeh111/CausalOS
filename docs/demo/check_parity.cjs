/* Reproduce cross-language public-replay checks with Node 20+; no dependencies. */
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),crypto=require('node:crypto');
const directory=__dirname;
const fixtures=JSON.parse(fs.readFileSync(path.join(directory,'fixtures.json')));
const window={fetch:async url=>new Response(fs.readFileSync(path.join(directory,new URL(url).pathname.split('/').at(-1))),{headers:{'Content-Type':'application/json'}})};
vm.runInNewContext(fs.readFileSync(path.join(directory,'demo-api.js'),'utf8'),{window,Response,URL,location:{href:'https://example.test/CausalOS/demo/'}});
(async()=>{
  const digest=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const manifest=JSON.parse(fs.readFileSync(path.join(directory,'build-manifest.json')));
  for(const [name,expected] of Object.entries(manifest.source_sha256)){
    if(digest(path.resolve(directory,'../..',name))!==expected)throw Error('Stale replay source: '+name+'; rebuild with python examples/build_demo.py.');
  }
  for(const [name,expected] of Object.entries(manifest.artifact_sha256)){
    if(digest(path.join(directory,name))!==expected)throw Error('Replay artifact differs from verified build: '+name+'; rebuild.');
  }
  for(const name of fs.readdirSync(path.resolve(directory,'../../causalos/web'))){
    if(['index.html','styles.css'].includes(name))continue;
    if(digest(path.resolve(directory,'../../causalos/web',name))!==digest(path.join(directory,name)))throw Error('Copied canonical asset differs: '+name);
  }
  const html=fs.readFileSync(path.join(directory,'index.html'),'utf8');
  if(!html.includes('CausalOS#run-the-laboratory')||html.includes('#quick-start'))throw Error('Run locally link does not target the installation heading.');
  if(!fs.readFileSync(path.resolve(directory,'../../README.md'),'utf8').includes('## Run the laboratory'))throw Error('Run locally target heading was renamed.');
  const parity=JSON.parse(fs.readFileSync(path.join(directory,'parity.json')));
  let maximumError=0,ties=0;
  const post=async(endpoint,body)=>{const response=await window.fetch('/api/'+endpoint,{method:'POST',body:JSON.stringify(body)});return {response,value:await response.json()};};
  for(const expected of parity){
    let state=(await post(expected.kind+'/reset',{seed:42,scenario:expected.scenario})).value;
    for(const action_id of expected.actions)state=(await post(expected.kind+'/step',{state,action_id,seed:42,scenario:expected.scenario})).value;
    if(state.detail)throw Error(state.detail);
    for(let i=0;i<state.posterior.length;i++)maximumError=Math.max(maximumError,Math.abs(state.posterior[i]-expected.posterior[i]));
    maximumError=Math.max(maximumError,Math.abs(state.effect_mean-expected.effect_mean));
    if(state.claim!==expected.claim||state.model_conflict!==expected.model_conflict)throw Error('Claim parity failed '+expected.kind+' '+expected.scenario);
    for(const a of state.candidates){const e=expected.candidate_utilities.find(x=>x.id===a.id);for(const k of ['expected_information_gain','expected_query_information_gain','expected_decision_gain'])maximumError=Math.max(maximumError,Math.abs(a[k]-e[k]));}
    if(state.recommendations.selected!==expected.selected){
      const actual=state.recommendations.ranked[0]?.score??0;
      const expectedScore=state.recommendations.ranked.find(a=>a.id===expected.selected)?.score??0;
      if(Math.abs(actual-expectedScore)>1e-12)throw Error('Acquisition ranking differs beyond floating-point tolerance.');
      ties++;
    }
  }
  if(maximumError>1e-12)throw Error('Numerical error exceeds tolerance: '+maximumError);
  for(const [name,e]of Object.entries(fixtures.examples)){
    if(!(await post('analyze',{data:e.data,question:e.question})).response.ok)throw Error('Built-in rejected '+name);
    const q=structuredClone(e.question);q.assumptions.exchangeability=false;
    if((await post('analyze',{data:e.data,question:q})).response.status!==422)throw Error('Changed assumptions accepted '+name);
    const graph=structuredClone(e.question);graph.graph.edges=[];graph.graph.unresolved=[['T','Y']];
    if((await post('analyze',{data:e.data,question:graph})).response.status!==422)throw Error('Changed graph accepted '+name);
    const changedData=structuredClone(e.data);changedData[0][Object.keys(changedData[0])[0]]=99999;
    if((await post('analyze',{data:changedData,question:e.question})).response.status!==422)throw Error('Changed data accepted '+name);
  }
  const exported=await window.fetch('/api/report',{body:JSON.stringify({question:{note:'<script>alert(1)</script>'}})});
  const report=await exported.text();
  if(report.includes('<script>')||!report.includes('&lt;script&gt;'))throw Error('Report escape failed.');
  console.log(JSON.stringify({status:'pass',build_sources:Object.keys(manifest.source_sha256).length,build_artifacts:Object.keys(manifest.artifact_sha256).length,canonical_web_assets:'match',run_locally_link:'pass',scenario_comparisons:parity.length,maximum_absolute_numerical_error:maximumError,near_zero_acquisition_ties:ties,builtin_analyses:Object.keys(fixtures.examples).length,altered_analysis_blocks:Object.keys(fixtures.examples).length*3,report_html_escape:'pass'},null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
