/* Public reproducible replay adapter. Canonical Python implementation: causalos/.
   No statistical fitting happens on this static page. Bayesian finite-model
   updates and quadrature acquisition functions execute here; the independent
   evaluator supplies precomputed observations by scenario, action, and repeat.
   No true model index or ground-truth treatment effect is exposed to the policy. */
'use strict';
(() => {
  const originalFetch = window.fetch.bind(window);
  const clone = value => JSON.parse(JSON.stringify(value));
  const load = async name => {
    const response = await originalFetch(new URL(name, location.href));
    if (!response.ok) throw Error('Public replay fixture unavailable. Reload this page.');
    return response.json();
  };
  const ready = Promise.all([load('./fixtures.json'), load('./observations.json')]);
  const json = (value, status=200) => new Response(JSON.stringify(value), {status,headers:{'Content-Type':'application/json'}});
  const localOnly = 'Public replay supports the exact bundled examples only. Run CAUSALOS locally (see Run locally above) to analyze new datasets, changed graphs, or edited identifying assumptions. No estimate was generated for the modified question.';
  const canonical = value => JSON.stringify(normalize(value));
  function normalize(value) {if (Array.isArray(value)) return value.map(normalize);if(value&&typeof value==='object')return Object.fromEntries(Object.keys(value).sort().map(k=>[k,normalize(value[k])]));return value;}
  const sum = values => values.reduce((a,b)=>a+b,0);
  const entropy = p => -sum(p.filter(v=>v>0).map(v=>v*Math.log(v)));
  const classes = (p,evidence) => evidence?[sum(p.slice(0,2)),sum(p.slice(2,4)),sum(p.slice(4))]:[sum(p.slice(0,2)),sum(p.slice(2))];
  const risk = (p,evidence) => evidence?Math.min(.10,1-Math.max(...classes(p,true))):Math.min(...classes(p,false));
  function posterior(p, means, scales, observation) {
    const logits=p.map((v,i)=>Math.log(Math.max(v,1e-300))-Math.log(scales[i])-.5*((observation-means[i])/scales[i])**2);
    const max=Math.max(...logits), terms=logits.map(v=>Math.exp(v-max)),total=sum(terms);
    return terms.map(v=>v/total);
  }
  function utility(p, action, quadrature, evidence) {
    const mus=action.mean_predictions, sds=action.standard_errors;
    let eh=0,eq=0,er=0;
    for(let model=0;model<p.length;model++){
      for(let j=0;j<quadrature.nodes.length;j++){
        const observation=mus[model]+sds[model]*quadrature.nodes[j];
        const post=posterior(p,mus,sds,observation),weight=p[model]*quadrature.weights[j];
        eh+=weight*entropy(post);eq+=weight*entropy(classes(post,evidence));er+=weight*risk(post,evidence);
      }
    }
    return {expected_information_gain:Math.max(0,entropy(p)-eh),expected_query_information_gain:Math.max(0,entropy(classes(p,evidence))-eq),expected_decision_gain:Math.max(0,risk(p,evidence)-er),predictive_variance:sum(p.map((v,i)=>v*(sds[i]**2+mus[i]**2)))-sum(p.map((v,i)=>v*mus[i]))**2};
  }
  function recommend(state, fixtures, evidence) {
    const params=fixtures.design_parameters;
    const ranked=state.candidates.filter(a=>a.cost<=state.budget_remaining+1e-9).map(a=>{
      const action=clone(a);
      action.score=evidence?(.5*action.expected_decision_gain/.10+.5*action.expected_query_information_gain/Math.log(3))/action.cost:(params.decision_weight*2*action.expected_decision_gain+params.query_weight*action.expected_query_information_gain/Math.log(2)+params.graph_weight*action.expected_information_gain/Math.log(4))/action.cost**params.cost_power;
      action.affordable=true;return action;
    }).sort((a,b)=>b.score-a.score||a.cost-b.cost||a.id.localeCompare(b.id));
    return {selected:ranked[0]?.id||null,ranked,policy:evidence?'evidence':'causalos',reason:evidence?'Select evidence that distinguishes positive effect, zero effect and failure of the nominal model family.':'Maximize expected reduction in decision regret and query uncertainty per experiment cost.',simulated:true,...(!evidence?{parameters:params}:{})};
  }
  function decorate(state, fixtures, evidence) {
    const p=state.posterior;
    state.entropy=entropy(p);
    state.hypotheses.forEach((h,i)=>h.posterior=p[i]);
    if(evidence){
      const nominalWeight=sum(p.slice(0,4)),nominal=p.slice(0,4).map(v=>v/Math.max(nominalWeight,1e-300));
      state.nominal_posterior=nominal;
      state.query_probability=sum(p.slice(0,2));state.nominal_query_probability=sum(nominal.slice(0,2));state.shadow_probability=sum(p.slice(4));
      state.effect_mean=sum(p.map((v,i)=>v*state.hypotheses[i].effect));
      const ordered=state.hypotheses.map((h,i)=>({effect:h.effect,p:p[i]})).sort((a,b)=>a.effect-b.effect);
      state.effect_interval=[.025,.975].map(q=>{let c=0;for(const h of ordered){c+=h.p;if(c>=q)return h.effect;}return ordered.at(-1).effect;});
      state.confidence=Math.max(...classes(p,true));state.decision=state.effect_mean>state.rho/2?'intervene':'do not intervene';
      state.model_conflict=state.shadow_probability>.20;
      const nq=state.nominal_query_probability;
      state.claim=Math.max(nq,1-nq)<.95||state.model_conflict?'abstain':nq>.5?'positive effect':'zero effect';
    }else{
      const forward=sum(p.slice(0,2));state.query_probability=forward;state.effect_mean=forward*state.rho;
      state.effect_interval=[1-forward>=.025?0:state.rho,forward>=.025?state.rho:0];
      state.confidence=Math.max(forward,1-forward);state.decision=forward>.5?'intervene':'do not intervene';
      state.claim=state.model_conflict||state.confidence<.95?'abstain':forward>.5?'positive effect':'zero effect';
    }
    state.candidates=state.candidates.map(a=>({...a,...utility(p,a,fixtures.quadrature,evidence)}));
    state.recommendations=recommend(state,fixtures,evidence);
    return state;
  }
  function step(input,actionId,fixtures,banks,evidence,scenario) {
    const state=clone(input);
    if(!Array.isArray(state.posterior)||state.posterior.length!==(evidence?8:4)||state.posterior.some(v=>!Number.isFinite(v)||v<0)||Math.abs(sum(state.posterior)-1)>1e-8)throw Error('Invalid posterior. Reset the public replay.');
    const base=evidence?fixtures.evidence[scenario]:fixtures.design[state.scenario];
    if(!base||Math.abs(base.rho-state.rho)>1e-12)throw Error('This public replay supports only its recorded seed and structural family. Reset the experiment.');
    // Reconstruct predictive parameters from canonical public state, ignoring edits.
    const action=clone(base.candidates.find(a=>a.id===actionId));
    if(!action||action.cost>state.budget_remaining+1e-9)throw Error('Unknown or unaffordable experiment.');
    const repeat=state.history.filter(h=>h.action_id===actionId).length;
    // Evaluator access occurs after action selection, outside utility/recommend.
    const observation=banks[evidence?'evidence':'design'][evidence?scenario:state.scenario]?.[actionId]?.[repeat];
    if(!Number.isFinite(observation))throw Error('Recorded evaluator bank exhausted. Reset the public replay.');
    const p=state.posterior;
    let conflict=false;
    if(!evidence){conflict=p.every((v,i)=>v<=.001||Math.abs((observation-action.mean_predictions[i])/action.standard_errors[i])>3.5);state.model_conflict=state.model_conflict||conflict;if(conflict&&!state.warnings.includes('Predictive conflict: observation is inconsistent with every currently plausible candidate; causal claims withheld.'))state.warnings.push('Predictive conflict: observation is inconsistent with every currently plausible candidate; causal claims withheld.');}
    state.posterior=posterior(p,action.mean_predictions,action.standard_errors,observation);
    state.spent+=action.cost;state.budget_remaining=Math.max(0,state.budget_total-state.spent);
    state.candidates=clone(base.candidates);
    decorate(state,fixtures,evidence);
    state.history.push({round:state.history.length+1,action_id:actionId,label:action.label,observation,readout:action.readout,posterior:state.posterior.slice(),entropy:state.entropy,effect_mean:state.effect_mean,confidence:state.confidence,cost:action.cost,cumulative_cost:state.spent,...(evidence?{shadow_probability:state.shadow_probability}:{model_conflict:conflict})});
    return state;
  }
  function counterfactual(body){
    const {state,observed,target,value}=body;
    if(state?.posterior?.length!==4||!['A','Y','B','Z'].includes(target)||!Object.values(observed).every(Number.isFinite)||!Number.isFinite(value))throw Error('Use the supplied four-model structural scenario and finite numerical values.');
    const models=state.hypotheses.map((model,i)=>{
      const result={...observed,[target]:value};
      for(const[source,child]of model.edges){if(source===target)result[child]+=((source==='A'&&child==='Y'||source==='Y'&&child==='A')?state.rho:.8)*(value-observed[source]);}
      return {hypothesis:model.id,label:model.label,probability:state.posterior[i],predicted:result,change_in_Y:result.Y-observed.Y};
    });
    return {evidence_status:'structural-model-conditional counterfactual',observed,intervention:{target,value},models,posterior_mean_Y:sum(models.map(m=>m.probability*m.predicted.Y)),limitations:['Hypothetical unit within the stated linear Gaussian SCMs. No real participant outcome is predicted.','Each model infers exogenous noise from all four observed variables, then holds that noise fixed.','Competing graphs imply different counterfactuals. Their posterior average does not identify an individual causal outcome.','This comparison is unavailable for uploaded datasets without a fully specified structural model.']};
  }
  const htmlEscape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  window.fetch=async(input,options={})=>{
    const url=new URL(typeof input==='string'?input:input.url,location.href);
    if(!url.pathname.startsWith('/api/'))return originalFetch(input,options);
    try{
      const [fixtures,banks]=await ready;
      const path=url.pathname,body=options.body?JSON.parse(options.body):{};
      if(body.seed!==undefined&&body.seed!==42)throw Error('The public replay uses recorded seed 42. Run locally for additional seeds.');
      if(path==='/api/health')return json({status:'ok',version:fixtures.version,mode:'public-reproducible-replay',external_model_api:false});
      if(path==='/api/examples')return json({examples:fixtures.examples_list});
      if(path.startsWith('/api/examples/')){const key=decodeURIComponent(path.split('/').at(-1)),example=fixtures.examples[key];if(!example)throw Error('Unknown bundled example.');const response=clone(example);delete response.analysis;return json(response);}
      if(path==='/api/profile'||path==='/api/discover')return json({detail:localOnly},422);
      if(path==='/api/analyze'){
        const entry=Object.values(fixtures.examples).find(e=>canonical(e.data)===canonical(body.data)&&canonical(e.question)===canonical(body.question));
        if(!entry)return json({detail:localOnly},422);
        return json(entry.analysis);
      }
      if(path==='/api/design')return json(fixtures.design.fork);
      if(path==='/api/design/reset'){
        if(body.budget!==undefined&&body.budget!==12)throw Error('Public replay has a recorded budget of 12 units. Run locally for custom budgets.');
        const state=fixtures.design[body.scenario||'fork'];if(!state)throw Error('Unknown recorded scenario.');return json(state);
      }
      if(path==='/api/design/step')return json(step(body.state,body.action_id,fixtures,banks,false));
      if(path==='/api/evidence')return json(fixtures.evidence.nominal);
      if(path==='/api/evidence/reset'){
        const state=fixtures.evidence[body.scenario||'weak_confounding'];if(!state)throw Error('Unknown evaluator scenario.');return json(state);
      }
      if(path==='/api/evidence/step')return json(step(body.state,body.action_id,fixtures,banks,true,body.scenario||'weak_confounding'));
      if(path==='/api/counterfactual')return json(counterfactual(body));
      if(path==='/api/benchmarks')return json(fixtures.benchmarks);
      if(path==='/api/report'){
        const report={format:'causalos-public-replay-evidence-v1',created_utc:new Date().toISOString(),software:{causalos:fixtures.version,mode:'public reproducible replay'},public_replay_limits:localOnly,...body,reproducibility:{...body.reproducibility,public_fixture_source:fixtures.reproducibility,seed:42}};
        if(url.searchParams.get('format')==='json')return json(report);
        return new Response('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CAUSALOS public replay evidence</title><style>body{font:15px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:24px;color:#243132;background:#f5f4ee}h1{font:48px Georgia}pre{background:white;border:1px solid #dce0d2;padding:20px;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.6 monospace}</style><h1>CAUSALOS</h1><p>Public reproducible replay. Built-in analyses are precomputed. Simulated experimental evidence is conditional on specified structural models.</p><pre>'+htmlEscape(JSON.stringify(report,null,2))+'</pre></html>',{headers:{'Content-Type':'text/html'}});
      }
      return json({detail:'This operation requires the local Python application. See Run locally.'},422);
    }catch(error){return json({detail:error.message||String(error)},422);}
  };
})();
