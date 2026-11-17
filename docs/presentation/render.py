"""Source-bound finite-model illustration; no simulation, fitting or benchmark."""
from pathlib import Path
import json,hashlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
P=Path(__file__).resolve().parent
I=json.loads((P/'initial-state.json').read_text());F=json.loads((P/'completed-replay.json').read_text());V=json.loads((P/'provenance.json').read_text())
for name,h in V['source_sha256'].items():assert hashlib.sha256((P.parents[1]/name).read_bytes()).hexdigest()==h
A={a['id']:a for a in I['candidates']};assert A['b-probe']['expected_query_information_gain']==0 and F['claim']=='zero effect'
ink='#222b32';blue='#2846c9';muted='#566159';paper='#f5f4ee';green='#465e2e'
def render(phone=False):
 w,h=(440,1370) if phone else (1000,825)
 with plt.rc_context({'font.family':['DejaVu Sans','Arial','sans-serif'],'svg.fonttype':'none','svg.hashsalt':'causalos-question-replay'}):
  fig=plt.figure(figsize=(w/100,h/100),dpi=100,facecolor=paper);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
  def txt(x,y,t,size=18,c=ink,weight='normal',ha='left'):return ax.text(x,y,t,fontsize=size*.72,color=c,weight=weight,ha=ha,va='center',zorder=5)
  m=24 if phone else 40
  txt(m,32,'CAUSALOS / FINITE-MODEL EXPERIMENT',12,muted,'bold')
  if phone:txt(m,79,'Spend on the',31,weight='bold');txt(m,117,'causal question.',31,weight='bold');txt(m,159,'Built-in simulation · seed 42 · four SCMs',16)
  else:txt(m,86,'Spend on the causal question.',42,weight='bold');txt(m,131,'Built-in simulation · seed 42 · four supplied linear Gaussian models',20)
  gy=210 if phone else 196
  txt(m,gy,'SAME OBSERVATIONS, OPPOSITE DIRECTIONS',13,blue,'bold')
  txt(m,gy+31,f'Both imply correlation rho = {I["rho"]:.3f}.',17)
  for y,forward,label in [(gy+83,True,'Prior 50%'),(gy+146,False,'Prior 50%')]:
   x1,x2=m+36,m+165
   ax.annotate('',xy=(x2-19,y) if forward else (x1+19,y),xytext=(x1+19,y) if forward else (x2-19,y),arrowprops={'arrowstyle':'->','color':blue,'lw':1.8})
   for x,node in [(x1,'A'),(x2,'Y')]:ax.add_patch(Circle((x,y),19,facecolor='#fffef9',edgecolor=blue,lw=1.5));txt(x,y,node,18,ha='center')
   txt(m+208,y,label,18)
  txt(m,gy+190,'Four models = two A–Y directions × two B–Z directions.',12,muted)
  ty=468 if phone else 196;tx=24 if phone else 530
  txt(tx,ty,'WHICH EXPERIMENT EARNS THE BUDGET?',13,blue,'bold')
  txt(tx,ty+29,'Two actions from the five-action menu.',15,muted)
  # Each pair of bars shares a zero baseline and the same [0,0.70] nat scale.
  for i,(aid,label) in enumerate([('b-probe','Probe B, read Z'),('a-low','Set A = +1, read Y  /  selected')]):
   a=A[aid];yy=ty+75+i*115;txt(tx,yy,label,18,weight='bold');txt(tx,yy+25,f'n = {a["n"]} · cost {a["cost"]:g} simulated units',14,muted)
   bw=155 if phone else 190;bx=tx+180;by=yy+52
   for j,(key,lab,color) in enumerate([('expected_information_gain','Graph',green),('expected_query_information_gain','Query',blue)]):
    val=a[key];txt(tx,by+j*23,f'{lab}: {val:.3f} nats',14,color)
    ax.plot([bx,bx+bw],[by+j*23]*2,color='#daddd2',lw=5,solid_capstyle='butt')
    ax.plot([bx,bx+bw*val/.70],[by+j*23]*2,color=color,lw=5,solid_capstyle='butt')[0].set_gid(f'bar-{aid}-{key}')
  txt(tx,ty+297,'Bar scale: 0 to 0.70 nats, shared zero baseline.',12,muted)
  py=867 if phone else 523;ph=235 if phone else 186;px=62 if phone else 82;pw=340 if phone else 855
  txt(m,py-53,'FOLLOW THE ACTUAL POSTERIOR',14,blue,'bold')
  txt(m,py-25,'P(A → Y | simulated evidence, supplied family)',16)
  plot=fig.add_axes([px/w,(h-py-ph)/h,pw/w,ph/h],facecolor='#fffef9')
  costs=[0]+[r['cumulative_cost'] for r in F['history']];prob=[I['query_probability']]+[sum(r['posterior'][:2]) for r in F['history']]
  plot.step(costs,prob,where='post',color=blue,lw=1.7)[0].set_gid('query-posterior-step');plot.scatter(costs,prob,color=blue,s=23,zorder=4).set_gid('query-posterior-points')
  plot.set(xlim=(-.25,12.3),ylim=(-.025,1.025),xticks=[0,2,4,6,8,10,12],yticks=[0,.5,1]);plot.tick_params(labelsize=10 if phone else 11,colors=muted)
  plot.set_xlabel('Spent budget / simulated cost units',fontsize=11,color=muted)
  plot.spines[['top','right']].set_visible(False);plot.spines[['left','bottom']].set_color('#a9b3a7');plot.grid(axis='y',color='#e4e7de',lw=.6)
  plot.axhline(.05,color=green,lw=.8,ls=':');plot.text(6.1,.56,'Round 3: P = 0.0325\nconditional zero-effect claim',fontsize=10,color=green)
  plot.annotate('',xy=(6,prob[3]),xytext=(6.6,.48),arrowprops={'arrowstyle':'->','color':green,'lw':.8})
  foot=1171 if phone else 773
  if phone:
   txt(m,foot,'Final claim: zero effect, within this family.',17,weight='bold');txt(m,foot+29,'Spent 12 / 12 · B–Z direction still unresolved.',15)
   txt(m,foot+77,'Known likelihoods · perfect interventions',15,muted);txt(m,foot+103,'Equal model priors · no omitted confounding',15,muted);txt(m,foot+142,'A simulator decision, not real-world causal proof.',14,muted)
  else:
   txt(m,foot,'Final: conditional zero-effect claim · spent 12 / 12 · unrelated B–Z direction remains unresolved.',17,weight='bold')
   txt(m,foot+28,'Equal model priors · known Gaussian likelihoods · perfect interventions · assumed family completeness · simulated evidence',14,muted)
  name='question-first-mobile' if phone else 'question-first'
  fig.savefig(P/(name+'.png'),dpi=100,facecolor=paper);fig.savefig(P/(name+'.svg'),facecolor=paper,metadata={'Date':None});plt.close(fig)
  svg=(P/(name+'.svg')).read_text();svg=re.sub(r'font-family:[^;\"]+',"font-family: 'DejaVu Sans', Arial, sans-serif",svg)
  svg=svg.replace('<svg ','<svg role="img" aria-labelledby="co-title co-desc" ',1);i=svg.index('>',svg.index('<svg '))+1
  svg=svg[:i]+'<title id="co-title">Spend on the causal question.</title><desc id="co-desc">Actual seed 42 finite-model simulation. B probe has graph information 0.693 nats but query information 0. The selected A probe has query information 0.289 nats. Six simulated experiments spend 12 units and produce a zero-effect claim conditional on the four models, equal prior, known likelihoods and perfect interventions. No real-world causal proof.</desc>'+svg[i:]
  (P/(name+'.svg')).write_text(svg)
render();render(True)
