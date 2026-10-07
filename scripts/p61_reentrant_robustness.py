#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p61_reentrant_robustness.py

Purpose:
    Briefly describe the specific computational analysis performed by this
    script and its role in the overall study.

Study context:
    This script is part of the computational workflow used to investigate
    coupling-induced multistability, synchronization loss, re-entrant P/M
    regime selection, and temporal switching in mutually coupled third-order
    phase-locked loops.

Physical regime convention:
    P regime: (D1, D2) = (+, -)
    M regime: (D1, D2) = (-, +)

    P and M denote finite-time detector-phase drift orientations. They should
    not be interpreted automatically as distinct asymptotic attractors.

Inputs:
    - List the required input files, parameters, or outputs from earlier
      scripts.
    - Use "None" when the script is self-contained.

Outputs:
    - List the principal CSV, NPZ, JSON, PNG, PDF, or other generated files.
    - Identify output directories when appropriate.

Manuscript relevance:
    - Section: [Methods/Results section]
    - Figure/Table: [Figure X, Table X, Supplementary Figure SX, or N/A]
    - Principal result: [one-sentence description]

Reproducibility:
    This program is part of the reproducibility repository accompanying the
    manuscript. Numerical classifications and dynamical interpretations should
    be understood within the finite-time framework described in the paper.

Author:
    Hamid Ismail, Ph.D.

Repository:
    Coupled Third-Order PLL Dynamics

License:
    See the LICENSE file in the repository root.
===============================================================================
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p25_boundary_mapping import integrate

DESIGN=[
 {'transition_id':'e30_r0_b04_t04','ic_id':'e30_r0_b04','edge':30,'x0':[-30.7414393037252,-0.6184411906215073,-0.4627857546586868,2306.5828194495507,0.2087647253360028,-0.703451896101081],'zeta_left':0.89899965,'zeta_right':0.8989996625000001,'left':'P','right':'M'},
 {'transition_id':'e30_r1_b02_t01','ic_id':'e30_r1_b02','edge':30,'x0':[-30.74162510136024,-0.6186889004633828,-0.4622845326162891,2306.5829696949736,0.2091833535274833,-0.7036267328733311],'zeta_left':0.8989998078125,'zeta_right':0.8989998203124999,'left':'P','right':'M'},
 {'transition_id':'e34_r0_b01_t04','ic_id':'e34_r0_b01','edge':34,'x0':[-30.740063019991457,-0.6179420331328805,-0.4640668789878922,2306.583374708984,0.2083947185500864,-0.7039363405934709],'zeta_left':0.89899965,'zeta_right':0.8989996625000001,'left':'P','right':'M'},
 {'transition_id':'e34_r1_b03_t04','ic_id':'e34_r1_b03','edge':34,'x0':[-30.73982331624468,-0.6179415656138276,-0.4641320154590908,2306.5835806248465,0.2085072666269868,-0.704138661019736],'zeta_left':0.8989998453125,'zeta_right':0.8989998578125,'left':'P','right':'M'},
]

def orient(d1,d2,zt=.05):
    if d1 < -zt and d2 > zt:return 'M'
    if d1 > zt and d2 < -zt:return 'P'
    return 'ambiguous'
def classify(t,Y,z):
    p1=Y[:,3]-Y[:,0];p2=z*Y[:,0]-Y[:,3];dur=float(t[-1]-t[0])
    d1=float((p1[-1]-p1[0])/dur);d2=float((p2[-1]-p2[0])/dur)
    return orient(d1,d2),d1,d2

def stochastic_integrate(x,p,duration,dt,sigma,seed,max_abs):
    """Euler-Maruyama stress test; independent additive noise enters x3 and x6 derivatives."""
    n=int(round(duration/dt)); rng=np.random.default_rng(seed); y=np.asarray(x,float).copy()
    # retain every 5th step to limit memory; endpoint drift is insensitive to dense storage
    stride=max(1,int(round(.5/dt))); ts=[0.0]; ys=[y.copy()]; sq=math.sqrt(dt)
    for k in range(1,n+1):
        f=rotating_rhs(0.0,y,p); noise=np.zeros(6); noise[[2,5]]=sigma*sq*rng.standard_normal(2)
        y=y+f*dt+noise
        if not np.all(np.isfinite(y)) or np.max(np.abs(y))>max_abs: raise RuntimeError('stochastic trajectory escaped/nonfinite')
        if k%stride==0 or k==n: ts.append(k*dt);ys.append(y.copy())
    return np.asarray(ts),np.asarray(ys)

def savej(p,o):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rows,fields):
    with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',default='results/author_reentrant_robustness')
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=3200.);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--ic-amplitudes',default='2e-6,2e-5');ap.add_argument('--ic-directions',type=int,default=3)
    ap.add_argument('--noise-sigmas',default='1e-5,1e-4');ap.add_argument('--noise-seeds',type=int,default=3);ap.add_argument('--noise-dt',type=float,default=.05)
    ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13);ap.add_argument('--max-abs',type=float,default=1e7);ap.add_argument('--seed',type=int,default=6101)
    ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); amps=[float(v) for v in a.ic_amplitudes.split(',')]; sigmas=[float(v) for v in a.noise_sigmas.split(',')]
    if min(amps)<=0 or min(sigmas)<=0 or a.ic_directions<1 or a.noise_seeds<1 or a.noise_dt<=0:ap.error('invalid perturbation/noise settings')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cp=out/'p61_checkpoint.json';cf=out/'p61_config.json'
    config={'design':DESIGN,'settings':{k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')}}
    if a.resume and cf.exists() and json.loads(cf.read_text())!=config:raise ValueError('Resume configuration mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
    # deterministic unit directions fixed globally for reproducibility
    rng=np.random.default_rng(a.seed);dirs=[]
    for _ in range(a.ic_directions):
        v=rng.standard_normal(6);v/=np.linalg.norm(v);dirs.append(v)
    jobs=[]
    for tr in DESIGN:
      for side in ('left','right'):
        z=tr['zeta_'+side]; expected=tr[side]; base=dict(transition_id=tr['transition_id'],ic_id=tr['ic_id'],edge=tr['edge'],side=side,zeta=z,expected=expected,x0=tr['x0'])
        jobs.append(dict(**base,job_id=f'{tr["transition_id"]}_{side}_baseline',arm='baseline'))
        for ai,amp in enumerate(amps):
          for di in range(a.ic_directions):
            for sgn in (-1,1):jobs.append(dict(**base,job_id=f'{tr["transition_id"]}_{side}_ic_a{ai}_d{di}_{sgn:+d}',arm='ic',amplitude=amp,direction=di,sign=sgn))
        for si,sigma in enumerate(sigmas):
          for ns in range(a.noise_seeds):jobs.append(dict(**base,job_id=f'{tr["transition_id"]}_{side}_noise_s{si}_r{ns}',arm='noise',sigma=sigma,noise_rep=ns))
    if a.diagnostic_only:
        savej(out/'p61_preflight.json',{'status':'PASS','representative_transitions':len(DESIGN),'anchors':2*len(DESIGN),'jobs':len(jobs),'baseline_jobs':8,'ic_jobs':8*len(amps)*a.ic_directions*2,'noise_jobs':8*len(sigmas)*a.noise_seeds,'noise_model':'uncalibrated additive independent acceleration forcing in x3dot/x6dot; Euler-Maruyama'})
        print(f'P61 preflight PASS: {len(DESIGN)} transitions, 8 anchors, {len(jobs)} total integrations');return
    savej(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];done={r['job_id'] for r in records}
    for i,j in enumerate(jobs,1):
      if j['job_id'] in done:continue
      rec={k:v for k,v in j.items() if k!='x0'};rec['status']='failed'
      try:
        p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=j['zeta'],omega=1.,W=100.);x=np.asarray(j['x0'],float).copy()
        if j['arm']=='ic':x=x+j['sign']*j['amplitude']*dirs[j['direction']]
        if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
        if j['arm']=='noise':
            seed=a.seed+100000*next(k for k,t in enumerate(DESIGN) if t['transition_id']==j['transition_id'])+10000*(j['side']=='right')+100*j['noise_rep']+int(round(-math.log10(j['sigma'])))
            t,Y=stochastic_integrate(x,p,a.observe,a.noise_dt,j['sigma'],seed,a.max_abs);rec['noise_seed']=seed
        else:t,Y=integrate(x,p,a.observe,a.dt,a.rtol,a.atol,a.max_abs)
        o,d1,d2=classify(t,Y,j['zeta']);rec.update(status='complete',orientation=o,drift1=d1,drift2=d2,reproduced=(o==j['expected']))
      except Exception as e:rec['error']=repr(e)
      records.append(rec);done.add(j['job_id']);savej(cp,records);print(f'{len(records):03d}/{len(jobs)} {j["job_id"]}: {rec.get("orientation","failed")}',flush=True)
    fields=['job_id','arm','transition_id','ic_id','edge','side','zeta','expected','amplitude','direction','sign','sigma','noise_rep','noise_seed','status','orientation','drift1','drift2','reproduced','error']
    csvout(out/'p61_integrations.csv',records,fields)
    # aggregate robustness by arm and perturbation level
    groups={}
    for r in records:
      if r['arm']=='baseline':key=('baseline','0')
      elif r['arm']=='ic':key=('ic',f'{r["amplitude"]:.12g}')
      else:key=('noise',f'{r["sigma"]:.12g}')
      groups.setdefault(key,[]).append(r)
    summary_rows=[]
    for (arm,level),rr in groups.items():
      comp=[r for r in rr if r['status']=='complete'];summary_rows.append({'arm':arm,'level':level,'n':len(rr),'n_complete':len(comp),'n_reproduced':sum(bool(r.get('reproduced')) for r in comp),'reproduction_fraction':sum(bool(r.get('reproduced')) for r in comp)/len(comp) if comp else None,'n_ambiguous':sum(r.get('orientation')=='ambiguous' for r in comp)})
    csvout(out/'p61_robustness_summary.csv',summary_rows,['arm','level','n','n_complete','n_reproduced','reproduction_fraction','n_ambiguous'])
    # per-transition side reproduction
    pair=[]
    for tr in DESIGN:
      for side in ('left','right'):
        rr=[r for r in records if r['transition_id']==tr['transition_id'] and r['side']==side and r['arm']!='baseline' and r['status']=='complete']
        pair.append({'transition_id':tr['transition_id'],'edge':tr['edge'],'side':side,'expected':tr[side],'n_perturbed':len(rr),'n_reproduced':sum(bool(r.get('reproduced')) for r in rr),'reproduction_fraction':sum(bool(r.get('reproduced')) for r in rr)/len(rr) if rr else None})
    csvout(out/'p61_anchor_summary.csv',pair,['transition_id','edge','side','expected','n_perturbed','n_reproduced','reproduction_fraction'])
    sm={'n_planned':len(jobs),'n_complete':sum(r['status']=='complete' for r in records),'n_failed':sum(r['status']!='complete' for r in records),'n_representative_transitions':len(DESIGN),'n_anchors':8,'baseline_reproduced':sum(r['arm']=='baseline' and r.get('reproduced') for r in records),'robustness_by_level':summary_rows}
    savej(out/'p61_summary.json',sm)
    savej(out/'p61_report.json',{'scope':'P61 controlled robustness of four representative persistence-verified P60 re-entrant P/M transitions','design':DESIGN,'settings':config['settings'],'summary':sm,'interpretation_guardrails':['Initial-condition perturbation robustness is finite-time local robustness, not proof of an open asymptotic basin.','The stochastic arm is an uncalibrated numerical stress test with independent additive forcing in the two acceleration derivatives; it must not be interpreted as a calibrated physical PLL noise model.','Noise trajectories use Euler-Maruyama and therefore are not numerically identical to deterministic DOP853 trajectories; baseline reproduction is assessed separately with DOP853.','Four representative P60 transitions are tested; results do not establish robustness of all 78 transitions or global parameter-space topology.']})
    print(f'P61 finished: {sm["n_complete"]}/{len(jobs)} complete; baseline {sm["baseline_reproduced"]}/8 reproduced')
if __name__=='__main__':main()

