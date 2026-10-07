#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p58_synchronization_transition_mechanism.py

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
import argparse,csv,json,hashlib,math
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate

def readcsv(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def save(p,o):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rr,ff):
    with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=ff,extrasaction='ignore');w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
    if not x:raise ValueError(m)
def orientation(d,zt=.05):
    d1,d2=map(float,d)
    if d1 < -zt and d2 > zt:return 'M'
    if d1 > zt and d2 < -zt:return 'P'
    return 'ambiguous'
def circ_R(phi): return float(abs(np.mean(np.exp(1j*np.asarray(phi)))))
def slope(t,y):
    t=np.asarray(t);y=np.asarray(y);q=np.polyfit(t-t[0],y,1);return float(q[0])
def metrics(t,Y,zeta):
    th1=Y[:,0];v1=Y[:,1];a1=Y[:,2];th2=Y[:,3];v2=Y[:,4];a2=Y[:,5]
    phi1=th2-th1; phi2=zeta*th1-th2
    dur=float(t[-1]-t[0]); d1=float((phi1[-1]-phi1[0])/dur); d2=float((phi2[-1]-phi2[0])/dur)
    w1=v2-v1; w2=zeta*v1-v2
    # Net slips are signed; total variation counts accumulated winding regardless of direction.
    out=dict(drift1=d1,drift2=d2,orientation=orientation((d1,d2)),
      phi1_R=circ_R(phi1),phi2_R=circ_R(phi2),
      phi1_net_slips=float((phi1[-1]-phi1[0])/(2*np.pi)),phi2_net_slips=float((phi2[-1]-phi2[0])/(2*np.pi)),
      phi1_total_slips=float(np.sum(np.abs(np.diff(phi1)))/(2*np.pi)),phi2_total_slips=float(np.sum(np.abs(np.diff(phi2)))/(2*np.pi)),
      mismatch1_mean=float(np.mean(w1)),mismatch1_abs_mean=float(np.mean(np.abs(w1))),mismatch1_rms=float(np.sqrt(np.mean(w1*w1))),mismatch1_std=float(np.std(w1)),
      mismatch2_mean=float(np.mean(w2)),mismatch2_abs_mean=float(np.mean(np.abs(w2))),mismatch2_rms=float(np.sqrt(np.mean(w2*w2))),mismatch2_std=float(np.std(w2)),
      v1_mean=float(np.mean(v1)),v2_mean=float(np.mean(v2)),v1_std=float(np.std(v1)),v2_std=float(np.std(v2)),
      a1_rms=float(np.sqrt(np.mean(a1*a1))),a2_rms=float(np.sqrt(np.mean(a2*a2))))
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p55-fixed-ics',default='results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv')
    ap.add_argument('--p57-thresholds',default='results/author_high_resolution_coupling_threshold_refinement/p57_refined_thresholds.csv')
    ap.add_argument('--p57-report',default='results/author_high_resolution_coupling_threshold_refinement/p57_report.json')
    ap.add_argument('--output-dir',default='results/author_synchronization_transition_mechanism')
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=6400.)
    ap.add_argument('--dt',type=float,default=.5);ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); paths={'p55_fixed_ics':Path(a.p55_fixed_ics),'p57_thresholds':Path(a.p57_thresholds),'p57_report':Path(a.p57_report)}
    for k,p in paths.items():req(p.is_file(),f'Missing {k}: {p}')
    rep=json.loads(paths['p57_report'].read_text()); req(rep.get('n_refined')==23 and rep.get('n_both_endpoints_persistent')==23,'P57 must contain 23/23 persistent refined boundaries')
    ics={r['ic_id']:r for r in readcsv(paths['p55_fixed_ics'])}; thr=readcsv(paths['p57_thresholds']); req(len(thr)==23,'Expected 23 P57 thresholds')
    for r in thr:
      req(r['status']=='refined' and r['both_endpoints_persistent']=='True','Every P57 threshold must be refined and persistent')
      req(r['orientation_left']=='P' and r['orientation_right']=='M','Expected P->M boundaries')
      req(r['ic_id'] in ics,f'Missing IC {r["ic_id"]}')
    X={k:np.array([float(v[f'x{i}']) for i in range(1,7)],float) for k,v in ics.items()}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    source_sha={k:sha(v) for k,v in paths.items()}; settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')}; config=dict(source_sha256=source_sha,settings=settings,n_thresholds=23)
    cf=out/'p58_config.json'; cp=out/'p58_checkpoint.json'
    if a.resume and cf.exists():req(json.loads(cf.read_text())==config,'Resume configuration mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
    if a.diagnostic_only:
      save(out/'p58_preflight.json',dict(status='PASS',n_thresholds=23,conditions_per_ic=4,planned_integrations=92,observe=a.observe,conditions=['anchor_P_0.8990','threshold_left_P','threshold_right_M','anchor_M_0.8991']))
      print('P58 preflight PASS: 23 ICs x 4 coupling conditions = 92 integrations')
      return
    save(cf,config); records=json.loads(cp.read_text()) if a.resume and cp.exists() else []; by={r['job_id']:r for r in records};req(len(by)==len(records),'Duplicate checkpoint IDs')
    def run(ic,z,condition,expected):
      jid=f'{ic}_{condition}_z{z:.12f}'
      if jid in by:return by[jid]
      p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.);x=X[ic].copy();rec=dict(job_id=jid,ic_id=ic,condition=condition,zeta=z,expected_orientation=expected,status='failed')
      try:
        if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
        t,Y=integrate(x,p,a.observe,a.dt,a.rtol,a.atol,a.max_abs); rec.update(status='complete',**metrics(t,Y,z));rec['matches_expected']=(rec['orientation']==expected)
      except Exception as ex:rec['error']=repr(ex)
      records.append(rec);by[jid]=rec;save(cp,records);print(f'{len(records):03d}/92 {ic} {condition}: {rec.get("orientation","failed")}',flush=True);return rec
    pairs=[]
    for r in thr:
      ic=r['ic_id']; zl=float(r['zeta_left']);zr=float(r['zeta_right'])
      aa=run(ic,.899,'anchor_P_0.8990','P'); L=run(ic,zl,'threshold_left_P','P'); R=run(ic,zr,'threshold_right_M','M'); bb=run(ic,.8991,'anchor_M_0.8991','M')
      q=dict(ic_id=ic,edge_index=int(r['edge_index']),row_index=int(r['row_index']),band_index=int(r['band_index']),zeta_left=zl,zeta_right=zr,zeta_midpoint=float(r['zeta_midpoint']),bracket_width=float(r['bracket_width']),left_orientation=L.get('orientation'),right_orientation=R.get('orientation'),transition_reproduced=(L.get('orientation')=='P' and R.get('orientation')=='M'))
      for key in ['drift1','drift2','phi1_R','phi2_R','phi1_net_slips','phi2_net_slips','phi1_total_slips','phi2_total_slips','mismatch1_abs_mean','mismatch1_rms','mismatch1_std','mismatch2_abs_mean','mismatch2_rms','mismatch2_std','v1_mean','v2_mean','v1_std','v2_std','a1_rms','a2_rms']:
        if key in L and key in R:q[f'left_{key}']=L[key];q[f'right_{key}']=R[key];q[f'delta_{key}']=R[key]-L[key]
      pairs.append(q)
    fields=['job_id','ic_id','condition','zeta','expected_orientation','status','orientation','matches_expected','drift1','drift2','phi1_R','phi2_R','phi1_net_slips','phi2_net_slips','phi1_total_slips','phi2_total_slips','mismatch1_mean','mismatch1_abs_mean','mismatch1_rms','mismatch1_std','mismatch2_mean','mismatch2_abs_mean','mismatch2_rms','mismatch2_std','v1_mean','v2_mean','v1_std','v2_std','a1_rms','a2_rms','error']
    csvout(out/'p58_metrics.csv',records,fields)
    pfields=['ic_id','edge_index','row_index','band_index','zeta_left','zeta_right','zeta_midpoint','bracket_width','left_orientation','right_orientation','transition_reproduced']+[f'{s}_{k}' for k in ['drift1','drift2','phi1_R','phi2_R','phi1_net_slips','phi2_net_slips','phi1_total_slips','phi2_total_slips','mismatch1_abs_mean','mismatch1_rms','mismatch1_std','mismatch2_abs_mean','mismatch2_rms','mismatch2_std','v1_mean','v2_mean','v1_std','v2_std','a1_rms','a2_rms'] for s in ('left','right','delta')]
    csvout(out/'p58_paired_transition_metrics.csv',pairs,pfields)
    complete=[r for r in records if r['status']=='complete']; matched=[r for r in complete if r.get('matches_expected')]
    def stat(k,cond):
      v=np.array([r[k] for r in complete if r['condition']==cond and k in r],float);return dict(n=len(v),mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else 0.,min=float(np.min(v)),max=float(np.max(v))) if len(v) else None
    keys=['drift1','drift2','phi1_R','phi2_R','mismatch1_abs_mean','mismatch1_rms','mismatch2_abs_mean','mismatch2_rms','phi1_total_slips','phi2_total_slips']
    conds=['anchor_P_0.8990','threshold_left_P','threshold_right_M','anchor_M_0.8991']
    summary=dict(n_integrations=len(records),n_complete=len(complete),n_failed=len(records)-len(complete),n_expected_orientation_matches=len(matched),n_transition_pairs=len(pairs),n_transition_pairs_reproduced=sum(bool(r['transition_reproduced']) for r in pairs),condition_statistics={c:{k:stat(k,c) for k in keys} for c in conds},paired_delta_statistics={})
    for k in keys:
      vals=np.array([r.get('delta_'+k,np.nan) for r in pairs],float);vals=vals[np.isfinite(vals)];summary['paired_delta_statistics'][k]=dict(n=len(vals),mean=float(np.mean(vals)),median=float(np.median(vals)),std=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,min=float(np.min(vals)),max=float(np.max(vals)))
    save(out/'p58_summary.json',summary)
    report=dict(scope='P58 synchronization-observable characterization across the 23 P57 persistent finite-time P->M orientation boundaries',source_sha256=source_sha,settings=settings,summary=summary,observable_definitions={'M':'detector drifts (-,+)','P':'detector drifts (+,-)','phi1':'theta2-theta1','phi2':'zeta*theta1-theta2','phi_R':'circular resultant length |mean(exp(i phi))|; 1 concentrated, 0 dispersed','mismatch1':'d(phi1)/dt = v2-v1','mismatch2':'d(phi2)/dt = zeta*v1-v2','net_slips':'net unwrapped detector-phase change / 2pi','total_slips':'total variation of unwrapped detector phase / 2pi'},limitations=['These are finite-time synchronization observables for selected fixed initial conditions; they do not establish asymptotic synchronization or an invariant-set bifurcation.','A nonzero detector drift indicates phase slipping rather than strict phase locking; circular concentration is descriptive even when phase slips occur.','The left/right P57 endpoints are separated by about 1e-7 in zeta, so discontinuous-looking metric changes should be interpreted as regime-selection changes for these ICs, not automatically as a discontinuity of an attractor branch.','P58 characterizes the mechanism at 23 previously identified switching ICs and does not estimate global basin fractions.'])
    save(out/'p58_report.json',report);print(f'P58 finished: {len(complete)}/{len(records)} complete; expected orientations {len(matched)}/{len(complete)}; reproduced transitions {summary["n_transition_pairs_reproduced"]}/23')
if __name__=='__main__':main()

