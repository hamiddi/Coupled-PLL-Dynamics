#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p59_critical_transition_early_warning.py

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
def circ_R(phi):return float(abs(np.mean(np.exp(1j*np.asarray(phi)))))
def ac(x,lag):
    x=np.asarray(x,float); lag=int(lag)
    if lag<1 or len(x)<=lag+2:return float('nan')
    a=x[:-lag]-np.mean(x[:-lag]);b=x[lag:]-np.mean(x[lag:]);den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>0 else 0.0
def rankdata(a):
    a=np.asarray(a,float); order=np.argsort(a,kind='mergesort'); ranks=np.empty(len(a),float);i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        ranks[order[i:j]]=(i+j-1)/2+1;i=j
    return ranks
def spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y);x=x[m];y=y[m]
    if len(x)<3:return float('nan')
    rx=rankdata(x);ry=rankdata(y);rx-=rx.mean();ry-=ry.mean();den=np.sqrt(np.dot(rx,rx)*np.dot(ry,ry))
    return float(np.dot(rx,ry)/den) if den>0 else 0.0
def metrics(t,Y,zeta,blocks=16):
    th1=Y[:,0];v1=Y[:,1];th2=Y[:,3];v2=Y[:,4]
    phi1=th2-th1;phi2=zeta*th1-th2;dur=float(t[-1]-t[0]);w1=v2-v1;w2=zeta*v1-v2
    d1=float((phi1[-1]-phi1[0])/dur);d2=float((phi2[-1]-phi2[0])/dur)
    bd1=[];bd2=[]
    for j in range(blocks):
        lo=int(j*(len(t)-1)/blocks);hi=int((j+1)*(len(t)-1)/blocks)
        if hi<=lo:continue
        dd=float(t[hi]-t[lo]);bd1.append(float((phi1[hi]-phi1[lo])/dd));bd2.append(float((phi2[hi]-phi2[lo])/dd))
    return dict(orientation=orientation((d1,d2)),drift1=d1,drift2=d2,
      phi1_R=circ_R(phi1),phi2_R=circ_R(phi2),
      slip1_rate=float(np.sum(np.abs(np.diff(phi1)))/(2*np.pi*dur)),slip2_rate=float(np.sum(np.abs(np.diff(phi2)))/(2*np.pi*dur)),
      mismatch1_mean=float(np.mean(w1)),mismatch1_abs_mean=float(np.mean(np.abs(w1))),mismatch1_var=float(np.var(w1)),mismatch1_std=float(np.std(w1)),mismatch1_rms=float(np.sqrt(np.mean(w1*w1))),
      mismatch2_mean=float(np.mean(w2)),mismatch2_abs_mean=float(np.mean(np.abs(w2))),mismatch2_var=float(np.var(w2)),mismatch2_std=float(np.std(w2)),mismatch2_rms=float(np.sqrt(np.mean(w2*w2))),
      mismatch1_ac1=ac(w1,1),mismatch2_ac1=ac(w2,1),mismatch1_ac10=ac(w1,10),mismatch2_ac10=ac(w2,10),
      block_drift1_std=float(np.std(bd1,ddof=1)),block_drift2_std=float(np.std(bd2,ddof=1)),
      block_drift1_range=float(np.ptp(bd1)),block_drift2_range=float(np.ptp(bd2)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p55-fixed-ics',default='results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv')
    ap.add_argument('--p57-thresholds',default='results/author_high_resolution_coupling_threshold_refinement/p57_refined_thresholds.csv')
    ap.add_argument('--p57-report',default='results/author_high_resolution_coupling_threshold_refinement/p57_report.json')
    ap.add_argument('--output-dir',default='results/author_critical_transition_early_warning')
    ap.add_argument('--distances',default='0,1e-7,2e-7,4e-7,8e-7,1.6e-6,3.2e-6,6.4e-6,1.28e-5')
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=6400.);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13);ap.add_argument('--max-abs',type=float,default=1e7);ap.add_argument('--blocks',type=int,default=16)
    ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();paths={'p55_fixed_ics':Path(a.p55_fixed_ics),'p57_thresholds':Path(a.p57_thresholds),'p57_report':Path(a.p57_report)}
    for k,p in paths.items():req(p.is_file(),f'Missing {k}: {p}')
    rep=json.loads(paths['p57_report'].read_text());req(rep.get('n_refined')==23 and rep.get('n_both_endpoints_persistent')==23,'P57 must contain 23/23 persistent refined boundaries')
    ics={r['ic_id']:r for r in readcsv(paths['p55_fixed_ics'])};thr=readcsv(paths['p57_thresholds']);req(len(thr)==23,'Expected 23 P57 thresholds')
    distances=[float(x) for x in a.distances.split(',')];req(len(distances)>=5 and min(distances)>=0 and len(set(distances))==len(distances),'Need >=5 unique nonnegative distances')
    distances=sorted(distances)
    for r in thr:
        req(r['status']=='refined' and r['both_endpoints_persistent']=='True','Every threshold must be refined/persistent');req(r['orientation_left']=='P' and r['orientation_right']=='M','Expected P->M');req(r['ic_id'] in ics,f'Missing IC {r["ic_id"]}')
    X={k:np.array([float(v[f'x{i}']) for i in range(1,7)],float) for k,v in ics.items()}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cp=out/'p59_checkpoint.json';cf=out/'p59_config.json'
    source_sha={k:sha(v) for k,v in paths.items()};settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')};config=dict(source_sha256=source_sha,settings=settings,n_thresholds=23,distances=distances)
    if a.resume and cf.exists():req(json.loads(cf.read_text())==config,'Resume configuration mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
    if a.diagnostic_only:
        save(out/'p59_preflight.json',dict(status='PASS',n_thresholds=23,n_distances=len(distances),planned_integrations=23*len(distances),distances_below_p57_left_endpoint=distances,observe=a.observe,note='distance=0 is the persistence-verified P57 left endpoint; positive distances approach it from lower zeta'))
        print(f'P59 preflight PASS: 23 ICs x {len(distances)} P-side distances = {23*len(distances)} integrations');return
    save(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];by={r['job_id']:r for r in records};req(len(by)==len(records),'Duplicate checkpoint IDs')
    total=23*len(distances)
    for tr in thr:
        ic=tr['ic_id'];zl=float(tr['zeta_left']);zm=float(tr['zeta_midpoint']);bw=float(tr['bracket_width'])
        for dist in distances:
            z=zl-dist;jid=f'{ic}_d{dist:.3e}_z{z:.12f}'
            if jid in by:continue
            rec=dict(job_id=jid,ic_id=ic,edge_index=int(tr['edge_index']),row_index=int(tr['row_index']),band_index=int(tr['band_index']),zeta=z,zeta_left=zl,zeta_midpoint=zm,bracket_width=bw,distance_below_left=dist,distance_to_midpoint=zm-z,proximity_score=-math.log10(max(zm-z,1e-16)),status='failed')
            try:
                p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.);x=X[ic].copy()
                if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
                t,Y=integrate(x,p,a.observe,a.dt,a.rtol,a.atol,a.max_abs);rec.update(status='complete',**metrics(t,Y,z,a.blocks));rec['p_side_retained']=(rec['orientation']=='P')
            except Exception as ex:rec['error']=repr(ex)
            records.append(rec);by[jid]=rec;save(cp,records);print(f'{len(records):03d}/{total} {ic} d={dist:.2e}: {rec.get("orientation","failed")}',flush=True)
    fields=['job_id','ic_id','edge_index','row_index','band_index','zeta','zeta_left','zeta_midpoint','bracket_width','distance_below_left','distance_to_midpoint','proximity_score','status','orientation','p_side_retained','drift1','drift2','phi1_R','phi2_R','slip1_rate','slip2_rate','mismatch1_mean','mismatch1_abs_mean','mismatch1_var','mismatch1_std','mismatch1_rms','mismatch2_mean','mismatch2_abs_mean','mismatch2_var','mismatch2_std','mismatch2_rms','mismatch1_ac1','mismatch2_ac1','mismatch1_ac10','mismatch2_ac10','block_drift1_std','block_drift2_std','block_drift1_range','block_drift2_range','error']
    csvout(out/'p59_metrics.csv',records,fields)
    complete=[r for r in records if r['status']=='complete'];pvalid=[r for r in complete if r.get('p_side_retained')]
    indicators=['mismatch1_var','mismatch2_var','mismatch1_ac1','mismatch2_ac1','mismatch1_ac10','mismatch2_ac10','phi1_R','phi2_R','slip1_rate','slip2_rate','mismatch1_rms','mismatch2_rms','block_drift1_std','block_drift2_std']
    trends=[]
    for tr in thr:
        ic=tr['ic_id'];rr=sorted([r for r in pvalid if r['ic_id']==ic],key=lambda r:r['distance_to_midpoint'],reverse=True)
        row=dict(ic_id=ic,edge_index=int(tr['edge_index']),row_index=int(tr['row_index']),band_index=int(tr['band_index']),n_complete=sum(r['ic_id']==ic for r in complete),n_p_side=len(rr),all_points_p_side=(len(rr)==len(distances)))
        prox=[r['proximity_score'] for r in rr]
        for k in indicators:row['rho_proximity_'+k]=spearman(prox,[r[k] for r in rr])
        trends.append(row)
    tfields=['ic_id','edge_index','row_index','band_index','n_complete','n_p_side','all_points_p_side']+['rho_proximity_'+k for k in indicators]
    csvout(out/'p59_ic_trends.csv',trends,tfields)
    distance_summary=[]
    for dist in distances:
        rr=[r for r in pvalid if abs(r['distance_below_left']-dist)<=max(1e-15,abs(dist)*1e-9)]
        q=dict(distance_below_left=dist,n_p_side=len(rr))
        for k in indicators:
            v=np.array([r[k] for r in rr],float);q[k+'_mean']=float(np.mean(v)) if len(v) else '';q[k+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else (0.0 if len(v)==1 else '')
        distance_summary.append(q)
    dfields=['distance_below_left','n_p_side']+sum(([k+'_mean',k+'_std'] for k in indicators),[])
    csvout(out/'p59_distance_summary.csv',distance_summary,dfields)
    aggregate={}
    for k in indicators:
        vals=np.array([r['rho_proximity_'+k] for r in trends],float);vals=vals[np.isfinite(vals)]
        aggregate[k]=dict(n=int(len(vals)),mean_rho=float(np.mean(vals)),median_rho=float(np.median(vals)),n_positive=int(np.sum(vals>0)),n_negative=int(np.sum(vals<0)),n_abs_ge_0_5=int(np.sum(np.abs(vals)>=.5)))
    # trend of across-IC means versus proximity; use midpoint distance = distance + half bracket width mean approximately, but actual rows are used directly pooled by level
    group_trends={}
    for k in indicators:
        xs=[];ys=[]
        for dist in distances:
            rr=[r for r in pvalid if abs(r['distance_below_left']-dist)<=max(1e-15,abs(dist)*1e-9)]
            if rr:xs.append(float(np.mean([r['proximity_score'] for r in rr])));ys.append(float(np.mean([r[k] for r in rr])))
        group_trends[k]=dict(n_levels=len(xs),rho_proximity_of_level_mean=spearman(xs,ys))
    summary=dict(n_planned=total,n_records=len(records),n_complete=len(complete),n_failed=len(records)-len(complete),n_p_side_retained=len(pvalid),n_non_p_complete=len(complete)-len(pvalid),n_ics_all_points_p_side=sum(bool(r['all_points_p_side']) for r in trends),distances_below_left=distances,aggregate_ic_spearman=aggregate,across_distance_level_trends=group_trends)
    save(out/'p59_summary.json',summary)
    report=dict(scope='P59 deterministic candidate early-warning indicators approaching 23 individualized P57 finite-time P->M regime-selection boundaries from the P side',source_sha256=source_sha,settings=settings,summary=summary,indicator_definitions={'proximity_score':'-log10(zeta_midpoint-zeta); larger means closer to the refined boundary','mismatch_var':'variance of instantaneous detector-frequency mismatch over the 6400-unit observation','mismatch_ac1':'lag-1 sample autocorrelation at dt=0.5','mismatch_ac10':'lag-10 sample autocorrelation (5 time units at default dt)','phi_R':'circular resultant length of detector phase','slip_rate':'total variation of unwrapped detector phase / (2*pi*observation duration)','block_drift_std':'standard deviation of detector drift estimated in 16 equal observation blocks','rho_proximity_*':'Spearman correlation between an indicator and proximity to the boundary within each fixed IC'},interpretation_guardrails=['These are deterministic trajectory statistics, not stochastic critical-slowing-down diagnostics. High autocorrelation is expected for smoothly sampled ODE trajectories and must not be interpreted alone as critical slowing down.','Only points that remain in physical orientation P are used for approach-side trend summaries; non-P/re-entrant points are retained in p59_metrics.csv and reported as a validity warning.','A systematic trend is descriptive evidence of a candidate precursor, not proof of a bifurcation or universal early-warning signal.','The P57 thresholds are finite-time regime-selection boundaries for selected initial conditions; P59 does not establish global basin topology or asymptotic attraction.'])
    save(out/'p59_report.json',report);print(f'P59 finished: {len(complete)}/{total} complete; P-side points {len(pvalid)}/{len(complete)}; ICs with all approach points P {summary["n_ics_all_points_p_side"]}/23')
if __name__=='__main__':main()

