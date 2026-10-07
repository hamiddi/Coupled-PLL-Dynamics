#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p24_basin_attraction.py

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

def integrate(x,p,duration,dt,rtol,atol,max_abs):
    n=int(np.ceil(duration/dt)); t=np.linspace(0,duration,n+1)
    def escape(t,y):return max_abs-np.max(np.abs(y))
    escape.terminal=True;escape.direction=-1
    sol=solve_ivp(lambda t,y:rotating_rhs(t,y,p),(0,duration),x,t_eval=t,
        method='DOP853',max_step=min(.5,dt),rtol=rtol,atol=atol,events=escape)
    if not sol.success or len(sol.t)!=len(t):raise RuntimeError(f'incomplete integration: {sol.message}')
    return t,sol.y.T

def detector(Y,z):return np.column_stack((Y[:,3]-Y[:,0],z*Y[:,0]-Y[:,3]))

def stats(t,Y,z):
    ph=detector(Y,z)
    drift=np.array([np.polyfit(t,ph[:,i],1)[0] for i in range(2)])
    trig=np.column_stack((np.sin(ph),np.cos(ph)))
    return dict(drift=drift.tolist(),trig_mean=trig.mean(axis=0).tolist(),
        velocity_mean=Y[:,[1,2,4,5]].mean(axis=0).tolist(),
        velocity_std=Y[:,[1,2,4,5]].std(axis=0).tolist(),
        phase_residual_rms=[float(np.std(ph[:,i]-np.polyval(np.polyfit(t,ph[:,i],1),t))) for i in range(2)])

def classify(drift,ref,tol):
    distances={key:float(np.linalg.norm(np.array(drift)-np.array(value))) for key,value in ref.items()}
    best=min(distances,key=distances.get)
    return (best if distances[best]<=tol else 'unclassified'),distances

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
    ap.add_argument('--output-dir',default='results/author_basin_attraction')
    ap.add_argument('--per-cluster',type=int,default=2)
    ap.add_argument('--perturb-scales',default='0.01,0.1')
    ap.add_argument('--replicates',type=int,default=2)
    ap.add_argument('--transient',type=float,default=400)
    ap.add_argument('--observe',type=float,default=800)
    ap.add_argument('--windows',type=int,default=4)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--drift-tolerance',type=float,default=.08)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--seed',type=int,default=2401)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    scales=[float(x) for x in args.perturb_scales.split(',')]
    if not scales or any(not np.isfinite(x) or x<=0 for x in scales):ap.error('perturb-scales must be positive finite numbers')
    if args.per_cluster<1 or args.replicates<1 or args.windows<2 or args.dt<=0 or args.observe<=0 or args.transient<0:ap.error('invalid counts/durations')
    src=json.loads(Path(args.p23_report).read_text());out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    groups={}; refs={}
    for c in src['comparisons']:
        z=round(float(c['zeta']),9)
        refs[z]={key:val for key,val in zip(c['clusters'],c['mean_detector_drifts'])}
    for rec in src['trajectories']:
        if rec['status']=='complete':groups.setdefault((round(float(rec['zeta']),9),rec['source_cluster']),[]).append(rec)
    rng=np.random.default_rng(args.seed); jobs=[]
    for (z,cluster),members in sorted(groups.items()):
        members=sorted(members,key=lambda x:x['seed_id'])
        # Deterministic selection, reproducible across resume runs.
        for r in members[:args.per_cluster]:
            jobs.append((z,cluster,r,0.,-1,None))
            for scale in scales:
                for k in range(args.replicates):
                    direction=rng.standard_normal(6);direction/=np.linalg.norm(direction)
                    # Perturb co-rotating phases and their derivatives, not unwrapped phase offsets.
                    jobs.append((z,cluster,r,scale,k,direction))
    path=out/'p24_checkpoint.json';records=json.loads(path.read_text()) if args.resume and path.exists() else []
    done={r['job_id'] for r in records};n=len(jobs)
    for i,(z,cluster,source,scale,rep,direction) in enumerate(jobs):
        job_id=f'z{z:.3f}_{cluster}_{source["seed_id"]}_s{scale:g}_r{rep}'
        if job_id in done:continue
        p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,
            omega=float(src.get('settings',{}).get('omega',1.)),W=float(src.get('settings',{}).get('W',100.)))
        # P23 settings normally omit omega and W; P22 benchmark fixes omega=1,W=100.
        x=np.array(source['final_state'],float)
        if direction is not None:x=x+scale*direction
        record=dict(job_id=job_id,zeta=z,source_cluster=cluster,source_seed=source['seed_id'],
                    scale=scale,replicate=rep,perturb_direction=direction.tolist() if direction is not None else None)
        try:
            if args.transient:
                _,Y=integrate(x,p,args.transient,args.dt,args.rtol,args.atol,args.max_abs);x=Y[-1]
            t,Y=integrate(x,p,args.observe,args.dt,args.rtol,args.atol,args.max_abs)
            windows=[]
            for j in range(args.windows):
                lo=int(j*(len(t)-1)/args.windows);hi=int((j+1)*(len(t)-1)/args.windows)+1
                w=stats(t[lo:hi],Y[lo:hi],z);w['window']=j
                w['class'],w['distances']=classify(w['drift'],refs[z],args.drift_tolerance)
                windows.append(w)
            last=windows[-1];record.update(status='complete',windows=windows,
                final_class=last['class'],return_to_source=last['class']==cluster,
                consistent_late_class=all(w['class']==last['class'] for w in windows[-2:]),
                drift_window_range=np.ptp(np.array([w['drift'] for w in windows]),axis=0).tolist(),
                final_state=Y[-1].tolist())
        except Exception as e:record.update(status='failed',error=str(e))
        records.append(record);done.add(job_id)
        tmp=out/'p24_checkpoint.tmp';tmp.write_text(json.dumps(records,indent=2,allow_nan=False));tmp.replace(path)
        print(f'{i+1}/{n} {job_id}: {record["status"]} {record.get("final_class","")}',flush=True)
    rows=[];summary=[]
    for r in records:
        if r['status']=='complete':
            for w in r['windows']:
                rows.append(dict(job_id=r['job_id'],zeta=r['zeta'],source_cluster=r['source_cluster'],
                    source_seed=r['source_seed'],scale=r['scale'],replicate=r['replicate'],window=w['window'],
                    drift1=w['drift'][0],drift2=w['drift'][1],classification=w['class'],
                    distance_to_source=w['distances'][r['source_cluster']]))
    for (z,cluster),members in sorted(groups.items()):
        for scale in [0.]+scales:
            rr=[r for r in records if r['zeta']==z and r['source_cluster']==cluster and r['scale']==scale]
            completed=[r for r in rr if r['status']=='complete']
            summary.append(dict(zeta=z,source_cluster=cluster,scale=scale,total=len(rr),complete=len(completed),
                return_count=sum(r['return_to_source'] for r in completed),
                switch_count=sum(r['final_class'] not in (cluster,'unclassified') for r in completed),
                unclassified_count=sum(r['final_class']=='unclassified' for r in completed),
                consistent_late_count=sum(r['consistent_late_class'] for r in completed)))
    report=dict(scope=src.get('scope'),source_p23=args.p23_report,settings=vars(args),
        reference_drifts={str(k):v for k,v in refs.items()},summary=summary,trajectories=records,
        limitations=['Finite-time return to a drift class does not prove asymptotic attraction.',
        'No exhaustive basin volume is estimated: perturbations are local and finite in number.',
        'The unwrapped phase coordinates drift; compare detector phases modulo 2pi and rotation rates.',
        'Near-zero or positive finite-time Lyapunov exponents require separate convergence studies.',
        'This is a normalized benchmark, not a physical PLL parameter calibration.'])
    (out/'p24_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'p24_windows.csv').open('w',newline='') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'p24_summary.csv').open('w',newline='') as f:
        if summary:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps(summary,indent=2));print('Saved',out/'p24_report.json')
if __name__=='__main__':main()

