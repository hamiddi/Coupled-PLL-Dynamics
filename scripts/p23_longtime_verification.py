#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p23_longtime_verification.py

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
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p04_periodic_orbit_floquet import full_jacobian

def phases(Y,z):
    return np.column_stack((Y[:,3]-Y[:,0],z*Y[:,0]-Y[:,3]))

def integrate(y,p,T,dt,rtol,atol,max_abs):
    t=np.linspace(0,T,int(np.ceil(T/dt))+1)
    def escape(t,x):return max_abs-np.max(np.abs(x))
    escape.terminal=True;escape.direction=-1
    sol=solve_ivp(lambda t,x:rotating_rhs(t,x,p),(0,T),y,method='DOP853',
        t_eval=t,rtol=rtol,atol=atol,max_step=min(.5,dt),events=escape)
    if not sol.success or len(sol.t)!=len(t):
        raise RuntimeError(f'integration incomplete: {sol.message}')
    return sol.t,sol.y.T

def window_stats(t,Y,z):
    ph=phases(Y,z);v=Y[:,[1,2,4,5]]
    slopes=[float(np.polyfit(t,ph[:,k],1)[0]) for k in range(2)]
    fit=[float(np.sqrt(np.mean((ph[:,k]-(np.polyval(np.polyfit(t,ph[:,k],1),t)))**2))) for k in range(2)]
    trig=np.column_stack((np.sin(ph),np.cos(ph)))
    return dict(detector_drift=slopes,detector_detrended_rms=fit,
        velocity_mean=v.mean(axis=0).tolist(),velocity_std=v.std(axis=0).tolist(),
        trig_mean=trig.mean(axis=0).tolist(),trig_std=trig.std(axis=0).tolist(),
        max_rhs=float(max(np.linalg.norm(rotating_rhs(0,x,params_placeholder),ord=np.inf) for x in Y[::max(1,len(Y)//50)])) if False else None)

def feature(w):
    return np.array(w['detector_drift']+w['detector_detrended_rms']+w['velocity_mean']+w['velocity_std']+w['trig_mean']+w['trig_std'])

def ftle(y,p,T,chunk,rtol,atol):
    if T<=0:return None
    v=np.ones(6)/np.sqrt(6);s=0.;elapsed=0.
    while elapsed<T-1e-9:
        h=min(chunk,T-elapsed)
        def f(t,u):return np.r_[rotating_rhs(t,u[:6],p),full_jacobian(u[:6],p)@u[6:]]
        sol=solve_ivp(f,(0,h),np.r_[y,v],method='DOP853',rtol=rtol,atol=atol,max_step=.5)
        if not sol.success:return None
        y=sol.y[:6,-1];w=sol.y[6:,-1];n=np.linalg.norm(w)
        if not np.isfinite(n) or n==0:return None
        s+=np.log(n);v=w/n;elapsed+=h
    return float(s/elapsed)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p22-report',default='results/author_attractor_survey/p22_report.json')
    ap.add_argument('--output-dir',default='results/author_longtime_verification')
    ap.add_argument('--per-cluster',type=int,default=3)
    ap.add_argument('--additional-transient',type=float,default=600.)
    ap.add_argument('--observe',type=float,default=1200.)
    ap.add_argument('--windows',type=int,default=4)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--lyapunov-time',type=float,default=200.)
    ap.add_argument('--lyapunov-chunk',type=float,default=2.)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--seed',type=int,default=2301)
    args=ap.parse_args()
    if args.per_cluster<1 or args.windows<2 or args.observe<=0 or args.dt<=0:ap.error('invalid counts/durations')
    src=json.loads(Path(args.p22_report).read_text());base=src['settings'];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    traj={(round(float(r['zeta']),9),r['seed_id']):r for r in src['trajectories']}
    rng=np.random.default_rng(args.seed);selected=[]
    for c in src['candidate_clusters']:
        members=[traj[(round(float(c['zeta']),9),name)] for name in c['members'] if (round(float(c['zeta']),9),name) in traj]
        # Random representatives avoid privileging the first equilibrium-origin seed.
        indices=rng.choice(len(members),size=min(args.per_cluster,len(members)),replace=False)
        for i in indices:selected.append((c['cluster_id'],members[int(i)]))
    records=[];rows=[]
    for k,(cluster,r) in enumerate(selected):
        z=float(r['zeta']);p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=float(base.get('omega',1.)),W=float(base.get('W',100.)))
        # Prefer P22 final states, continuing already-cleared trajectories.
        x=np.asarray(r['final_state'],float);rec=dict(zeta=z,source_cluster=cluster,seed_id=r['seed_id'],source_p22_ftle=r.get('lyapunov_ftle'))
        try:
            if args.additional_transient>0:_,Y=integrate(x,p,args.additional_transient,args.dt,args.rtol,args.atol,args.max_abs);x=Y[-1]
            t,Y=integrate(x,p,args.observe,args.dt,args.rtol,args.atol,args.max_abs)
            windows=[]
            for j in range(args.windows):
                lo=int(j*(len(t)-1)/args.windows);hi=int((j+1)*(len(t)-1)/args.windows)+1
                w=window_stats(t[lo:hi],Y[lo:hi],z);w['index']=j;w['start']=float(t[lo]);w['end']=float(t[hi-1]);windows.append(w)
                rows.append(dict(zeta=z,source_cluster=cluster,seed_id=r['seed_id'],window=j,
                    detector1_drift=w['detector_drift'][0],detector2_drift=w['detector_drift'][1],
                    detector1_detrended_rms=w['detector_detrended_rms'][0],detector2_detrended_rms=w['detector_detrended_rms'][1],
                    v1_mean=w['velocity_mean'][0],v2_mean=w['velocity_mean'][2]))
            features=np.array([feature(w) for w in windows]);drift=np.array([w['detector_drift'] for w in windows]);
            rec.update(status='complete',windows=windows,late_feature=features[-1].tolist(),
                window_feature_change=float(np.linalg.norm(features[-1]-features[-2])/(1+np.linalg.norm(features[-1]))),
                drift_range=np.ptp(drift,axis=0).tolist(),final_state=Y[-1].tolist(),
                late_ftle=ftle(Y[-1],p,args.lyapunov_time,args.lyapunov_chunk,args.rtol,args.atol))
        except Exception as e:rec.update(status='failed',error=str(e))
        records.append(rec);print(f'{k+1}/{len(selected)} zeta={z:.3f} {cluster} {r["seed_id"]}: {rec["status"]}',flush=True)
        tmp=out/'p23_checkpoint.tmp';tmp.write_text(json.dumps(records,indent=2,allow_nan=False));tmp.replace(out/'p23_checkpoint.json')
    comparisons=[]
    for z in sorted(set(r['zeta'] for r in records)):
        group=[r for r in records if r['zeta']==z and r['status']=='complete'];names=sorted(set(r['source_cluster'] for r in group))
        if len(names)!=2:continue
        g1=[r for r in group if r['source_cluster']==names[0]];g2=[r for r in group if r['source_cluster']==names[1]]
        A=np.array([r['late_feature'] for r in g1]);B=np.array([r['late_feature'] for r in g2]);
        # Normalize by pooled within-group scale; report both raw and normalized separations.
        within=np.r_[np.linalg.norm(A-A.mean(axis=0),axis=1),np.linalg.norm(B-B.mean(axis=0),axis=1)]
        dist=float(np.linalg.norm(A.mean(axis=0)-B.mean(axis=0)))
        comparisons.append(dict(zeta=z,clusters=names,n_per_cluster=[len(A),len(B)],
            centroid_distance=dist,mean_within_distance=float(within.mean()),
            separation_ratio=float(dist/(within.mean()+1e-12)),
            mean_detector_drifts=[np.mean([r['windows'][-1]['detector_drift'] for r in g],axis=0).tolist() for g in (g1,g2)],
            note='Exploratory separation, not proof of distinct attracting invariant sets.'))
    report=dict(scope=src['scope'],source_p22=str(args.p22_report),settings=vars(args),trajectories=records,
        comparisons=comparisons,limitations=['Finite-time results cannot establish exhaustive basin structure or chaos.',
        'P22 clustering may split one attractor by drift direction or transient history.',
        'Distinct mean drift can reflect different rotation numbers, but persistence and attraction require further checks.'])
    (out/'p23_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'p23_windows.csv').open('w',newline='') as f:
        if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'p23_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2));print('Saved',out/'p23_report.json')
if __name__=='__main__':main()

