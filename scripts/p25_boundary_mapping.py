#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p25_boundary_mapping.py

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

def classify(t,Y,z,refs,tol):
    ph=np.column_stack((Y[:,3]-Y[:,0],z*Y[:,0]-Y[:,3]))
    drifts=[float(np.polyfit(t,ph[:,i],1)[0]) for i in range(2)]
    distances={k:float(np.linalg.norm(np.array(drifts)-np.array(v))) for k,v in refs.items()}
    best=min(distances,key=distances.get)
    return dict(drift=drifts,classification=best if distances[best]<=tol else 'unclassified',distances=distances)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
    ap.add_argument('--p24-report',default='results/author_basin_attraction/p24_report.json')
    ap.add_argument('--output-dir',default='results/author_boundary_mapping')
    ap.add_argument('--scales',default='0,0.005,0.01,0.02,0.04,0.06,0.08,0.09,0.095,0.1,0.105,0.11,0.12,0.15,0.2')
    ap.add_argument('--angular-degrees',default='0,5,15,30')
    ap.add_argument('--angular-replicates',type=int,default=2)
    ap.add_argument('--angular-scales',default='0.06,0.08,0.1,0.12,0.15')
    ap.add_argument('--transient',type=float,default=400)
    ap.add_argument('--observe',type=float,default=1600)
    ap.add_argument('--windows',type=int,default=8)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--drift-tolerance',type=float,default=.08)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--seed',type=int,default=2501)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.windows<4 or args.observe<=0 or args.dt<=0 or args.angular_replicates<1:ap.error('invalid settings')
    p23=json.loads(Path(args.p23_report).read_text());p24=json.loads(Path(args.p24_report).read_text())
    switches=[r for r in p24['trajectories'] if r['status']=='complete' and r['final_class'] not in (r['source_cluster'],'unclassified')]
    if not switches:raise ValueError('No documented P24 switch found')
    # Explicitly select the P24 event at zeta=0.899, C01->C00, magnitude 0.1.
    matches=[r for r in switches if abs(r['zeta']-.899)<1e-9 and abs(r['scale']-.1)<1e-9 and r['source_cluster'].endswith('C01')]
    if len(matches)!=1:raise ValueError(f'Expected exactly one P24 target switch; found {len(matches)}')
    target=matches[0];z=float(target['zeta']);src=next((r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-9 and r['seed_id']==target['source_seed'] and r['source_cluster']==target['source_cluster']),None)
    if src is None:raise ValueError('P24 target source not found in P23; verify matching files')
    x0=np.asarray(src['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    c=next(c for c in p23['comparisons'] if abs(c['zeta']-z)<1e-9)
    refs=dict(zip(c['clusters'],c['mean_detector_drifts']))
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    scales=[float(s) for s in args.scales.split(',')];ang_scales=[float(s) for s in args.angular_scales.split(',')];angles=[float(s) for s in args.angular_degrees.split(',')]
    if any(s<0 or not np.isfinite(s) for s in scales+ang_scales) or any(a<0 or a>180 for a in angles):ap.error('invalid scales/angles')
    rng=np.random.default_rng(args.seed);jobs=[]
    for s in scales:jobs.append((f'ray_s{s:.8g}',s,0.,0,base.copy()))
    for angle in angles:
        if angle==0:continue
        for rep in range(args.angular_replicates):
            v=rng.standard_normal(6);v-=v.dot(base)*base;v/=np.linalg.norm(v)
            direction=np.cos(np.deg2rad(angle))*base+np.sin(np.deg2rad(angle))*v
            for s in ang_scales:jobs.append((f'angle{angle:g}_rep{rep}_s{s:.8g}',s,angle,rep,direction.copy()))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'p25_checkpoint.json';records=json.loads(checkpoint.read_text()) if args.resume and checkpoint.exists() else []
    # Checkpoint reuse is only safe with identical job design and integration settings.
    config=out/'p25_config.json';current=dict(p23_source=str(Path(args.p23_report).resolve()),p24_source=str(Path(args.p24_report).resolve()),target=target['job_id'],settings={k:v for k,v in vars(args).items() if k not in ('resume','output_dir')})
    if args.resume and config.exists() and json.loads(config.read_text())!=current:raise ValueError('Resume settings differ from saved configuration')
    config.write_text(json.dumps(current,indent=2)+'\n')
    done={r['job_id'] for r in records}
    for i,(job_id,scale,angle,rep,direction) in enumerate(jobs,1):
        if job_id in done:continue
        rec=dict(job_id=job_id,scale=scale,angle_degrees=angle,angular_replicate=rep,direction=direction.tolist(),source_cluster=target['source_cluster'],target_cluster=target['final_class'])
        try:
            x=x0+scale*direction
            if args.transient:
                _,Y=integrate(x,p,args.transient,args.dt,args.rtol,args.atol,args.max_abs);x=Y[-1]
            t,Y=integrate(x,p,args.observe,args.dt,args.rtol,args.atol,args.max_abs)
            ws=[]
            for j in range(args.windows):
                lo=int(j*(len(t)-1)/args.windows);hi=int((j+1)*(len(t)-1)/args.windows)+1
                ws.append(dict(window=j,**classify(t[lo:hi],Y[lo:hi],z,refs,args.drift_tolerance)))
            late=[w['classification'] for w in ws[-4:]]
            verdict=late[0] if len(set(late))==1 and late[0]!='unclassified' else 'ambiguous'
            rec.update(status='complete',windows=ws,verdict=verdict,all_window_classes=[w['classification'] for w in ws],final_state=Y[-1].tolist())
        except Exception as e:rec.update(status='failed',error=str(e),verdict='failed')
        records.append(rec);done.add(job_id)
        tmp=out/'p25_checkpoint.tmp';tmp.write_text(json.dumps(records,indent=2,allow_nan=False));tmp.replace(checkpoint)
        print(f'{i}/{len(jobs)} {job_id}: {rec["verdict"]}',flush=True)
    ray=sorted((r for r in records if r['job_id'].startswith('ray_') and r['status']=='complete'),key=lambda r:r['scale'])
    brackets=[]
    for a,b in zip(ray,ray[1:]):
        if a['verdict']!=b['verdict'] and a['verdict']!='ambiguous' and b['verdict']!='ambiguous':brackets.append(dict(lower=a['scale'],upper=b['scale'],lower_class=a['verdict'],upper_class=b['verdict']))
    # Multiple alternating classes indicate nonmonotonic directional outcome: no unique threshold.
    report=dict(scope='P25 finite-time targeted local directional regime-switch map',source_p24_target=target['job_id'],source_p23_seed=src['seed_id'],zeta=z,reference_drifts=refs,settings=vars(args),n_jobs=len(jobs),n_complete=sum(r['status']=='complete' for r in records),ray_brackets=brackets,ray_verdicts=[dict(scale=r['scale'],verdict=r['verdict']) for r in ray],trajectories=records,limitations=['Finite-time drift classification, not a rigorous basin boundary or proof of attraction.','Different outcomes along one ray need not be monotone; report all observed brackets.','Only one P24 switching source state and a finite number of nearby directions are sampled.','Normalized Harb benchmark; not physical PLL K0 calibration.'])
    (out/'p25_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'p25_summary.csv').open('w',newline='') as f:
        fields=['job_id','scale','angle_degrees','angular_replicate','status','verdict','first_window_class','last_window_class','last_drift1','last_drift2']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            ws=r.get('windows',[]);w.writerow(dict(job_id=r['job_id'],scale=r['scale'],angle_degrees=r['angle_degrees'],angular_replicate=r['angular_replicate'],status=r['status'],verdict=r['verdict'],first_window_class=ws[0]['classification'] if ws else '',last_window_class=ws[-1]['classification'] if ws else '',last_drift1=ws[-1]['drift'][0] if ws else '',last_drift2=ws[-1]['drift'][1] if ws else ''))
    with (out/'p25_windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['job_id','scale','angle_degrees','angular_replicate','window','drift1','drift2','classification']);w.writeheader()
        for r in records:
            for v in r.get('windows',[]):w.writerow(dict(job_id=r['job_id'],scale=r['scale'],angle_degrees=r['angle_degrees'],angular_replicate=r['angular_replicate'],window=v['window'],drift1=v['drift'][0],drift2=v['drift'][1],classification=v['classification']))
    print(f'Complete {report["n_complete"]}/{len(jobs)}; ray class-change brackets: {brackets}');print('Saved',out/'p25_report.json')
if __name__=='__main__':main()

