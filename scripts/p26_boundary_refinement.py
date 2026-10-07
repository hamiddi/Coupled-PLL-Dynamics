#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p26_boundary_refinement.py

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
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate,classify

def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
    ap.add_argument('--p24-report',default='results/author_basin_attraction/p24_report.json')
    ap.add_argument('--p25-report',default='results/author_boundary_mapping/p25_report.json')
    ap.add_argument('--output-dir',default='results/author_boundary_refinement')
    ap.add_argument('--iterations',type=int,default=8,help='Bisection evaluations per original bracket')
    ap.add_argument('--transient',type=float,default=400)
    ap.add_argument('--observe',type=float,default=1600)
    ap.add_argument('--windows',type=int,default=8)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--drift-tolerance',type=float,default=.08)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--verify-observe',type=float,default=3200)
    ap.add_argument('--verify-dt',type=float,default=.25)
    ap.add_argument('--verify-rtol',type=float,default=1e-11)
    ap.add_argument('--verify-atol',type=float,default=1e-13)
    ap.add_argument('--verify',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.iterations<0 or args.windows<4 or args.observe<=0 or args.dt<=0 or args.verify_observe<=0:ap.error('invalid duration or iteration settings')
    p23=json.loads(Path(args.p23_report).read_text());p24=json.loads(Path(args.p24_report).read_text());p25=json.loads(Path(args.p25_report).read_text())
    target=next((r for r in p24['trajectories'] if r['job_id']==p25['source_p24_target']),None)
    if target is None:raise ValueError('P25 target P24 trajectory missing')
    z=float(target['zeta']);src=next((r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-9 and r['seed_id']==target['source_seed'] and r['source_cluster']==target['source_cluster']),None)
    if src is None:raise ValueError('P23 source missing')
    if p25['source_p23_seed']!=src['seed_id'] or abs(p25['zeta']-z)>1e-9:raise ValueError('P25 source mismatch')
    x0=np.asarray(src['final_state'],float);direction=np.asarray(target['perturb_direction'],float);direction/=np.linalg.norm(direction)
    comp=next(c for c in p23['comparisons'] if abs(c['zeta']-z)<1e-9)
    refs=dict(zip(comp['clusters'],comp['mean_detector_drifts']))
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    original=p25['ray_brackets']
    if len(original)<2:raise ValueError('P25 must contain both observed ray brackets')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    config=dict(sources={k:str(Path(getattr(args,k)).resolve()) for k in ('p23_report','p24_report','p25_report')},target=target['job_id'],settings={k:v for k,v in vars(args).items() if k not in ('resume','output_dir')})
    cfg=out/'p26_config.json';cp=out/'p26_checkpoint.json'
    if args.resume and cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Resume configuration differs')
    if args.resume and cp.exists() and not cfg.exists():raise ValueError('Checkpoint exists without configuration')
    if not args.resume and cp.exists():raise ValueError('Existing checkpoint: choose a new output directory or --resume')
    write_json(cfg,config)
    records=json.loads(cp.read_text()) if args.resume and cp.exists() else []
    lookup={r['job_id']:r for r in records}
    def run(job_id,scale,mode,bracket_index):
        if job_id in lookup:return lookup[job_id]
        verified=mode=='verify'
        dt=args.verify_dt if verified else args.dt
        rtol=args.verify_rtol if verified else args.rtol
        atol=args.verify_atol if verified else args.atol
        observe=args.verify_observe if verified else args.observe
        rec=dict(job_id=job_id,scale=float(scale),mode=mode,bracket_index=bracket_index,settings=dict(transient=args.transient,observe=observe,dt=dt,rtol=rtol,atol=atol))
        try:
            x=x0+float(scale)*direction
            if args.transient:x=integrate(x,p,args.transient,dt,rtol,atol,args.max_abs)[1][-1]
            t,Y=integrate(x,p,observe,dt,rtol,atol,args.max_abs)
            windows=[]
            for j in range(args.windows):
                lo=int(j*(len(t)-1)/args.windows);hi=int((j+1)*(len(t)-1)/args.windows)+1
                windows.append(dict(window=j,t_start=float(t[lo]),t_end=float(t[hi-1]),**classify(t[lo:hi],Y[lo:hi],z,refs,args.drift_tolerance)))
            late=[w['classification'] for w in windows[-4:]]
            verdict=late[0] if len(set(late))==1 and late[0]!='unclassified' else 'ambiguous'
            rec.update(status='complete',verdict=verdict,all_windows_consistent=len(set(w['classification'] for w in windows))==1,windows=windows,final_state=Y[-1].tolist())
        except Exception as e:rec.update(status='failed',verdict='failed',error=str(e))
        records.append(rec);lookup[job_id]=rec;write_json(cp,records)
        print(f'{len(records)} {job_id} scale={scale:.12g} {rec["verdict"]}',flush=True)
        return rec
    refined=[]
    for i,b in enumerate(original):
        lo=float(b['lower']);hi=float(b['upper']);lc=b['lower_class'];hc=b['upper_class']
        if lc==hc:raise ValueError('Invalid original bracket')
        steps=[];valid=True
        for k in range(args.iterations):
            mid=(lo+hi)/2;rec=run(f'bracket{i}_bisect{k:02d}',mid,'refine',i)
            steps.append(dict(scale=mid,verdict=rec['verdict']))
            if rec['verdict']==lc:lo=mid
            elif rec['verdict']==hc:hi=mid
            else:
                valid=False;print(f'Bracket {i} unresolved at {mid:.12g}: {rec["verdict"]}; stop bisection',flush=True);break
        refined.append(dict(bracket_index=i,original=b,lower=lo,upper=hi,lower_class=lc,upper_class=hc,width=hi-lo,locally_resolved=valid,steps=steps))
    verification=[]
    if args.verify:
        for b in refined:
            i=b['bracket_index'];lo=b['lower'];hi=b['upper'];width=hi-lo
            # Re-evaluate both sides, and two interior points of the original intervals.
            # For the final narrow bracket, adjacent interior points test tolerance sensitivity.
            for label,s in [('below',max(0,lo-width/2)),('lower',lo),('upper',hi),('above',hi+width/2)]:
                r=run(f'bracket{i}_verify_{label}',s,'verify',i)
                verification.append(dict(bracket_index=i,position=label,scale=s,verdict=r['verdict'],all_windows_consistent=r.get('all_windows_consistent')))
        for label,s in [('unperturbed',0.),('p24_switch',.1),('p25_return',.2)]:
            r=run(f'control_verify_{label}',s,'verify',None)
            verification.append(dict(bracket_index=None,position=label,scale=s,verdict=r['verdict'],all_windows_consistent=r.get('all_windows_consistent')))
    # Explicitly compare standard and verified labels only where standard data are available.
    p25_ray={float(r['scale']):r['verdict'] for r in p25['ray_verdicts']}
    control_agreement=[dict(scale=r['scale'],p25=p25_ray.get(r['scale']),p26=r['verdict'],agree=p25_ray.get(r['scale'])==r['verdict']) for r in verification if r['position'] in ('unperturbed','p24_switch','p25_return')]
    report=dict(scope='P26 two-interval local ray refinement and long-time numerical verification',zeta=z,source_p24_target=target['job_id'],source_p23_seed=src['seed_id'],direction=direction.tolist(),reference_drifts=refs,settings=vars(args),original_brackets=original,refined_brackets=refined,verification=verification,control_agreement=control_agreement,n_complete=sum(r['status']=='complete' for r in records),n_total=len(records),trajectories=records,limitations=['Local sampled class-change intervals along one ray, not a globally unique threshold or rigorous basin boundary.','Bisection assumes locally adjacent classifications; unsampled finer alternations remain possible.','Tighter tolerances and longer observation test robustness but do not prove asymptotic attraction.','Unwrapped detector phases drift; classification uses fitted detector drift, not raw phase-state distance.','Normalized benchmark, not physical PLL calibration.'])
    write_json(out/'p26_report.json',report)
    with (out/'p26_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['job_id','scale','mode','bracket_index','status','verdict','all_windows_consistent','last_drift1','last_drift2']);w.writeheader()
        for r in records:
            ws=r.get('windows',[]);w.writerow({**{k:r.get(k,'') for k in ('job_id','scale','mode','bracket_index','status','verdict','all_windows_consistent')},'last_drift1':ws[-1]['drift'][0] if ws else '', 'last_drift2':ws[-1]['drift'][1] if ws else ''})
    with (out/'p26_windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['job_id','scale','mode','bracket_index','window','t_start','t_end','drift1','drift2','classification']);w.writeheader()
        for r in records:
            for q in r.get('windows',[]):w.writerow(dict(job_id=r['job_id'],scale=r['scale'],mode=r['mode'],bracket_index=r['bracket_index'],window=q['window'],t_start=q['t_start'],t_end=q['t_end'],drift1=q['drift'][0],drift2=q['drift'][1],classification=q['classification']))
    with (out/'p26_brackets.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['bracket_index','lower','upper','width','lower_class','upper_class','locally_resolved']);w.writeheader()
        for b in refined:w.writerow({k:b[k] for k in w.fieldnames})
    print('Refined brackets:',[(b['lower'],b['upper'],b['width']) for b in refined]);print('Completed:',report['n_complete'],'/',report['n_total']);print('Saved',out/'p26_report.json')
if __name__=='__main__':main()

