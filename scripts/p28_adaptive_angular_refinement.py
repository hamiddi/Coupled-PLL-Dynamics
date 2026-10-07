#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p28_adaptive_angular_refinement.py

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
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate,classify

def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def run(x0,p,scale,direction,refs,z,transient,observe,windows,dt,rtol,atol,tol,max_abs):
    x=x0+scale*direction
    if transient:x=integrate(x,p,transient,dt,rtol,atol,max_abs)[1][-1]
    t,Y=integrate(x,p,observe,dt,rtol,atol,max_abs)
    ws=[]
    for w in range(windows):
        lo=int(w*(len(t)-1)/windows);hi=int((w+1)*(len(t)-1)/windows)+1
        ws.append(dict(window=w,t_start=float(t[lo]),t_end=float(t[hi-1]),**classify(t[lo:hi],Y[lo:hi],z,refs,tol)))
    last=[w['classification'] for w in ws[-4:]]
    verdict=last[0] if len(set(last))==1 and last[0]!='unclassified' else 'ambiguous'
    return dict(status='complete',verdict=verdict,all_windows_consistent=len(set(w['classification'] for w in ws))==1,windows=ws,final_state=Y[-1].tolist())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
    ap.add_argument('--p24-report',default='results/author_basin_attraction/p24_report.json')
    ap.add_argument('--p27-report',default='results/author_local_boundary_atlas/p27_report.json')
    ap.add_argument('--output-dir',default='results/author_adaptive_angular_refinement')
    ap.add_argument('--rounds',type=int,default=3,help='Bisection rounds for each observed angular neighbor class change')
    ap.add_argument('--max-edges',type=int,default=0,help='0=all P27 angular edges; otherwise first N sorted edges')
    ap.add_argument('--verify-edges',type=int,default=12,help='Long verification of final two bracket endpoints, stratified across bracket/axis')
    ap.add_argument('--transient',type=float,default=400);ap.add_argument('--observe',type=float,default=1600)
    ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9);ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--verify-observe',type=float,default=3200);ap.add_argument('--verify-dt',type=float,default=.25)
    ap.add_argument('--verify-rtol',type=float,default=1e-11);ap.add_argument('--verify-atol',type=float,default=1e-13)
    ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.rounds<1 or args.max_edges<0 or args.verify_edges<0 or args.windows<4 or min(args.dt,args.verify_dt,args.observe,args.verify_observe)<=0:ap.error('invalid parameters')
    paths={k:Path(getattr(args,k)) for k in ('p23_report','p24_report','p27_report')}
    p23=json.loads(paths['p23_report'].read_text());p24=json.loads(paths['p24_report'].read_text());p27=json.loads(paths['p27_report'].read_text())
    z=float(p27['zeta']);target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
    src=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(src['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('P27 base direction does not match P24')
    perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
    refs=dict(p27['reference_drifts']);comp=next(c for c in p23['comparisons'] if abs(c['zeta']-z)<1e-10)
    p23refs=dict(zip(comp['clusters'],comp['mean_detector_drifts']))
    if set(refs)!=set(p23refs) or any(not np.allclose(refs[k],p23refs[k],atol=1e-10) for k in refs):raise ValueError('P27 drift references disagree with P23')
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    original={r['job_id']:r for r in p27['trajectories']}
    edges=[e for e in p27['neighbor_class_changes'] if e['axis']=='angle']
    edges.sort(key=lambda e:(e['bracket_index'],e['direction_index'],e['scale_a'],e['angle_a']))
    if args.max_edges:edges=edges[:args.max_edges]
    for e in edges:
        a=original[e['job_a']];b=original[e['job_b']]
        if a['verdict']==b['verdict'] or not np.isclose(a['scale'],b['scale'],atol=1e-12):raise ValueError('P27 edge inconsistent')
        if a['status']!='complete' or b['status']!='complete':raise ValueError('Incomplete P27 endpoint')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(sources={k:str(v.resolve()) for k,v in paths.items()},sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},settings={k:v for k,v in vars(args).items() if k not in ('resume','output_dir')},edges=edges)
    config=out/'p28_config.json';cp=out/'p28_checkpoint.json'
    if args.resume and cp.exists() and not config.exists():raise ValueError('checkpoint without config')
    if args.resume and config.exists() and json.loads(config.read_text())!=cfg:raise ValueError('source or configuration changed')
    if not args.resume and cp.exists():raise ValueError('existing checkpoint; use --resume or another output directory')
    atomic(config,cfg);records=json.loads(cp.read_text()) if args.resume and cp.exists() else []
    lookup={r['job_id']:r for r in records}
    def get(jid,bi,di,scale,angle,mode):
        if jid in lookup:return lookup[jid]
        d=np.cos(np.deg2rad(angle))*base+np.sin(np.deg2rad(angle))*perps[di]
        rec=dict(job_id=jid,bracket_index=bi,direction_index=di,scale=float(scale),angle_degrees=float(angle),mode=mode,status='failed',verdict='failed')
        try:
            settings=(args.transient,args.observe,args.windows,args.dt,args.rtol,args.atol) if mode=='refine' else (args.transient,args.verify_observe,args.windows,args.verify_dt,args.verify_rtol,args.verify_atol)
            rec.update(run(x0,p,scale,d,refs,z,*settings,args.drift_tolerance,args.max_abs))
        except Exception as ex:rec['error']=str(ex)
        records.append(rec);lookup[jid]=rec;atomic(cp,records)
        print(f'{len(records)} {jid} scale={scale:.10f} angle={angle:.8f} verdict={rec["verdict"]}',flush=True)
        return rec
    refined=[]
    for ei,e in enumerate(edges):
        left,right=original[e['job_a']],original[e['job_b']]
        al,ar=float(left['angle_degrees']),float(right['angle_degrees'])
        cl,cr=left['verdict'],right['verdict'];bi=int(e['bracket_index']);di=int(e['direction_index']);scale=float(e['scale_a'])
        history=[];resolved=True
        for level in range(args.rounds):
            mid=(al+ar)/2;rec=get(f'e{ei:03d}_r{level:02d}',bi,di,scale,mid,'refine')
            history.append(dict(angle=mid,verdict=rec['verdict'],all_windows_consistent=rec.get('all_windows_consistent',False)))
            if rec['verdict']==cl:al=mid
            elif rec['verdict']==cr:ar=mid
            else:resolved=False;break
        refined.append(dict(edge_index=ei,bracket_index=bi,direction_index=di,scale=scale,initial_angle_left=float(e['angle_a']),initial_angle_right=float(e['angle_b']),class_left=cl,class_right=cr,angle_left=al,angle_right=ar,angular_width_degrees=ar-al,locally_resolved=resolved,history=history))
    # Balanced verification across bracket and perpendicular direction; never infer from refinement alone.
    groups={(bi,di):[] for bi in range(2) for di in range(len(perps))}
    for r in refined:
        if r['locally_resolved']:groups[(r['bracket_index'],r['direction_index'])].append(r)
    selected=[]
    while len(selected)<args.verify_edges and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected)<args.verify_edges:selected.append(groups[key].pop(0))
    verifications=[]
    for r in selected:
        for side in ('left','right'):
            rec=get(f'e{r["edge_index"]:03d}_verify_{side}',r['bracket_index'],r['direction_index'],r['scale'],r['angle_'+side],'verify')
            verifications.append(dict(edge_index=r['edge_index'],side=side,expected_class=r['class_'+side],observed_class=rec['verdict'],reproduced=rec['verdict']==r['class_'+side],all_windows_consistent=rec.get('all_windows_consistent',False),job_id=rec['job_id']))
    report=dict(scope='P28 adaptive finite-time angular refinement and selected long verification',zeta=z,source_p27=p27['scope'],source_p24_target=target['job_id'],source_p23_seed=src['seed_id'],reference_drifts=refs,settings=vars(args),n_p27_angular_edges=len([e for e in p27['neighbor_class_changes'] if e['axis']=='angle']),n_selected_edges=len(edges),n_completed=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),refined_edges=refined,verifications=verifications,trajectories=records,limitations=['Finite-time classifications at one source state and two local scale intervals.','Angular bisection assumes only endpoint disagreement; intermediate alternations may exist.','Locally resolved means sampled endpoint classes differ, not a unique or smooth boundary.','Long verification is limited to selected refined endpoints; not a basin proof.'])
    atomic(out/'p28_report.json',report)
    def write(name,fields,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    write('p28_refined_edges.csv',['edge_index','bracket_index','direction_index','scale','initial_angle_left','initial_angle_right','class_left','class_right','angle_left','angle_right','angular_width_degrees','locally_resolved'],refined)
    write('p28_verifications.csv',['edge_index','side','expected_class','observed_class','reproduced','all_windows_consistent','job_id'],verifications)
    write('p28_summary.csv',['job_id','bracket_index','direction_index','scale','angle_degrees','mode','status','verdict','all_windows_consistent'],records)
    write('p28_windows.csv',['job_id','window','t_start','t_end','drift1','drift2','classification'],[dict(job_id=r['job_id'],window=w['window'],t_start=w['t_start'],t_end=w['t_end'],drift1=w['drift'][0],drift2=w['drift'][1],classification=w['classification']) for r in records for w in r.get('windows',[])])
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        for bi in range(2):
            for di in range(len(perps)):
                subset=[r for r in refined if r['bracket_index']==bi and r['direction_index']==di]
                if not subset:continue
                fig,ax=plt.subplots(figsize=(9,5))
                for r in subset:
                    ax.plot([r['angle_left'],r['angle_right']],[r['scale']]*2,'-',lw=2)
                    ax.scatter([r['angle_left'],r['angle_right']],[r['scale']]*2,s=15)
                ax.set(xlabel='Angle (degrees)',ylabel='Perturbation scale',title=f'P28 finite-time angular brackets: transition {bi}, axis {di}')
                fig.tight_layout();fig.savefig(out/f'p28_brackets_b{bi}_d{di}.png',dpi=200);plt.close(fig)
    except ImportError:print('matplotlib missing: skipped plots',flush=True)
    print(f'P28 finished: {len(edges)} angular edges, {len(records)} trajectories, {len(verifications)} verification runs; output {out}',flush=True)
if __name__=='__main__':main()

