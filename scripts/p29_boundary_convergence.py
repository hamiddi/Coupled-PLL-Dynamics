#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p29_boundary_convergence.py

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
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate, classify

def save(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def write_csv(path, rows, fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def simulate(x0,p,scale,direction,refs,z,transient,observe,windows,dt,rtol,atol,tol,max_abs):
    x=x0+scale*direction
    if transient: x=integrate(x,p,transient,dt,rtol,atol,max_abs)[1][-1]
    t,Y=integrate(x,p,observe,dt,rtol,atol,max_abs)
    ws=[]
    for i in range(windows):
        lo=int(i*(len(t)-1)/windows); hi=int((i+1)*(len(t)-1)/windows)+1
        ws.append(dict(window=i,t_start=float(t[lo]),t_end=float(t[hi-1]),**classify(t[lo:hi],Y[lo:hi],z,refs,tol)))
    classes=[w['classification'] for w in ws]
    verdict=classes[-1] if len(set(classes[-4:]))==1 and classes[-1]!='unclassified' else 'ambiguous'
    return dict(status='complete',verdict=verdict,all_windows_consistent=len(set(classes))==1,windows=ws,final_state=Y[-1].tolist())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
    ap.add_argument('--p24-report',default='results/author_basin_attraction/p24_report.json')
    ap.add_argument('--p27-report',default='results/author_local_boundary_atlas/p27_report.json')
    ap.add_argument('--p28-report',default='results/author_adaptive_angular_refinement/p28_report.json')
    ap.add_argument('--output-dir',default='results/author_boundary_convergence')
    ap.add_argument('--edges-per-group',type=int,default=2,help='Select this many intervals in each bracket/axis group, spaced across scales')
    ap.add_argument('--samples',type=int,default=17,help='Odd number >=9, with P28 endpoints exactly represented')
    ap.add_argument('--margin-widths',type=float,default=.5,help='Extend each interval by this fraction of its width on each side')
    ap.add_argument('--transient',type=float,default=400)
    ap.add_argument('--observe',type=float,default=1600)
    ap.add_argument('--windows',type=int,default=8)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--verify-observes',type=float,nargs='+',default=[3200,6400])
    ap.add_argument('--verify-dt',type=float,default=.25)
    ap.add_argument('--verify-rtol',type=float,default=1e-11)
    ap.add_argument('--verify-atol',type=float,default=1e-13)
    ap.add_argument('--drift-tolerance',type=float,default=.08)
    ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.edges_per_group<1 or args.samples<9 or args.samples%2!=1 or args.margin_widths<0 or args.windows<4 or any(v<=0 for v in [args.dt,args.verify_dt,args.observe,*args.verify_observes]):ap.error('invalid parameters')
    paths={k:Path(getattr(args,k)) for k in ('p23_report','p24_report','p27_report','p28_report')}
    src={k:json.loads(v.read_text()) for k,v in paths.items()}
    p23,p24,p27,p28=(src[k] for k in paths)
    z=float(p28['zeta'])
    if abs(z-float(p27['zeta']))>1e-12:raise ValueError('P27/P28 zeta mismatch')
    target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
    seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(seed['final_state'],float)
    base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('P24/P27 direction mismatch')
    perps=[np.asarray(a,float) for a in p27['perpendicular_axes']]
    refs=dict(p27['reference_drifts'])
    if refs!=p28['reference_drifts']:raise ValueError('P27/P28 reference drift mismatch')
    comp=next(c for c in p23['comparisons'] if abs(c['zeta']-z)<1e-10)
    p23refs=dict(zip(comp['clusters'],comp['mean_detector_drifts']))
    if set(refs)!=set(p23refs) or any(not np.allclose(refs[k],p23refs[k],atol=1e-10) for k in refs):raise ValueError('P23 drift references mismatch')
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    groups={}
    for edge in p28['refined_edges']:
        if edge['locally_resolved'] and edge['class_left']!=edge['class_right']:
            groups.setdefault((edge['bracket_index'],edge['direction_index']),[]).append(edge)
    selected=[]
    for group,es in sorted(groups.items()):
        es.sort(key=lambda e:(e['scale'],e['angle_left'],e['edge_index']))
        n=min(args.edges_per_group,len(es))
        indices=sorted(set(int(round(i*(len(es)-1)/max(n-1,1))) for i in range(n))) if n>1 else [len(es)//2]
        selected.extend(es[i] for i in indices)
    if not selected:raise ValueError('No eligible P28 refined intervals')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(source_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},settings={k:v for k,v in vars(args).items() if k not in ('resume','output_dir')},selected_edges=[e['edge_index'] for e in selected])
    config=out/'p29_config.json';cp=out/'p29_checkpoint.json'
    if args.resume and cp.exists() and not config.exists():raise ValueError('checkpoint without config')
    if args.resume and config.exists() and json.loads(config.read_text())!=cfg:raise ValueError('sources or settings changed')
    if not args.resume and cp.exists():raise ValueError('checkpoint exists; use --resume or a new output directory')
    save(config,cfg)
    records=json.loads(cp.read_text()) if args.resume and cp.exists() else []
    lookup={r['job_id']:r for r in records}
    def get(job_id,e,angle,mode,observe,dt,rtol,atol):
        if job_id in lookup:return lookup[job_id]
        bi=int(e['bracket_index']);di=int(e['direction_index']);scale=float(e['scale'])
        direction=np.cos(np.deg2rad(angle))*base+np.sin(np.deg2rad(angle))*perps[di]
        rec=dict(job_id=job_id,edge_index=e['edge_index'],bracket_index=bi,direction_index=di,scale=scale,angle_degrees=float(angle),mode=mode,observe=float(observe),dt=float(dt),rtol=float(rtol),atol=float(atol),status='failed',verdict='failed')
        try:rec.update(simulate(x0,p,scale,direction,refs,z,args.transient,observe,args.windows,dt,rtol,atol,args.drift_tolerance,args.max_abs))
        except Exception as ex:rec['error']=str(ex)
        records.append(rec);lookup[job_id]=rec;save(cp,records)
        print(f'{len(records)} {job_id} angle={angle:.9f} verdict={rec["verdict"]}',flush=True)
        return rec
    dense=[];crossings=[];stability=[]
    for e in selected:
        ei=int(e['edge_index']);left=float(e['angle_left']);right=float(e['angle_right']);w=right-left
        lo=left-args.margin_widths*w;hi=right+args.margin_widths*w
        angles=np.linspace(lo,hi,args.samples)
        # Preserve the P28 endpoint positions in the dense grid when margin=.5 and samples=17.
        rows=[]
        for i,angle in enumerate(angles):
            r=get(f'e{ei:03d}_grid_{i:03d}',e,float(angle),'dense',args.observe,args.dt,args.rtol,args.atol)
            rows.append(r)
            dense.append(dict(edge_index=ei,bracket_index=e['bracket_index'],direction_index=e['direction_index'],scale=e['scale'],sample=i,angle_degrees=float(angle),verdict=r['verdict'],status=r['status'],all_windows_consistent=r.get('all_windows_consistent',False),job_id=r['job_id']))
        for a,b in zip(rows[:-1],rows[1:]):
            if a['verdict'] in refs and b['verdict'] in refs and a['verdict']!=b['verdict']:
                crossings.append(dict(edge_index=ei,bracket_index=e['bracket_index'],direction_index=e['direction_index'],scale=e['scale'],angle_left=a['angle_degrees'],angle_right=b['angle_degrees'],width_degrees=b['angle_degrees']-a['angle_degrees'],class_left=a['verdict'],class_right=b['verdict']))
        for side,angle,expected in [('left',left,e['class_left']),('right',right,e['class_right'])]:
            # Independent of dense-grid rounding: use exact P28 endpoint angle.
            for vi,ob in enumerate(args.verify_observes):
                r=get(f'e{ei:03d}_{side}_long_{vi:02d}',e,angle,'long',ob,args.verify_dt,args.verify_rtol,args.verify_atol)
                stability.append(dict(edge_index=ei,bracket_index=e['bracket_index'],direction_index=e['direction_index'],side=side,angle_degrees=angle,expected_class=expected,observe=ob,observed_class=r['verdict'],reproduced=r['verdict']==expected,all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
    per_edge=[]
    for e in selected:
        ei=e['edge_index'];rr=[r for r in dense if r['edge_index']==ei];cc=[c for c in crossings if c['edge_index']==ei];vv=[v for v in stability if v['edge_index']==ei]
        per_edge.append(dict(edge_index=ei,bracket_index=e['bracket_index'],direction_index=e['direction_index'],scale=e['scale'],p28_left=e['angle_left'],p28_right=e['angle_right'],n_dense=len(rr),n_adjacent_class_changes=len(cc),n_ambiguous=sum(r['verdict'] not in refs for r in rr),n_long=len(vv),n_long_reproduced=sum(v['reproduced'] for v in vv),all_long_reproduced=all(v['reproduced'] for v in vv)))
    report=dict(scope='P29 dense finite-time angular sampling and longer numerical convergence',zeta=z,reference_drifts=refs,selected_p28_edge_indices=[e['edge_index'] for e in selected],source_sha256=cfg['source_sha256'],settings=vars(args),per_edge=per_edge,dense_samples=dense,observed_neighbor_changes=crossings,long_verifications=stability,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['All classifications are finite-time at one source state and selected scales.','Grid can miss changes between samples; a single observed change does not imply a unique boundary.','Long-time agreement is numerical evidence, not a proof of asymptotic attraction or fractal geometry.','Unclassified and inconsistent cases are not forced into either reference class.'])
    save(out/'p29_report.json',report)
    write_csv(out/'p29_dense_samples.csv',dense,['edge_index','bracket_index','direction_index','scale','sample','angle_degrees','verdict','status','all_windows_consistent','job_id'])
    write_csv(out/'p29_neighbor_changes.csv',crossings,['edge_index','bracket_index','direction_index','scale','angle_left','angle_right','width_degrees','class_left','class_right'])
    write_csv(out/'p29_long_verifications.csv',stability,['edge_index','bracket_index','direction_index','side','angle_degrees','expected_class','observe','observed_class','reproduced','all_windows_consistent','status','job_id'])
    write_csv(out/'p29_summary.csv',per_edge,['edge_index','bracket_index','direction_index','scale','p28_left','p28_right','n_dense','n_adjacent_class_changes','n_ambiguous','n_long','n_long_reproduced','all_long_reproduced'])
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        for e in selected:
            ei=e['edge_index'];rr=[r for r in dense if r['edge_index']==ei]
            cmap={k:i for i,k in enumerate(sorted(refs))}
            fig,ax=plt.subplots(figsize=(8,2.8));xs=[r['angle_degrees'] for r in rr];ys=[cmap.get(r['verdict'],np.nan) for r in rr]
            ax.plot(xs,ys,'o-',label='finite-time class');ax.axvspan(e['angle_left'],e['angle_right'],alpha=.12,label='P28 bracket')
            ax.set(yticks=list(cmap.values()),yticklabels=[k.split('_')[-1] for k in sorted(cmap)],xlabel='Angle (degrees)',title=f'P29 edge {ei}: bracket {e["bracket_index"]}, axis {e["direction_index"]}, scale {e["scale"]:.8f}')
            ax.legend(loc='best');fig.tight_layout();fig.savefig(out/f'p29_edge_{ei:03d}.png',dpi=200);plt.close(fig)
    except ImportError:print('matplotlib unavailable: skipping plots',flush=True)
    print(f'P29 finished: {len(selected)} edges; {len(records)} simulations; {len(crossings)} sampled neighbor changes; {len(stability)} long verifications; output={out}',flush=True)
if __name__=='__main__':main()

