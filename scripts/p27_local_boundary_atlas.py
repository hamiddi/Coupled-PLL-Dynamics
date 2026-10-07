#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p27_local_boundary_atlas.py

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
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate,classify

def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def parse_floats(s):return [float(v.strip()) for v in s.split(',')]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
    ap.add_argument('--p24-report',default='results/author_basin_attraction/p24_report.json')
    ap.add_argument('--p26-report',default='results/author_boundary_refinement/p26_report.json')
    ap.add_argument('--output-dir',default='results/author_local_boundary_atlas')
    ap.add_argument('--scale-offsets',default='-4,-2,-0.5,0,0.5,2,4',help='Offsets in P26 bracket widths about each midpoint')
    ap.add_argument('--angles',default='-5,-2,0,2,5',help='Signed angular deviations in degrees')
    ap.add_argument('--directions',type=int,default=2,help='Independent seeded perpendicular axes')
    ap.add_argument('--seed',type=int,default=2701)
    ap.add_argument('--transient',type=float,default=400)
    ap.add_argument('--observe',type=float,default=1600)
    ap.add_argument('--windows',type=int,default=8)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--drift-tolerance',type=float,default=.08)
    ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    offsets=parse_floats(args.scale_offsets);angles=parse_floats(args.angles)
    if args.directions<1 or args.windows<4 or args.observe<=0 or args.dt<=0 or len(set(offsets))!=len(offsets) or len(set(angles))!=len(angles) or not all(np.isfinite(offsets+angles)) or any(abs(a)>=90 for a in angles):ap.error('Invalid grid or integration parameters')
    if 0. not in angles:ap.error('Include angle 0 to anchor the original ray')
    p23=json.loads(Path(args.p23_report).read_text());p24=json.loads(Path(args.p24_report).read_text());p26=json.loads(Path(args.p26_report).read_text())
    target=next(r for r in p24['trajectories'] if r['job_id']==p26['source_p24_target'])
    z=float(target['zeta']);source=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p26['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(source['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p26['direction'],atol=1e-12):raise ValueError('P26 direction disagrees with P24 source')
    comp=next(c for c in p23['comparisons'] if abs(c['zeta']-z)<1e-10)
    refs=dict(zip(comp['clusters'],comp['mean_detector_drifts']))
    if set(refs)!=set(p26['reference_drifts']):raise ValueError('P23 and P26 reference labels disagree')
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    brackets=p26['refined_brackets'];rng=np.random.default_rng(args.seed);perps=[]
    for i in range(args.directions):
        for retry in range(100):
            v=rng.standard_normal(6);v-=np.dot(v,base)*base
            for u in perps:v-=np.dot(v,u)*u
            n=np.linalg.norm(v)
            if n>1e-10:perps.append(v/n);break
        else:raise ValueError('Too many independent perpendicular axes for six-dimensional state')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    source_paths={k:str(Path(getattr(args,k)).resolve()) for k in ('p23_report','p24_report','p26_report')}
    source_hashes={k:hashlib.sha256(Path(getattr(args,k)).read_bytes()).hexdigest() for k in ('p23_report','p24_report','p26_report')}
    config=dict(sources=source_paths,source_sha256=source_hashes,settings={k:v for k,v in vars(args).items() if k not in ('resume','output_dir')},perpendicular_axes=[v.tolist() for v in perps])
    cfg=out/'p27_config.json';cp=out/'p27_checkpoint.json'
    if args.resume and cp.exists() and not cfg.exists():raise ValueError('Checkpoint without configuration')
    if args.resume and cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Resume configuration or source file changed')
    if not args.resume and cp.exists():raise ValueError('Checkpoint exists; use --resume or a new output directory')
    atomic_json(cfg,config)
    records=json.loads(cp.read_text()) if args.resume and cp.exists() else []
    lookup={r['job_id']:r for r in records}
    jobs=[]
    for bi,b in enumerate(brackets):
        if not b['locally_resolved']:raise ValueError(f'P26 bracket {bi} was not resolved')
        center=(b['lower']+b['upper'])/2;width=b['width']
        for si,offset in enumerate(offsets):
            scale=center+offset*width
            if scale<0:raise ValueError('Grid includes negative perturbation scale')
            for ai,angle in enumerate(angles):
                axes=[0] if angle==0 else list(range(args.directions))
                for di in axes:
                    rad=np.deg2rad(angle);direction=np.cos(rad)*base+np.sin(rad)*perps[di]
                    jid=f'b{bi}_s{si:02d}_a{ai:02d}_d{di}'
                    jobs.append((jid,bi,si,ai,di,offset,scale,angle,direction))
    for ji,(jid,bi,si,ai,di,offset,scale,angle,direction) in enumerate(jobs,1):
        if jid in lookup:continue
        r=dict(job_id=jid,bracket_index=bi,scale_index=si,angle_index=ai,direction_index=di,scale_offset_widths=offset,scale=scale,angle_degrees=angle,status='failed',verdict='failed')
        try:
            x=x0+scale*direction
            if args.transient:x=integrate(x,p,args.transient,args.dt,args.rtol,args.atol,args.max_abs)[1][-1]
            t,Y=integrate(x,p,args.observe,args.dt,args.rtol,args.atol,args.max_abs)
            ws=[]
            for w in range(args.windows):
                lo=int(w*(len(t)-1)/args.windows);hi=int((w+1)*(len(t)-1)/args.windows)+1
                ws.append(dict(window=w,t_start=float(t[lo]),t_end=float(t[hi-1]),**classify(t[lo:hi],Y[lo:hi],z,refs,args.drift_tolerance)))
            late=[w['classification'] for w in ws[-4:]]
            verdict=late[0] if len(set(late))==1 and late[0]!='unclassified' else 'ambiguous'
            r.update(status='complete',verdict=verdict,all_windows_consistent=len({w['classification'] for w in ws})==1,windows=ws,final_state=Y[-1].tolist())
        except Exception as e:r['error']=str(e)
        records.append(r);lookup[jid]=r;atomic_json(cp,records)
        print(f'{ji}/{len(jobs)} {jid} scale={scale:.10f} angle={angle:g} verdict={r["verdict"]}',flush=True)
    # Grid-neighbor disagreement is an observed finite-resolution class change, not a proven smooth boundary.
    edges=[]
    for bi in range(len(brackets)):
        for di in range(args.directions):
            for si in range(len(offsets)):
                for ai in range(len(angles)):
                    if angles[ai]==0 and di!=0:continue
                    a=lookup.get(f'b{bi}_s{si:02d}_a{ai:02d}_d{di}')
                    if not a:continue
                    for sj,aj,axis in ((si+1,ai,'scale'),(si,ai+1,'angle')):
                        if sj>=len(offsets) or aj>=len(angles):continue
                        b=lookup.get(f'b{bi}_s{sj:02d}_a{aj:02d}_d{di if angles[aj]!=0 else 0}')
                        if b and a['verdict']!=b['verdict'] and a['verdict'] not in ('ambiguous','failed') and b['verdict'] not in ('ambiguous','failed'):
                            edges.append(dict(bracket_index=bi,direction_index=di,axis=axis,job_a=a['job_id'],job_b=b['job_id'],scale_a=a['scale'],scale_b=b['scale'],angle_a=a['angle_degrees'],angle_b=b['angle_degrees'],class_a=a['verdict'],class_b=b['verdict']))
    summary=[]
    for bi in range(len(brackets)):
        for di in range(args.directions):
            rs=[r for r in records if r['bracket_index']==bi and (r['direction_index']==di or (r['angle_degrees']==0 and di>0))]
            summary.append(dict(bracket_index=bi,direction_index=di,n=len(rs),complete=sum(r['status']=='complete' for r in rs),C00=sum(r['verdict']=='z0.899_C00' for r in rs),C01=sum(r['verdict']=='z0.899_C01' for r in rs),ambiguous=sum(r['verdict']=='ambiguous' for r in rs),failed=sum(r['verdict']=='failed' for r in rs),neighbor_class_changes=sum(e['bracket_index']==bi and e['direction_index']==di for e in edges)))
    report=dict(scope='P27 finite-time two-parameter local scale-angle atlas around both P26 ray transitions',zeta=z,source_p24_target=target['job_id'],source_p23_seed=source['seed_id'],reference_drifts=refs,base_direction=base.tolist(),perpendicular_axes=[v.tolist() for v in perps],scale_offsets=offsets,angles=angles,settings=vars(args),n_jobs=len(jobs),n_complete=sum(r['status']=='complete' for r in records),summary=summary,neighbor_class_changes=edges,trajectories=records,limitations=['Finite-time drift classes; not exact basin boundaries or asymptotic attraction.','Neighbor class differences do not establish a smooth or unique boundary.','Only one source point, two randomly oriented perpendicular axes, and two local scale intervals.','No claims of chaos or basin volume; normalized benchmark is not physical calibration.'])
    atomic_json(out/'p27_report.json',report)
    with (out/'p27_summary.csv').open('w',newline='') as f:
        fields=['job_id','bracket_index','direction_index','scale_index','angle_index','scale_offset_widths','scale','angle_degrees','status','verdict','all_windows_consistent','last_drift1','last_drift2'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            ws=r.get('windows',[]);w.writerow({**{k:r.get(k,'') for k in fields[:-2]},'last_drift1':ws[-1]['drift'][0] if ws else '', 'last_drift2':ws[-1]['drift'][1] if ws else ''})
    with (out/'p27_neighbor_changes.csv').open('w',newline='') as f:
        fields=['bracket_index','direction_index','axis','job_a','job_b','scale_a','scale_b','angle_a','angle_b','class_a','class_b'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(edges)
    with (out/'p27_windows.csv').open('w',newline='') as f:
        fields=['job_id','bracket_index','direction_index','window','t_start','t_end','drift1','drift2','classification'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            for q in r.get('windows',[]):w.writerow(dict(job_id=r['job_id'],bracket_index=r['bracket_index'],direction_index=r['direction_index'],window=q['window'],t_start=q['t_start'],t_end=q['t_end'],drift1=q['drift'][0],drift2=q['drift'][1],classification=q['classification']))
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap,BoundaryNorm
        cmap=ListedColormap(['#2563eb','#e5e7eb','#ef4444','#111827']);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
        for bi in range(len(brackets)):
            for di in range(args.directions):
                matrix=np.full((len(angles),len(offsets)),3.)
                for r in records:
                    if r['bracket_index']==bi and (r['direction_index']==di or r['angle_degrees']==0):
                        matrix[r['angle_index'],r['scale_index']]={'z0.899_C00':0,'ambiguous':1,'z0.899_C01':2}.get(r['verdict'],3)
                fig,ax=plt.subplots(figsize=(9,5));ax.imshow(matrix,origin='lower',aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
                ax.set_xticks(range(len(offsets)),[f'{(brackets[bi]["lower"]+brackets[bi]["upper"])/2+v*brackets[bi]["width"]:.7f}' for v in offsets],rotation=45,ha='right')
                ax.set_yticks(range(len(angles)),[f'{v:g}' for v in angles]);ax.set_xlabel('Perturbation scale');ax.set_ylabel('Signed angle (degrees)');ax.set_title(f'P27 bracket {bi}, perpendicular axis {di}: finite-time drift class')
                fig.tight_layout();fig.savefig(out/f'p27_atlas_bracket{bi}_axis{di}.png',dpi=200);plt.close(fig)
    except ImportError:print('matplotlib not installed; skipped atlas PNGs',flush=True)
    print(f'P27 completed {report["n_complete"]}/{len(jobs)}; neighbor class changes={len(edges)}; saved {out}',flush=True)
if __name__=='__main__':main()

