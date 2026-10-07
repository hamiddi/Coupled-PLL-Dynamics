#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p30_local_2d_boundary.py

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


def atomic_json(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def csvout(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def run(x0,p,scale,direction,refs,z,transient,observe,windows,dt,rtol,atol,tol,max_abs):
    x=x0+scale*direction
    if transient:x=integrate(x,p,transient,dt,rtol,atol,max_abs)[1][-1]
    t,Y=integrate(x,p,observe,dt,rtol,atol,max_abs)
    ws=[]
    for i in range(windows):
        lo=int(i*(len(t)-1)/windows);hi=int((i+1)*(len(t)-1)/windows)+1
        ws.append(dict(window=i,t_start=float(t[lo]),t_end=float(t[hi-1]),**classify(t[lo:hi],Y[lo:hi],z,refs,tol)))
    classes=[w['classification'] for w in ws]
    verdict=classes[-1] if len(set(classes[-4:]))==1 and classes[-1] in refs else 'ambiguous'
    return dict(status='complete',verdict=verdict,all_windows_consistent=len(set(classes))==1,windows=ws,final_state=Y[-1].tolist())

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for k,d in [('p23_report','results/author_longtime_verification/p23_report.json'),('p24_report','results/author_basin_attraction/p24_report.json'),('p27_report','results/author_local_boundary_atlas/p27_report.json'),('p28_report','results/author_adaptive_angular_refinement/p28_report.json'),('p29_report','results/author_boundary_convergence/p29_report.json')]:a.add_argument('--'+k.replace('_','-'),default=d)
    a.add_argument('--output-dir',default='results/author_local_2d_boundary')
    a.add_argument('--edges-per-group',type=int,default=1,help='Select one P29 crossing per bracket/axis group by default')
    a.add_argument('--only-edge',type=int,default=None,help='Optional single P29 edge for debugging or targeted follow-up')
    a.add_argument('--scale-points',type=int,default=7,help='Odd number >=3; includes P29 scale')
    a.add_argument('--angle-points',type=int,default=17,help='Odd number >=5; centered on P29 crossing midpoint')
    a.add_argument('--scale-halfwidth-brackets',type=float,default=2.,help='Scale halfwidth in units of P26 bracket width')
    a.add_argument('--angle-halfwidth-degrees',type=float,default=.25)
    a.add_argument('--transient',type=float,default=400.)
    a.add_argument('--observe',type=float,default=1600.)
    a.add_argument('--windows',type=int,default=8)
    a.add_argument('--dt',type=float,default=.5)
    a.add_argument('--rtol',type=float,default=1e-9)
    a.add_argument('--atol',type=float,default=1e-11)
    a.add_argument('--verify-pairs-per-group',type=int,default=2,help='Up to N adjacent opposite-class grid pairs per group')
    a.add_argument('--verify-observe',type=float,default=3200.)
    a.add_argument('--verify-dt',type=float,default=.25)
    a.add_argument('--verify-rtol',type=float,default=1e-11)
    a.add_argument('--verify-atol',type=float,default=1e-13)
    a.add_argument('--drift-tolerance',type=float,default=.08)
    a.add_argument('--max-abs',type=float,default=1e7)
    a.add_argument('--resume',action='store_true')
    args=a.parse_args()
    if (args.edges_per_group<1 or args.scale_points<3 or args.scale_points%2!=1 or args.angle_points<5 or args.angle_points%2!=1 or args.scale_halfwidth_brackets<=0 or args.angle_halfwidth_degrees<=0 or args.windows<4 or args.verify_pairs_per_group<0 or any(v<=0 for v in [args.observe,args.dt,args.verify_observe,args.verify_dt,args.rtol,args.atol,args.verify_rtol,args.verify_atol])):a.error('invalid grid/integration settings')
    paths={k:Path(getattr(args,k)) for k in ['p23_report','p24_report','p27_report','p28_report','p29_report']}
    src={k:json.loads(path.read_text()) for k,path in paths.items()}
    p23,p24,p27,p28,p29=(src[k] for k in paths)
    z=float(p29['zeta']);refs=p29['reference_drifts']
    if any(abs(float(src[k]['zeta'])-z)>1e-12 for k in ('p27_report','p28_report')):raise ValueError('zeta mismatch')
    if refs!=p27['reference_drifts'] or refs!=p28['reference_drifts']:raise ValueError('reference drift mismatch')
    # Require P29's provenance to match exactly the upstream input files.
    for k in ('p23_report','p24_report','p27_report','p28_report'):
        if p29['source_sha256'][k]!=hashlib.sha256(paths[k].read_bytes()).hexdigest():raise ValueError(f'P29 provenance mismatch: {k}')
    target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
    seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('base direction mismatch')
    perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    groups={}
    for c in p29['observed_neighbor_changes']:
        e=next((v for v in p29['per_edge'] if v['edge_index']==c['edge_index']),None)
        if e and e['all_long_reproduced'] and e['n_ambiguous']==0:groups.setdefault((c['bracket_index'],c['direction_index']),[]).append(c)
    selected=[]
    for key,cs in sorted(groups.items()):
        cs.sort(key=lambda c:(c['scale'],c['edge_index']))
        n=min(len(cs),args.edges_per_group)
        indices=[len(cs)//2] if n==1 else sorted(set(int(round(i*(len(cs)-1)/(n-1))) for i in range(n)))
        selected.extend(cs[i] for i in indices)
    if args.only_edge is not None:selected=[c for c in selected if int(c['edge_index'])==args.only_edge]
    if not selected:raise ValueError('No P29 long-verified transitions eligible (check --only-edge)')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg={'source_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},'settings':{k:v for k,v in vars(args).items() if k not in ('resume','output_dir')},'selected_edges':[int(c['edge_index']) for c in selected]}
    conf=out/'p30_config.json';cp=out/'p30_checkpoint.json'
    if args.resume and cp.exists() and (not conf.exists() or json.loads(conf.read_text())!=cfg):raise ValueError('checkpoint config/source mismatch')
    if not args.resume and cp.exists():raise ValueError('checkpoint exists; use --resume or a new output directory')
    atomic_json(conf,cfg)
    records=json.loads(cp.read_text()) if args.resume and cp.exists() else []
    lookup={r['job_id']:r for r in records}
    def get(jid,edge,scale,angle,mode,ob,dt,rtol,atol):
        if jid in lookup:return lookup[jid]
        di=int(edge['direction_index']);direction=np.cos(np.deg2rad(angle))*base+np.sin(np.deg2rad(angle))*perps[di]
        rec=dict(job_id=jid,edge_index=int(edge['edge_index']),bracket_index=int(edge['bracket_index']),direction_index=di,scale=float(scale),angle_degrees=float(angle),mode=mode,observe=float(ob),status='failed',verdict='failed')
        try:rec.update(run(x0,p,scale,direction,refs,z,args.transient,ob,args.windows,dt,rtol,atol,args.drift_tolerance,args.max_abs))
        except Exception as exc:rec['error']=repr(exc)
        records.append(rec);lookup[jid]=rec;atomic_json(cp,records)
        print(f'{len(records):4d} {jid} scale={scale:.9g} angle={angle:.7f} {rec["verdict"]}',flush=True)
        return rec
    grid=[];neighbors=[];long=[];summaries=[]
    for c in selected:
        ei=int(c['edge_index']);bi=int(c['bracket_index']);di=int(c['direction_index']);center=float(c['scale'])
        # P26 bracket widths are preserved by P27's seven scale coordinates.
        # Explicit P26 widths, independently recoverable from the P27 scale grid at zero angle.
        zero=sorted({float(r['scale']) for r in p27['trajectories'] if r['bracket_index']==bi and abs(r['angle_degrees'])<1e-12})
        if len(zero)!=7:raise ValueError(f'Expected seven P27 zero-angle scales for bracket {bi}, got {len(zero)}')
        width=zero[4]-zero[2]
        if width<=0:raise ValueError('invalid scale bracket width')
        scales=np.linspace(center-args.scale_halfwidth_brackets*width,center+args.scale_halfwidth_brackets*width,args.scale_points)
        if scales[0]<=0:raise ValueError('scale grid includes nonpositive values')
        ac=.5*(float(c['angle_left'])+float(c['angle_right']))
        angles=np.linspace(ac-args.angle_halfwidth_degrees,ac+args.angle_halfwidth_degrees,args.angle_points)
        mat=[]
        for si,s in enumerate(scales):
            row=[]
            for ai,angle in enumerate(angles):
                r=get(f'e{ei:03d}_s{si:02d}_a{ai:02d}',c,float(s),float(angle),'grid',args.observe,args.dt,args.rtol,args.atol)
                row.append(r)
                grid.append(dict(edge_index=ei,bracket_index=bi,direction_index=di,scale=float(s),angle_degrees=float(angle),scale_index=si,angle_index=ai,verdict=r['verdict'],status=r['status'],all_windows_consistent=r.get('all_windows_consistent',False),job_id=r['job_id']))
            mat.append(row)
        local=[]
        for si in range(len(scales)):
            for ai in range(len(angles)):
                for sj,aj,axis in [(si,ai+1,'angle'),(si+1,ai,'scale')]:
                    if sj>=len(scales) or aj>=len(angles):continue
                    r1=mat[si][ai];r2=mat[sj][aj]
                    if r1['verdict'] in refs and r2['verdict'] in refs and r1['verdict']!=r2['verdict']:
                        n=dict(edge_index=ei,bracket_index=bi,direction_index=di,axis=axis,scale_a=float(scales[si]),scale_b=float(scales[sj]),angle_a=float(angles[ai]),angle_b=float(angles[aj]),class_a=r1['verdict'],class_b=r2['verdict'],job_a=r1['job_id'],job_b=r2['job_id'])
                        local.append(n);neighbors.append(n)
        # Select angular opposite-class pairs closest to original crossing and original scale.
        candidates=[n for n in local if n['axis']=='angle']
        candidates.sort(key=lambda n:(abs(n['scale_a']-center),abs(.5*(n['angle_a']+n['angle_b'])-ac)))
        # One pair per distinct scale row, when available.
        chosen=[];seen=set()
        for n in candidates:
            if n['scale_a'] in seen:continue
            chosen.append(n);seen.add(n['scale_a'])
            if len(chosen)>=args.verify_pairs_per_group:break
        for pi,n in enumerate(chosen):
            for side in ('a','b'):
                expected=n['class_'+side];s=n['scale_'+side];angle=n['angle_'+side]
                r=get(f'e{ei:03d}_pair{pi:02d}_{side}_long',c,s,angle,'long',args.verify_observe,args.verify_dt,args.verify_rtol,args.verify_atol)
                long.append(dict(edge_index=ei,bracket_index=bi,direction_index=di,pair_index=pi,side=side,scale=s,angle_degrees=angle,expected_class=expected,observed_class=r['verdict'],reproduced=r['verdict']==expected,all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
        gr=[r for r in grid if r['edge_index']==ei];vr=[r for r in long if r['edge_index']==ei]
        summaries.append(dict(edge_index=ei,bracket_index=bi,direction_index=di,center_scale=center,center_angle=ac,scale_bracket_width=width,n_grid=len(gr),n_C00=sum(r['verdict'].endswith('C00') for r in gr),n_C01=sum(r['verdict'].endswith('C01') for r in gr),n_ambiguous=sum(r['verdict'] not in refs for r in gr),n_angular_neighbor_changes=sum(n['axis']=='angle' for n in local),n_radial_neighbor_changes=sum(n['axis']=='scale' for n in local),n_long=len(vr),n_long_reproduced=sum(v['reproduced'] for v in vr)))
        try:
            import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
            cmap={k:i for i,k in enumerate(sorted(refs))};M=np.array([[cmap.get(r['verdict'],np.nan) for r in row] for row in mat])
            fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(M,origin='lower',aspect='auto',interpolation='nearest',extent=[angles[0],angles[-1],scales[0],scales[-1]],vmin=0,vmax=1)
            ax.axvline(ac,color='white',ls=':',lw=.8);ax.axhline(center,color='white',ls=':',lw=.8)
            ax.set(xlabel='Perturbation angle (degrees)',ylabel='Perturbation scale',title=f'P30 edge {ei}: bracket {bi}, axis {di} (finite-time classes)')
            cb=fig.colorbar(im,ax=ax,ticks=[0,1]);cb.ax.set_yticklabels([k.split('_')[-1] for k in sorted(refs)]);fig.tight_layout();fig.savefig(out/f'p30_map_e{ei:03d}.png',dpi=220);plt.close(fig)
        except ImportError:print('matplotlib not installed; skipping maps',flush=True)
        # Publish intermediate products so interrupted runs retain inspectable results.
        report=dict(scope='P30 two-dimensional local finite-time basin-class mapping',zeta=z,reference_drifts=refs,selected_p29_edge_indices=[int(v['edge_index']) for v in selected],source_sha256=cfg['source_sha256'],settings=vars(args),summary=summaries,grid=grid,neighbor_changes=neighbors,long_verifications=long,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Finite-time grid classifications are not rigorous asymptotic basin membership.','Adjacent grid changes do not establish smoothness, uniqueness, fractality, or boundary dimension.','Unobserved subgrid changes and longer-time switching remain possible.','A two-dimensional slice does not characterize the full six-dimensional basin boundary.'])
        atomic_json(out/'p30_report.json',report)
        csvout(out/'p30_grid.csv',grid,['edge_index','bracket_index','direction_index','scale','angle_degrees','scale_index','angle_index','verdict','status','all_windows_consistent','job_id'])
        csvout(out/'p30_neighbor_changes.csv',neighbors,['edge_index','bracket_index','direction_index','axis','scale_a','scale_b','angle_a','angle_b','class_a','class_b','job_a','job_b'])
        csvout(out/'p30_long_verifications.csv',long,['edge_index','bracket_index','direction_index','pair_index','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
        csvout(out/'p30_summary.csv',summaries,['edge_index','bracket_index','direction_index','center_scale','center_angle','scale_bracket_width','n_grid','n_C00','n_C01','n_ambiguous','n_angular_neighbor_changes','n_radial_neighbor_changes','n_long','n_long_reproduced'])
    print(f'P30 finished: {len(selected)} maps, {len(grid)} grid points, {len(neighbors)} observed neighbor changes, {len(long)} long checks, {len(records)} integrations',flush=True)
if __name__=='__main__':main()

