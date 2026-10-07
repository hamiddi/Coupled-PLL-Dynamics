#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p31_adaptive_2d_convergence.py

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

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csvout(path,rows,fields):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def simulate(x0,p,base,perps,di,s,ang,refs,z,a,mode):
    vec=np.cos(np.deg2rad(ang))*base+np.sin(np.deg2rad(ang))*perps[di]
    x=x0+s*vec
    long=mode=='long';tight=mode=='tight'
    dt=a.verify_dt if long or tight else a.dt
    rt=a.verify_rtol if long or tight else a.rtol
    at=a.verify_atol if long or tight else a.atol
    obs=a.verify_observe if long else a.observe
    if a.transient: x=integrate(x,p,a.transient,dt,rt,at,a.max_abs)[1][-1]
    t,Y=integrate(x,p,obs,dt,rt,at,a.max_abs)
    wins=[]
    for i in range(a.windows):
        lo=int(i*(len(t)-1)/a.windows);hi=int((i+1)*(len(t)-1)/a.windows)+1
        wins.append(dict(window=i,t_start=float(t[lo]),t_end=float(t[hi-1]),**classify(t[lo:hi],Y[lo:hi],z,refs,a.drift_tolerance)))
    classes=[w['classification'] for w in wins]
    verdict=classes[-1] if len(set(classes[-4:]))==1 and classes[-1] in refs else 'ambiguous'
    return dict(status='complete',verdict=verdict,all_windows_consistent=len(set(classes))==1,windows=wins,final_state=Y[-1].tolist(),observe=obs,rtol=rt,atol=at,dt=dt)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    defaults={'p23_report':'results/author_longtime_verification/p23_report.json','p24_report':'results/author_basin_attraction/p24_report.json','p27_report':'results/author_local_boundary_atlas/p27_report.json','p28_report':'results/author_adaptive_angular_refinement/p28_report.json','p29_report':'results/author_boundary_convergence/p29_report.json','p30_report':'results/author_local_2d_boundary/p30_report.json'}
    for k,v in defaults.items():ap.add_argument('--'+k.replace('_','-'),default=v)
    ap.add_argument('--output-dir',default='results/author_adaptive_2d_convergence')
    ap.add_argument('--pairs-per-axis',type=int,default=2,help='Per P30 map and axis; default 16 selected adjacent pairs total')
    ap.add_argument('--only-edge',type=int,default=None)
    ap.add_argument('--levels',type=int,default=3,help='Bisection midpoint levels; original bracket / 2**levels')
    ap.add_argument('--transverse-probes',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--verify-pairs-per-axis',type=int,default=1,help='Per map and axis; both final endpoints long-verified')
    ap.add_argument('--tight-midpoints',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--transient',type=float,default=400.)
    ap.add_argument('--observe',type=float,default=1600.)
    ap.add_argument('--windows',type=int,default=8)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--verify-observe',type=float,default=3200.)
    ap.add_argument('--verify-dt',type=float,default=.25)
    ap.add_argument('--verify-rtol',type=float,default=1e-11)
    ap.add_argument('--verify-atol',type=float,default=1e-13)
    ap.add_argument('--drift-tolerance',type=float,default=.08)
    ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if a.pairs_per_axis<1 or a.levels<1 or a.verify_pairs_per_axis<0 or a.windows<4 or any(x<=0 for x in (a.observe,a.dt,a.rtol,a.atol,a.verify_observe,a.verify_dt,a.verify_rtol,a.verify_atol)) :ap.error('Invalid settings')
    paths={k:Path(getattr(a,k)) for k in defaults};src={k:json.loads(v.read_text()) for k,v in paths.items()}
    p23,p24,p27,p28,p29,p30=(src[k] for k in defaults)
    z=float(p30['zeta']);refs=p30['reference_drifts']
    if p30['n_failed'] or p30['n_complete']!=len(p30['trajectories']):raise ValueError('P30 incomplete/failed')
    if p30['reference_drifts']!=p29['reference_drifts'] or p30['reference_drifts']!=p27['reference_drifts']:raise ValueError('drift reference mismatch')
    if any(abs(float(src[k]['zeta'])-z)>1e-12 for k in ('p27_report','p28_report','p29_report')):raise ValueError('zeta mismatch')
    for k in ('p23_report','p24_report','p27_report','p28_report','p29_report'):
        if p30['source_sha256'][k]!=sha(paths[k]):raise ValueError('P30 upstream provenance mismatch: '+k)
    if p29['source_sha256']['p28_report']!=sha(paths['p28_report']):raise ValueError('P29 provenance mismatch')
    target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
    seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('direction mismatch')
    perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    # Select spatially separated P30 changes along each map/axis; never treat neighboring cells as independent trials.
    selected=[]
    for summary in p30['summary']:
        ei=int(summary['edge_index'])
        if a.only_edge is not None and ei!=a.only_edge:continue
        for axis in ('angle','scale'):
            cand=[v for v in p30['neighbor_changes'] if int(v['edge_index'])==ei and v['axis']==axis]
            cand.sort(key=lambda n:((n['scale_a']+n['scale_b'])/2,(n['angle_a']+n['angle_b'])/2))
            if not cand:continue
            n=min(a.pairs_per_axis,len(cand))
            indices=[len(cand)//2] if n==1 else sorted(set(int(round(i*(len(cand)-1)/(n-1))) for i in range(n)))
            for idx in indices:selected.append(dict(cand[idx],selection_index=idx))
    if not selected:raise ValueError('No P30 class changes selected')
    grid={(int(r['edge_index']),r['job_id']):r for r in p30['grid']}
    trajectories={r['job_id']:r for r in p30['trajectories']}
    for n in selected:
        for side in ('a','b'):
            r=trajectories[n['job_'+side]]
            if r['verdict']!=n['class_'+side] or r['status']!='complete':raise ValueError('P30 endpoint mismatch')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(source_sha256={k:sha(v) for k,v in paths.items()},settings={k:v for k,v in vars(a).items() if k not in ('resume','output_dir')},selected=[{k:v for k,v in n.items() if k!='job_a' and k!='job_b'} for n in selected])
    cp=out/'p31_checkpoint.json';conf=out/'p31_config.json'
    if a.resume and cp.exists() and (not conf.exists() or json.loads(conf.read_text())!=cfg):raise ValueError('Resume configuration/source mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint: --resume or choose new output dir')
    save(conf,cfg);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];lookup={r['job_id']:r for r in records}
    def get(jid,n,s,ang,mode):
        if jid in lookup:return lookup[jid]
        r=dict(job_id=jid,edge_index=int(n['edge_index']),bracket_index=int(n['bracket_index']),direction_index=int(n['direction_index']),axis=n['axis'],scale=float(s),angle_degrees=float(ang),mode=mode,status='failed',verdict='failed')
        try:r.update(simulate(x0,p,base,perps,int(n['direction_index']),float(s),float(ang),refs,z,a,mode))
        except Exception as exc:r['error']=repr(exc)
        records.append(r);lookup[jid]=r;save(cp,records)
        print(f'{len(records):4d} {jid} {mode} scale={s:.10g} angle={ang:.8g} {r["verdict"]}',flush=True)
        return r
    refinements=[];probes=[];verifications=[];summaries=[]
    for i,n in enumerate(selected):
        axis=n['axis'];ei=int(n['edge_index']);lo=dict(scale=float(n['scale_a']),angle=float(n['angle_a']),verdict=n['class_a']);hi=dict(scale=float(n['scale_b']),angle=float(n['angle_b']),verdict=n['class_b'])
        if lo['verdict']==hi['verdict']:raise ValueError('Selected P30 pair has no change')
        initial=abs(hi[axis]-lo[axis]);steps=[];status='resolved_on_sampled_line';last_mid=None
        for level in range(1,a.levels+1):
            s=(lo['scale']+hi['scale'])/2;ang=(lo['angle']+hi['angle'])/2
            mid=get(f'pair{i:02d}_level{level:02d}',n,s,ang,'refine');last_mid=mid
            if mid['verdict'] not in refs:
                status='ambiguous_midpoint';steps.append(dict(level=level,scale=s,angle_degrees=ang,verdict=mid['verdict'],status=status,job_id=mid['job_id']));break
            if mid['verdict']==lo['verdict']:lo=dict(scale=s,angle=ang,verdict=mid['verdict'])
            elif mid['verdict']==hi['verdict']:hi=dict(scale=s,angle=ang,verdict=mid['verdict'])
            else:status='new_class';break
            steps.append(dict(level=level,scale=s,angle_degrees=ang,verdict=mid['verdict'],status='bisected',job_id=mid['job_id']))
        if status=='resolved_on_sampled_line' and a.transverse_probes:
            # Two transverse points on opposite sides of the refined midpoint, half one original P30 transverse grid step.
            g=[r for r in p30['grid'] if int(r['edge_index'])==ei];scales=sorted(set(float(r['scale']) for r in g));angles=sorted(set(float(r['angle_degrees']) for r in g))
            trans='scale' if axis=='angle' else 'angle';step=min(np.diff(scales)) if trans=='scale' else min(np.diff(angles))
            s=(lo['scale']+hi['scale'])/2;ang=(lo['angle']+hi['angle'])/2
            for side,sgn in [('minus',-1),('plus',1)]:
                ss=s+sgn*step/2 if trans=='scale' else s;aa=ang+sgn*step/2 if trans=='angle' else ang
                if ss<=0:continue
                r=get(f'pair{i:02d}_trans_{side}',n,ss,aa,'transverse')
                probes.append(dict(pair_index=i,edge_index=ei,axis=axis,transverse_axis=trans,side=side,scale=ss,angle_degrees=aa,verdict=r['verdict'],status=r['status'],job_id=r['job_id']))
        if status=='resolved_on_sampled_line' and a.tight_midpoints:
            s=(lo['scale']+hi['scale'])/2;ang=(lo['angle']+hi['angle'])/2
            # Same midpoint as final bracket, standard and tight settings; do not assume midpoint equals either class.
            std=get(f'pair{i:02d}_final_mid_standard',n,s,ang,'refine')
            tight=get(f'pair{i:02d}_final_mid_tight',n,s,ang,'tight')
            verifications.append(dict(pair_index=i,edge_index=ei,axis=axis,mode='tolerance',side='midpoint',scale=s,angle_degrees=ang,expected_class=std['verdict'],observed_class=tight['verdict'],reproduced=std['verdict']==tight['verdict'] and std['verdict'] in refs,all_windows_consistent=tight.get('all_windows_consistent',False),status=tight['status'],job_id=tight['job_id'],reference_job_id=std['job_id']))
        if status=='resolved_on_sampled_line' and sum(int(v['edge_index'])==ei and v['axis']==axis for v in refinements)<a.verify_pairs_per_axis:
            for side,pt in [('left',lo),('right',hi)]:
                r=get(f'pair{i:02d}_long_{side}',n,pt['scale'],pt['angle'],'long')
                verifications.append(dict(pair_index=i,edge_index=ei,axis=axis,mode='long',side=side,scale=pt['scale'],angle_degrees=pt['angle'],expected_class=pt['verdict'],observed_class=r['verdict'],reproduced=r['verdict']==pt['verdict'],all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
        final=abs(hi[axis]-lo[axis]);refinements.append(dict(pair_index=i,edge_index=ei,bracket_index=int(n['bracket_index']),direction_index=int(n['direction_index']),axis=axis,selection_index=n['selection_index'],original_width=initial,final_width=final,refinement_factor=initial/final if final else None,n_levels_completed=sum(v['status']=='bisected' for v in steps),status=status,scale_a=lo['scale'],scale_b=hi['scale'],angle_a=lo['angle'],angle_b=hi['angle'],class_a=lo['verdict'],class_b=hi['verdict'],steps=steps,source_job_a=n['job_a'],source_job_b=n['job_b']))
        relevant=[v for v in verifications if v['pair_index']==i];summaries.append(dict(pair_index=i,edge_index=ei,axis=axis,status=status,original_width=initial,final_width=final,n_levels_completed=sum(v['status']=='bisected' for v in steps),n_transverse=sum(v['pair_index']==i for v in probes),n_checks=len(relevant),n_reproduced=sum(v['reproduced'] for v in relevant)))
        report=dict(scope='P31 adaptive two-dimensional finite-time class-transition refinement',zeta=z,reference_drifts=refs,source_sha256=cfg['source_sha256'],settings=vars(a),selected_pairs=selected,refinements=refinements,transverse_probes=probes,verifications=verifications,summary=summaries,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Nested bisection brackets finite-time classifications only; unresolved subgrid changes are possible.','Transverse probes are sparse local tests, not full two-dimensional convergence.','Agreement across tolerances and observation windows does not establish asymptotic attraction.','Observed class changes do not prove smoothness, uniqueness, fractality, or dimension of a basin boundary.','This is a two-dimensional slice of six-dimensional initial-condition space.'])
        save(out/'p31_report.json',report)
        csvout(out/'p31_refinements.csv',refinements,['pair_index','edge_index','bracket_index','direction_index','axis','selection_index','original_width','final_width','refinement_factor','n_levels_completed','status','scale_a','scale_b','angle_a','angle_b','class_a','class_b','source_job_a','source_job_b'])
        csvout(out/'p31_transverse_probes.csv',probes,['pair_index','edge_index','axis','transverse_axis','side','scale','angle_degrees','verdict','status','job_id'])
        csvout(out/'p31_verifications.csv',verifications,['pair_index','edge_index','axis','mode','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id','reference_job_id'])
        csvout(out/'p31_summary.csv',summaries,['pair_index','edge_index','axis','status','original_width','final_width','n_levels_completed','n_transverse','n_checks','n_reproduced'])
    print(f'P31 finished: {len(refinements)} pairs, {len(probes)} transverse probes, {len(verifications)} numerical checks, {len(records)} integrations',flush=True)
if __name__=='__main__':main()

