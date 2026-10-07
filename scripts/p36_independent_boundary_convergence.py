#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p36_independent_boundary_convergence.py

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
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p32_boundary_continuation import simulate, digest, save, csvout

def readcsv(p):
    with open(p,newline='') as f: return list(csv.DictReader(f))
def yes(v): return str(v).lower()=='true'
def midpoint(r): return (float(r['fine_angle_a'])+float(r['fine_angle_b']))/2

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    defaults={'p23_report':'results/author_longtime_verification/p23_report.json',
      'p24_report':'results/author_basin_attraction/p24_report.json',
      'p27_report':'results/author_local_boundary_atlas/p27_report.json',
      'p32_report':'results/author_boundary_continuation/p32_report.json',
      'p34_report':'results/author_geometry_validation/p34_report.json',
      'p34_tests':'results/author_geometry_validation/p34_tests.csv',
      'p34_summary':'results/author_geometry_validation/p34_summary.csv',
      'p34_verifications':'results/author_geometry_validation/p34_verifications.csv',
      'p35_report':'results/author_boundary_resolution/p35_report.json',
      'p35_refinements':'results/author_boundary_resolution/p35_refinements.csv',
      'p35_verifications':'results/author_boundary_resolution/p35_verifications.csv',
      'p35_slopes':'results/author_boundary_resolution/p35_slopes.csv',
      'p35_summary':'results/author_boundary_resolution/p35_summary.csv'}
    for k,v in defaults.items():ap.add_argument('--'+k.replace('_','-'),default=v)
    ap.add_argument('--output-dir',default='results/author_independent_boundary_convergence')
    ap.add_argument('--only-edge',type=int,choices=[30,34]);ap.add_argument('--gaps-per-map',type=int,default=3)
    ap.add_argument('--scan-halfwidth',type=float,default=.04,help='Degrees each side of P35 linear prediction')
    ap.add_argument('--scan-step',type=float,default=.01,help='Maximum angular scan spacing, degrees')
    ap.add_argument('--levels',type=int,default=7,help='Bisections after independent scan')
    ap.add_argument('--long-pairs-per-map',type=int,default=3)
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
    ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9);ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--verify-observe',type=float,default=3200.);ap.add_argument('--verify-dt',type=float,default=.25)
    ap.add_argument('--verify-rtol',type=float,default=1e-11);ap.add_argument('--verify-atol',type=float,default=1e-13)
    ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if a.gaps_per_map<1 or a.gaps_per_map>3 or a.scan_halfwidth<=0 or a.scan_step<=0 or a.levels<0 or a.long_pairs_per_map<0 or a.windows<4 or any(v<=0 for v in (a.observe,a.dt,a.rtol,a.atol,a.verify_observe,a.verify_dt,a.verify_rtol,a.verify_atol)):
        ap.error('Invalid settings')
    paths={k:Path(getattr(a,k)) for k in defaults}
    src={k:json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report','p34_report','p35_report')}
    p23,p24,p27,p32,p34,p35=(src[k] for k in ('p23_report','p24_report','p27_report','p32_report','p34_report','p35_report'))
    refs=p35['reference_drifts'];z=float(p35['zeta'])
    if any(float(src[k]['zeta'])!=z or src[k]['reference_drifts']!=refs for k in ('p27_report','p32_report','p34_report')):raise ValueError('Upstream zeta/reference mismatch')
    if p35['n_failed'] or any(int(r['n_unresolved']) or int(r['n_failed']) for r in readcsv(paths['p35_summary'])):raise ValueError('P35 incomplete/failed')
    for k in ('p23_report','p24_report','p27_report','p32_report','p34_report','p34_tests','p34_summary','p34_verifications'):
        if p35['source_sha256'][k]!=digest(paths[k]):raise ValueError('P35 source hash mismatch: '+k)
    for k in ('p23_report','p24_report','p27_report','p32_report'):
        if p34['source_sha256'][k]!=digest(paths[k]):raise ValueError('P34 source hash mismatch: '+k)
    rr=readcsv(paths['p35_refinements']);vv=readcsv(paths['p35_verifications']);ss=readcsv(paths['p35_slopes'])
    if len(rr)!=len(p35['refinements']) or len(vv)!=len(p35['verifications']) or len(ss)!=len(p35['slopes']):raise ValueError('P35 CSV/report length mismatch')
    for csvrows,jsonrows,keys in ((rr,p35['refinements'],('edge_index','test_index','scale','fine_angle_a','fine_angle_b')),(vv,p35['verifications'],('edge_index','test_index','angle_degrees')),(ss,p35['slopes'],('edge_index','scale_left','scale_right','p35_slope'))):
        for c,j in zip(csvrows,jsonrows):
            for k in keys:
                if abs(float(c[k])-float(j[k]))>1e-10:raise ValueError('P35 CSV/report mismatch: '+k)
    if any(not yes(r['reproduced']) for r in vv):raise ValueError('P35 endpoint verification not reproduced')
    target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
    seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('Base direction mismatch')
    perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
    directions={int(r['edge_index']):int(r['direction_index']) for r in p32['rows']}
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    config={'source_sha256':{k:digest(v) for k,v in paths.items()},'settings':{k:v for k,v in vars(a).items() if k not in ('resume','output_dir')}}
    cp=out/'p36_checkpoint.json';cf=out/'p36_config.json'
    if a.resume and cp.exists() and (not cf.exists() or json.loads(cf.read_text())!=config):raise ValueError('Resume source/settings mismatch')
    if not a.resume and cp.exists():raise ValueError('Checkpoint exists: use --resume or new output dir')
    save(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];byjob={r['job_id']:r for r in records}
    def run(e,s,angle,mode):
        jid=f'e{e}_s{s:.17g}_a{angle:.17g}_{mode}'
        if jid in byjob:return byjob[jid]
        r=dict(job_id=jid,edge_index=e,scale=float(s),angle_degrees=float(angle),mode=mode,status='failed',verdict='failed')
        try:r.update(simulate(x0,p,base,perps,directions[e],s,angle,refs,z,a,'long' if mode=='long' else 'grid'))
        except Exception as exc:r['error']=repr(exc)
        records.append(r);byjob[jid]=r;save(cp,records)
        print(f'{len(records):04d} e={e} s={s:.12g} angle={angle:.10g} {mode}: {r["verdict"]}',flush=True)
        return r
    rows=[];crossings=[];checks=[];summaries=[]
    def publish():
        report=dict(scope='P36 independent interleaved-radial finite-time boundary-convergence test',zeta=z,reference_drifts=refs,source_sha256=config['source_sha256'],settings=vars(a),rows=rows,crossings=crossings,verifications=checks,summary=summaries,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Only three interleaved radial positions per map by default; unobserved subgrid crossings remain possible.','Local angular scans may miss crossings outside the scanned interval.','Interpolation interval bounds reflect P35 angular bracket uncertainty only, not interpolation-model error.','Finite-time drift classifications on two-dimensional initial-condition slices do not prove a smooth global basin boundary or asymptotic attraction.'])
        save(out/'p36_report.json',report)
        csvout(out/'p36_rows.csv',rows,['edge_index','gap_index','scale','prediction_degrees','prediction_lower','prediction_upper','scan_min','scan_max','n_scan_samples','n_crossings','n_ambiguous','status'])
        csvout(out/'p36_crossings.csv',crossings,['edge_index','gap_index','scale','prediction_degrees','prediction_lower','prediction_upper','angle_a','angle_b','class_a','class_b','width_degrees','n_levels','status','midpoint_degrees','prediction_error_degrees','abs_prediction_error_degrees','inside_prediction_interval','p35_gap_scale_left','p35_gap_scale_right'])
        csvout(out/'p36_verifications.csv',checks,['edge_index','gap_index','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
        csvout(out/'p36_summary.csv',summaries,['edge_index','n_rows','n_single_crossing','n_no_crossing','n_multiple','n_ambiguous_rows','n_refined_crossings','median_abs_prediction_error_degrees','max_abs_prediction_error_degrees','n_inside_prediction_interval','n_long_checks','n_long_reproduced','n_complete','n_failed'])
    for e in (30,34):
        if a.only_edge is not None and e!=a.only_edge:continue
        anchors=sorted((r for r in rr if int(r['edge_index'])==e and r['status']=='refined'),key=lambda r:float(r['scale']))
        if len(anchors)<2:raise ValueError('Insufficient P35 anchors for edge '+str(e))
        gap_ids=sorted(set(int(round(v)) for v in np.linspace(0,len(anchors)-2,min(a.gaps_per_map,len(anchors)-1))))
        edgecross=[]
        for gi in gap_ids:
            left,right=anchors[gi],anchors[gi+1];sl=float(left['scale']);sr=float(right['scale']);s=(sl+sr)/2
            t=(s-sl)/(sr-sl);pred=(1-t)*midpoint(left)+t*midpoint(right)
            pl=(1-t)*float(left['fine_angle_a'])+t*float(right['fine_angle_a'])
            pu=(1-t)*float(left['fine_angle_b'])+t*float(right['fine_angle_b'])
            ca=left['class_a'];cb=left['class_b']
            if (ca,cb)!=(right['class_a'],right['class_b']):raise ValueError('P35 crossing orientation mismatch')
            n=max(2,int(math.ceil(2*a.scan_halfwidth/a.scan_step))+1)
            angles=np.linspace(pred-a.scan_halfwidth,pred+a.scan_halfwidth,n)
            samples=[dict(angle=float(ang),verdict=run(e,s,float(ang),'scan')['verdict']) for ang in angles]
            pairs=[]
            for l,r in zip(samples,samples[1:]):
                if l['verdict'] in refs and r['verdict'] in refs and l['verdict']!=r['verdict']:pairs.append((l,r))
            namb=sum(v['verdict'] not in refs for v in samples)
            status='single_crossing' if len(pairs)==1 and not namb else ('no_crossing' if not pairs and not namb else ('multiple_crossings' if len(pairs)>1 and not namb else 'ambiguous_scan'))
            row=dict(edge_index=e,gap_index=gi,scale=s,prediction_degrees=pred,prediction_lower=pl,prediction_upper=pu,scan_min=float(angles[0]),scan_max=float(angles[-1]),n_scan_samples=len(samples),n_crossings=len(pairs),n_ambiguous=namb,status=status,samples=samples)
            rows.append(row);publish()
            for l,r in pairs:
                lo,hi=l.copy(),r.copy();done=0;cs='refined'
                for _ in range(a.levels):
                    mid=(lo['angle']+hi['angle'])/2;v=run(e,s,mid,'refine')['verdict']
                    if v==lo['verdict']:lo=dict(angle=mid,verdict=v)
                    elif v==hi['verdict']:hi=dict(angle=mid,verdict=v)
                    else:cs='ambiguous_midpoint';break
                    done+=1
                m=(lo['angle']+hi['angle'])/2
                c=dict(edge_index=e,gap_index=gi,scale=s,prediction_degrees=pred,prediction_lower=pl,prediction_upper=pu,angle_a=lo['angle'],angle_b=hi['angle'],class_a=lo['verdict'],class_b=hi['verdict'],width_degrees=hi['angle']-lo['angle'],n_levels=done,status=cs,midpoint_degrees=m,prediction_error_degrees=m-pred,abs_prediction_error_degrees=abs(m-pred),inside_prediction_interval=pl<=m<=pu,p35_gap_scale_left=sl,p35_gap_scale_right=sr,endpoints=[lo,hi])
                crossings.append({k:v for k,v in c.items() if k!='endpoints'});edgecross.append(c);publish()
        valid=[c for c in edgecross if c['status']=='refined']
        selected=valid if len(valid)<=a.long_pairs_per_map else [valid[i] for i in sorted(set(int(round(v)) for v in np.linspace(0,len(valid)-1,a.long_pairs_per_map)))] if a.long_pairs_per_map else []
        for c in selected:
            for side,pt in zip(('a','b'),c['endpoints']):
                r=run(e,c['scale'],pt['angle'],'long')
                checks.append(dict(edge_index=e,gap_index=c['gap_index'],side=side,scale=c['scale'],angle_degrees=pt['angle'],expected_class=pt['verdict'],observed_class=r['verdict'],reproduced=r['status']=='complete' and r['verdict']==pt['verdict'],all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
                publish()
        er=[r for r in rows if r['edge_index']==e];ec=[c for c in crossings if c['edge_index']==e];ev=[v for v in checks if v['edge_index']==e];et=[r for r in records if r['edge_index']==e]
        summaries.append(dict(edge_index=e,n_rows=len(er),n_single_crossing=sum(r['status']=='single_crossing' for r in er),n_no_crossing=sum(r['status']=='no_crossing' for r in er),n_multiple=sum(r['status']=='multiple_crossings' for r in er),n_ambiguous_rows=sum(r['status']=='ambiguous_scan' for r in er),n_refined_crossings=sum(c['status']=='refined' for c in ec),median_abs_prediction_error_degrees=float(np.median([c['abs_prediction_error_degrees'] for c in ec])) if ec else None,max_abs_prediction_error_degrees=max((c['abs_prediction_error_degrees'] for c in ec),default=None),n_inside_prediction_interval=sum(c['inside_prediction_interval'] for c in ec),n_long_checks=len(ev),n_long_reproduced=sum(v['reproduced'] for v in ev),n_complete=sum(r['status']=='complete' for r in et),n_failed=sum(r['status']!='complete' for r in et)))
        publish()
    print(f'P36 complete: {len(rows)} rows, {len(crossings)} crossings, {len(checks)} longer checks, {len(records)} integrations; failed={sum(r["status"]!="complete" for r in records)}',flush=True)
if __name__=='__main__':main()

