#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p40_local_boundary_stress_test.py

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
import argparse, csv, itertools, json, math
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from p00_model_validation import Parameters
from p32_boundary_continuation import simulate, digest, save, csvout

DEFAULTS = {
'p23_report':'results/author_longtime_verification/p23_report.json',
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
'p35_summary':'results/author_boundary_resolution/p35_summary.csv',
'p36_report':'results/author_independent_boundary_convergence/p36_report.json',
'p36_rows':'results/author_independent_boundary_convergence/p36_rows.csv',
'p36_crossings':'results/author_independent_boundary_convergence/p36_crossings.csv',
'p36_verifications':'results/author_independent_boundary_convergence/p36_verifications.csv',
'p36_summary':'results/author_independent_boundary_convergence/p36_summary.csv',
'p37_report':'results/author_interpolation_validation/p37_report.json',
'p37_predictions':'results/author_interpolation_validation/p37_predictions.csv',
'p37_crossings':'results/author_interpolation_validation/p37_crossings.csv',
'p37_verifications':'results/author_interpolation_validation/p37_verifications.csv',
'p37_summary':'results/author_interpolation_validation/p37_summary.csv',
'p38_report':'results/author_prediction_uncertainty_validation/p38_report.json',
'p38_predictions':'results/author_prediction_uncertainty_validation/p38_predictions.csv',
'p38_crossings':'results/author_prediction_uncertainty_validation/p38_crossings.csv',
'p38_verifications':'results/author_prediction_uncertainty_validation/p38_verifications.csv',
'p38_summary':'results/author_prediction_uncertainty_validation/p38_summary.csv',
'p39_report':'results/author_adaptive_boundary_validation/p39_report.json',
'p39_predictions':'results/author_adaptive_boundary_validation/p39_predictions.csv',
'p39_crossings':'results/author_adaptive_boundary_validation/p39_crossings.csv',
'p39_verifications':'results/author_adaptive_boundary_validation/p39_verifications.csv',
'p39_summary':'results/author_adaptive_boundary_validation/p39_summary.csv',
}
METHODS=('linear','quadratic','pchip')
def readcsv(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def yes(v):return str(v).lower()=='true'
def anchors35(rows,e):
    return sorted([dict(scale=float(r['scale']),low=float(r['fine_angle_a']),high=float(r['fine_angle_b']),source='P35',class_a=r['class_a'],class_b=r['class_b']) for r in rows if int(r['edge_index'])==e and r['status']=='refined'],key=lambda r:r['scale'])
def anchors36(rows,e):
    return sorted([dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P36',class_a=r['class_a'],class_b=r['class_b']) for r in rows if int(r['edge_index'])==e and r['status']=='refined'],key=lambda r:r['scale'])
def estimate(method,points,s,corner=False):
    pts=sorted(points,key=lambda r:r['scale']); xs=np.array([r['scale'] for r in pts]); ys=np.array([r['high' if corner and r.get('use_high',False) else 'low' if corner else 'mid'] for r in pts])
    if method=='linear':
        k=max(0,min(len(xs)-2,int(np.searchsorted(xs,s)-1)));return float(np.interp(s,xs[k:k+2],ys[k:k+2]))
    if method=='quadratic':
        ids=sorted(np.argsort(abs(xs-s))[:3]);xx=xs[ids];yy=ys[ids];return float(np.polynomial.polynomial.polyval(s-np.mean(xx),np.polynomial.polynomial.polyfit(xx-np.mean(xx),yy,2)))
    return float(PchipInterpolator(xs,ys)(s))
def prediction(method,points,s):
    pts=[dict(p,mid=(p['low']+p['high'])/2) for p in points]
    mid=estimate(method,pts,s)
    # Propagate finite angular-bracket uncertainty, not interpolation-model error.
    if method=='linear':
        xs=[p['scale'] for p in pts];k=max(0,min(len(xs)-2,int(np.searchsorted(xs,s)-1)));vary=range(k,k+2)
    elif method=='quadratic':vary=sorted(np.argsort([abs(p['scale']-s) for p in pts])[:3])
    else:
        # PCHIP is local: interval k depends only on the adjacent four knots.
        xs=[p['scale'] for p in pts];k=max(0,min(len(xs)-2,int(np.searchsorted(xs,s)-1)))
        vary=range(max(0,k-1),min(len(xs),k+3))
    if len(vary)>12:raise ValueError('Too many anchor corners for exact interval propagation')
    vals=[]
    for flags in itertools.product((False,True),repeat=len(vary)):
        pp=[dict(p) for p in pts]
        for i,flag in zip(vary,flags):pp[i]['use_high']=flag
        vals.append(estimate(method,pp,s,corner=True))
    return dict(center=mid,numerical_low=min(vals),numerical_high=max(vals))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k,v in DEFAULTS.items():ap.add_argument('--'+k.replace('_','-'),default=v)
    ap.add_argument('--output-dir',default='results/author_local_boundary_stress_test')
    ap.add_argument('--only-edge',type=int,choices=[30,34]);ap.add_argument('--edge30-rows',type=int,default=2);ap.add_argument('--edge34-rows',type=int,default=4)
    ap.add_argument('--scan-margin',type=float,default=.025,help='Degrees beyond min/max calibrated prediction intervals')
    ap.add_argument('--scan-step',type=float,default=.01);ap.add_argument('--levels',type=int,default=7)
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
    ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9);ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--verify-observe',type=float,default=3200.);ap.add_argument('--verify-dt',type=float,default=.25)
    ap.add_argument('--verify-rtol',type=float,default=1e-11);ap.add_argument('--verify-atol',type=float,default=1e-13)
    ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if not 1<=a.edge30_rows<=5 or not 1<=a.edge34_rows<=6 or a.scan_margin<=0 or a.scan_step<=0 or a.levels<0 or a.windows<4 or any(v<=0 for v in (a.observe,a.dt,a.rtol,a.atol,a.verify_observe,a.verify_dt,a.verify_rtol,a.verify_atol)):
        ap.error('Invalid numerical settings')
    paths={k:Path(getattr(a,k)) for k in DEFAULTS}
    src={k:json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report','p34_report','p35_report','p36_report','p37_report','p38_report','p39_report')}
    p23,p24,p27,p32,p34,p35,p36,p37,p38,p39=[src[k] for k in ('p23_report','p24_report','p27_report','p32_report','p34_report','p35_report','p36_report','p37_report','p38_report','p39_report')]
    refs=p35['reference_drifts'];z=float(p35['zeta'])
    if any(abs(float(src[k]['zeta'])-z)>1e-12 or src[k]['reference_drifts']!=refs for k in ('p27_report','p32_report','p34_report','p36_report','p37_report','p38_report','p39_report')):raise ValueError('Source zeta/reference mismatch')
    if p35['n_failed'] or p36['n_failed'] or p37['n_failed'] or p38['n_failed'] or p39['n_failed']:raise ValueError('P35/P36 has failed integrations')
    for report,keys in ((p34,('p23_report','p24_report','p27_report','p32_report')),(p35,('p23_report','p24_report','p27_report','p32_report','p34_report','p34_tests','p34_summary','p34_verifications')),(p36,tuple(p36['source_sha256'])),(p37,tuple(p37['source_sha256'])),(p38,tuple(p38['source_sha256'])),(p39,tuple(p39['source_sha256']))):
        for k in keys:
            if digest(paths[k])!=report['source_sha256'][k]:raise ValueError('Source SHA mismatch '+k)
    r35=readcsv(paths['p35_refinements']);r36=readcsv(paths['p36_crossings']);v35=readcsv(paths['p35_verifications']);v36=readcsv(paths['p36_verifications']);r37=readcsv(paths['p37_crossings']);v37=readcsv(paths['p37_verifications']);pr37=readcsv(paths['p37_predictions']);sm37=readcsv(paths['p37_summary']);r38=readcsv(paths['p38_crossings']);v38=readcsv(paths['p38_verifications']);pr38=readcsv(paths['p38_predictions']);sm38=readcsv(paths['p38_summary']);r39=readcsv(paths['p39_crossings']);v39=readcsv(paths['p39_verifications']);pr39=readcsv(paths['p39_predictions']);sm39=readcsv(paths['p39_summary'])
    if len(r35)!=len(p35['refinements']) or len(r36)!=len(p36['crossings']):raise ValueError('Source CSV/report row count mismatch')
    for c,j in zip(r35,p35['refinements']):
        for k in ('edge_index','scale','fine_angle_a','fine_angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P35 CSV/report mismatch '+k)
    for c,j in zip(r36,p36['crossings']):
        for k in ('edge_index','scale','angle_a','angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P36 CSV/report mismatch '+k)
    if len(r37)!=len(p37['crossings']) or len(pr37)!=len(p37['predictions']) or len(sm37)!=len(p37['summary']):raise ValueError('P37 CSV/report count mismatch')
    for c,j in zip(r37,p37['crossings']):
        for k in ('edge_index','scale','angle_a','angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P37 crossing CSV/report mismatch '+k)
    if len(v39)!=12:raise ValueError('Expected 12 P39 long checks')
    if any(not yes(v['reproduced']) or not yes(v['all_windows_consistent']) for v in v35+v36+v37+v38+v39):raise ValueError('P35/P36 long verification incomplete')
    if any(r['status']!='refined' for r in r35+r36+r37+r38+r39):raise ValueError('Unresolved source crossings')
    if len(r39)!=len(p39['crossings']) or len(pr39)!=len(p39['predictions']) or len(sm39)!=len(p39['summary']):raise ValueError('P39 CSV/report count mismatch')
    for c,j in zip(r39,p39['crossings']):
        for k in ('edge_index','scale','angle_a','angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P39 crossing CSV/report mismatch '+k)
    target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
    seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('Base direction mismatch')
    perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
    directions={int(r['edge_index']):int(r['direction_index']) for r in p32['rows']}
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    config={'source_sha256':{k:digest(v) for k,v in paths.items()},'settings':{k:v for k,v in vars(a).items() if k not in ('resume','output_dir')}}
    cp=out/'p40_checkpoint.json';cf=out/'p40_config.json'
    if a.resume and cp.exists() and (not cf.exists() or json.loads(cf.read_text())!=config):raise ValueError('Resume configuration/source mismatch')
    if not a.resume and cp.exists():raise ValueError('Checkpoint exists: use --resume or new output directory')
    save(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];byjob={r['job_id']:r for r in records}
    def run(e,s,angle,mode):
        jid=f'e{e}_s{s:.17g}_a{angle:.17g}_{mode}'
        if jid in byjob:return byjob[jid]
        r=dict(job_id=jid,edge_index=e,scale=float(s),angle_degrees=float(angle),mode=mode,status='failed',verdict='failed')
        try:r.update(simulate(x0,p,base,perps,directions[e],s,angle,refs,z,a,'long' if mode=='long' else 'grid'))
        except Exception as exc:r['error']=repr(exc)
        records.append(r);byjob[jid]=r;save(cp,records)
        print(f'{len(records):04d} edge={e} scale={s:.12g} angle={angle:.10g} {mode}: {r["verdict"]}',flush=True)
        return r
    calibration=[];predictions=[];crossings=[];checks=[];summary=[];rows=[];selection=[]
    def publish():
        report=dict(scope='P40 prospective local stress test: P39 model errors guide sites; P39 calibration frozen; P40 holdouts excluded from fit and calibration',zeta=z,reference_drifts=refs,source_sha256=config['source_sha256'],settings=vars(a),selection=selection,calibration=calibration,rows=rows,predictions=predictions,crossings=crossings,verifications=checks,summary=summary,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['P40 stress sites are adaptively chosen using P39 errors; not random or independent of previous site selection.','Frozen P39 allowances are descriptive residual envelopes, not confidence intervals.','P40 holdouts never update the models or allowances.','Adjacent radial sites can be correlated.','Local scans may miss remote or subgrid crossings.','Finite-time 2D initial-condition slices only; no global or asymptotic claims.'])
        save(out/'p40_report.json',report)
        csvout(out/'p40_predictions.csv',predictions,['edge_index','row_index','scale','method','prediction_degrees','numerical_lower','numerical_upper','calibration_max_error_degrees','calibrated_lower','calibrated_upper','observed_lower','observed_upper','observed_midpoint','absolute_error_degrees','numerical_interval_intersects_observed','calibrated_interval_intersects_observed','crossing_status'])
        csvout(out/'p40_crossings.csv',crossings,['edge_index','row_index','scale','selection_reason','scan_min','scan_max','n_scan_samples','n_crossings','n_ambiguous','angle_a','angle_b','class_a','class_b','width_degrees','levels_completed','midpoint_degrees','status'])
        csvout(out/'p40_verifications.csv',checks,['edge_index','row_index','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
        csvout(out/'p40_summary.csv',summary,['edge_index','method','n_training','n_calibration','calibration_max_error_degrees','n_holdout_rows','n_refined_crossings','n_numerical_intersections','n_calibrated_intersections','median_abs_holdout_error_degrees','max_abs_holdout_error_degrees','n_long_checks','n_long_reproduced','n_complete','n_failed'])
    for e in (30,34):
        if a.only_edge is not None and e!=a.only_edge:continue
        old=anchors35(r35,e);cal=anchors36(r36,e)
        prior=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P37',class_a=r['class_a'],class_b=r['class_b']) for r in r37 if int(r['edge_index'])==e and r['status']=='refined']
        p38pts=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P38',class_a=r['class_a'],class_b=r['class_b'],row_index=int(r['row_index'])) for r in r38 if int(r['edge_index'])==e and r['status']=='refined']
        if len(old)!=4 or len(cal)!=3 or len(prior)!=2 or len(p38pts)!=4:raise ValueError('Expected complete P35-P38 crossings')
        if len({(v['class_a'],v['class_b']) for v in old+cal+prior+p38pts})!=1:raise ValueError('Crossing orientation mismatch')
        # P39 holdouts become historical training anchors only; calibration remains
        # the frozen P39 P38-reserved maxima, not recalibrated on P39 or P40.
        p39pts=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P39',class_a=r['class_a'],class_b=r['class_b'],row_index=int(r['row_index'])) for r in r39 if int(r['edge_index'])==e and r['status']=='refined']
        if len(p39pts)!=3:raise ValueError('Expected 3 complete P39 crossings per edge')
        if len({(v['class_a'],v['class_b']) for v in old+cal+prior+p38pts+p39pts})!=1:raise ValueError('Crossing orientation mismatch')
        train=sorted(old+cal+prior+[v for v in p38pts if v['row_index'] in (0,2)]+p39pts,key=lambda v:v['scale'])
        if len(train)!=14:raise ValueError('Expected 14 frozen training anchors')
        p39summ={r['method']:r for r in sm39 if int(r['edge_index'])==e}
        if set(p39summ)!=set(METHODS):raise ValueError('Incomplete P39 frozen allowances')
        margins={m:float(p39summ[m]['calibration_max_error_degrees']) for m in METHODS}
        errors={m:[margins[m]] for m in METHODS} # one frozen summary allowance; NOT one calibration observation
        for m in METHODS:calibration.append(dict(edge_index=e,method=m,calibration_max_error_degrees=margins[m],source='P39_frozen_P38_reserved_calibration',n_original_calibration=2))
        # Candidate geometry is determined entirely by historical anchors.
        candidates=sorted(old+cal+prior+p38pts+p39pts,key=lambda v:v['scale'])
        gaps=list(range(len(candidates)-1));chosen=[];reason_by_gap={}
        count=a.edge34_rows if e==34 else a.edge30_rows
        if e==34:
            # Probe BOTH sides of P39's worst quadratic miss, then the region
            # around the prior P39 edge-34 stress site. Do not use P40 outcomes.
            stress=max((r for r in pr39 if int(r['edge_index'])==34 and r['method']=='quadratic'),key=lambda r:float(r['absolute_error_degrees']))
            target=float(stress['scale']);j=min(range(len(candidates)),key=lambda i:abs(candidates[i]['scale']-target))
            if abs(candidates[j]['scale']-target)>1e-12:raise ValueError('P39 stress scale mismatch')
            for gi in (j-1,j):
                if gi in gaps and gi not in chosen:chosen.append(gi);reason_by_gap[gi]='both_sides_of_p39_worst_quadratic_miss'
            second=max((r for r in pr39 if int(r['edge_index'])==34 and r['method']=='linear' and int(r['row_index'])!=int(stress['row_index'])),key=lambda r:float(r['absolute_error_degrees']))
            target2=float(second['scale']);k=min(range(len(candidates)),key=lambda i:abs(candidates[i]['scale']-target2))
            for gi in sorted((k-1,k),key=lambda gi:-(candidates[gi+1]['scale']-candidates[gi]['scale']) if gi in gaps else float('inf')):
                if gi in gaps and gi not in chosen and len(chosen)<count:chosen.append(gi);reason_by_gap[gi]='p39_secondary_linear_error_neighborhood';break
        for gi in sorted(gaps,key=lambda k:(-(candidates[k+1]['scale']-candidates[k]['scale']),k)):
            if len(chosen)>=count:break
            if gi not in chosen and all(abs(gi-k)>1 for k in chosen):chosen.append(gi);reason_by_gap[gi]='widest_nonadjacent_gap'
        for gi in sorted(gaps,key=lambda k:(-(candidates[k+1]['scale']-candidates[k]['scale']),k)):
            if len(chosen)>=count:break
            if gi not in chosen:chosen.append(gi);reason_by_gap[gi]='widest_remaining_gap'
        if len(chosen)!=count:raise ValueError('Insufficient prospective gaps')
        for ri,gi in enumerate(sorted(chosen)):
            s=(candidates[gi]['scale']+candidates[gi+1]['scale'])/2
            reason=reason_by_gap[gi]
            selection.append(dict(edge_index=e,row_index=ri,scale=s,reason=reason,source_gap_index=gi,source_gap_width=candidates[gi+1]['scale']-candidates[gi]['scale']))
            pp={m:prediction(m,train,s) for m in METHODS}
            low=min(q['numerical_low']-margins[m] for m,q in pp.items())-a.scan_margin
            high=max(q['numerical_high']+margins[m] for m,q in pp.items())+a.scan_margin
            n=max(3,int(math.ceil((high-low)/a.scan_step))+1)
            angles=np.linspace(low,high,n)
            samples=[dict(angle=float(ang),verdict=run(e,s,float(ang),'scan')['verdict']) for ang in angles]
            pairs=[(l,r) for l,r in zip(samples,samples[1:]) if l['verdict'] in refs and r['verdict'] in refs and l['verdict']!=r['verdict']]
            namb=sum(v['verdict'] not in refs for v in samples)
            status='single_crossing' if len(pairs)==1 and not namb else ('no_crossing' if not pairs and not namb else ('multiple_crossings' if len(pairs)>1 and not namb else 'ambiguous_scan'))
            rows.append(dict(edge_index=e,row_index=ri,scale=s,selection_reason=reason,source_gap_index=gi,scan_min=low,scan_max=high,n_scan_samples=n,n_crossings=len(pairs),n_ambiguous=namb,status=status,samples=samples))
            if status!='single_crossing':
                crossings.append(dict(edge_index=e,row_index=ri,scale=s,selection_reason=reason,scan_min=low,scan_max=high,n_scan_samples=n,n_crossings=len(pairs),n_ambiguous=namb,status=status))
                for m,q in pp.items():predictions.append(dict(edge_index=e,row_index=ri,scale=s,method=m,prediction_degrees=q['center'],numerical_lower=q['numerical_low'],numerical_upper=q['numerical_high'],calibration_max_error_degrees=margins[m],calibrated_lower=q['numerical_low']-margins[m],calibrated_upper=q['numerical_high']+margins[m],crossing_status=status))
                publish();continue
            lo,hi=[dict(v) for v in pairs[0]];done=0;status='refined'
            for _ in range(a.levels):
                mid=(lo['angle']+hi['angle'])/2;v=run(e,s,mid,'refine')['verdict']
                if v==lo['verdict']:lo=dict(angle=mid,verdict=v)
                elif v==hi['verdict']:hi=dict(angle=mid,verdict=v)
                else:status='ambiguous_midpoint';break
                done+=1
            mid=(lo['angle']+hi['angle'])/2
            crossing=dict(edge_index=e,row_index=ri,scale=s,selection_reason=reason,scan_min=low,scan_max=high,n_scan_samples=n,n_crossings=1,n_ambiguous=namb,angle_a=lo['angle'],angle_b=hi['angle'],class_a=lo['verdict'],class_b=hi['verdict'],width_degrees=hi['angle']-lo['angle'],levels_completed=done,midpoint_degrees=mid,status=status)
            crossings.append(crossing)
            for m,q in pp.items():
                nl,nu=q['numerical_low'],q['numerical_high'];cl,cu=nl-margins[m],nu+margins[m]
                predictions.append(dict(edge_index=e,row_index=ri,scale=s,method=m,prediction_degrees=q['center'],numerical_lower=nl,numerical_upper=nu,calibration_max_error_degrees=margins[m],calibrated_lower=cl,calibrated_upper=cu,observed_lower=lo['angle'],observed_upper=hi['angle'],observed_midpoint=mid,absolute_error_degrees=abs(mid-q['center']),numerical_interval_intersects_observed=not (nu<lo['angle'] or nl>hi['angle']),calibrated_interval_intersects_observed=not (cu<lo['angle'] or cl>hi['angle']),crossing_status=status))
            publish()
            if status=='refined':
                for side,pt in (('a',lo),('b',hi)):
                    r=run(e,s,pt['angle'],'long')
                    checks.append(dict(edge_index=e,row_index=ri,side=side,scale=s,angle_degrees=pt['angle'],expected_class=pt['verdict'],observed_class=r['verdict'],reproduced=r['status']=='complete' and r['verdict']==pt['verdict'],all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
                    publish()
        for m in METHODS:
            pr=[v for v in predictions if v['edge_index']==e and v['method']==m];cr=[v for v in crossings if v['edge_index']==e];ve=[v for v in checks if v['edge_index']==e];tr=[v for v in records if v['edge_index']==e]
            valid=[v for v in pr if v['crossing_status']=='refined']
            summary.append(dict(edge_index=e,method=m,n_training=len(train),n_calibration=2,calibration_max_error_degrees=margins[m],n_holdout_rows=len(cr),n_refined_crossings=len(valid),n_numerical_intersections=sum(v['numerical_interval_intersects_observed'] for v in valid),n_calibrated_intersections=sum(v['calibrated_interval_intersects_observed'] for v in valid),median_abs_holdout_error_degrees=float(np.median([v['absolute_error_degrees'] for v in valid])) if valid else None,max_abs_holdout_error_degrees=max((v['absolute_error_degrees'] for v in valid),default=None),n_long_checks=len(ve),n_long_reproduced=sum(v['reproduced'] and v['all_windows_consistent'] for v in ve),n_complete=sum(v['status']=='complete' for v in tr),n_failed=sum(v['status']!='complete' for v in tr)))
        publish()
    print(f'P40 complete: {len(crossings)} new radial rows, {len(checks)} long checks, {len(records)} integrations; failures={sum(r["status"]!="complete" for r in records)}',flush=True)
if __name__=='__main__':main()

