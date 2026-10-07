#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p43_geometry_based_boundary_validation.py

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
'p40_report':'results/author_local_boundary_stress_test/p40_report.json',
'p40_predictions':'results/author_local_boundary_stress_test/p40_predictions.csv',
'p40_crossings':'results/author_local_boundary_stress_test/p40_crossings.csv',
'p40_verifications':'results/author_local_boundary_stress_test/p40_verifications.csv',
'p40_summary':'results/author_local_boundary_stress_test/p40_summary.csv',
'p41_report':'results/author_spatial_error_validation/p41_report.json',
'p41_predictions':'results/author_spatial_error_validation/p41_predictions.csv',
'p41_crossings':'results/author_spatial_error_validation/p41_crossings.csv',
'p41_verifications':'results/author_spatial_error_validation/p41_verifications.csv',
'p41_summary':'results/author_spatial_error_validation/p41_summary.csv',
'p41_diagnostics':'results/author_spatial_error_validation/p41_diagnostics.csv',
'p42_report':'results/author_prospective_error_horizon_validation/p42_report.json',
'p42_predictions':'results/author_prospective_error_horizon_validation/p42_predictions.csv',
'p42_crossings':'results/author_prospective_error_horizon_validation/p42_crossings.csv',
'p42_verifications':'results/author_prospective_error_horizon_validation/p42_verifications.csv',
'p42_extended_verifications':'results/author_prospective_error_horizon_validation/p42_extended_verifications.csv',
'p42_summary':'results/author_prospective_error_horizon_validation/p42_summary.csv',
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
    ap.add_argument('--output-dir',default='results/author_geometry_based_boundary_validation')
    ap.add_argument('--only-edge',type=int,choices=[30,34]);ap.add_argument('--edge30-rows',type=int,default=2);ap.add_argument('--edge34-rows',type=int,default=2);ap.add_argument('--diagnostic-only',action='store_true',help='Compute frozen spatial diagnostics without new integrations')
    ap.add_argument('--extended-observe',type=float,default=6400.,help='Additional endpoint observation horizon; 0 disables')
    ap.add_argument('--scan-margin',type=float,default=.025,help='Degrees beyond min/max calibrated prediction intervals')
    ap.add_argument('--scan-step',type=float,default=.01);ap.add_argument('--levels',type=int,default=7)
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
    ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9);ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--verify-observe',type=float,default=3200.);ap.add_argument('--verify-dt',type=float,default=.25)
    ap.add_argument('--verify-rtol',type=float,default=1e-11);ap.add_argument('--verify-atol',type=float,default=1e-13)
    ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if not 1<=a.edge30_rows<=5 or not 1<=a.edge34_rows<=6 or a.scan_margin<=0 or a.scan_step<=0 or a.levels<0 or a.windows<4 or a.extended_observe<0 or any(v<=0 for v in (a.observe,a.dt,a.rtol,a.atol,a.verify_observe,a.verify_dt,a.verify_rtol,a.verify_atol)):
        ap.error('Invalid numerical settings')
    paths={k:Path(getattr(a,k)) for k in DEFAULTS}
    src={k:json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report','p34_report','p35_report','p36_report','p37_report','p38_report','p39_report','p40_report','p41_report','p42_report')}
    p23,p24,p27,p32,p34,p35,p36,p37,p38,p39,p40,p41,p42=[src[k] for k in ('p23_report','p24_report','p27_report','p32_report','p34_report','p35_report','p36_report','p37_report','p38_report','p39_report','p40_report','p41_report','p42_report')]
    refs=p35['reference_drifts'];z=float(p35['zeta'])
    if any(abs(float(src[k]['zeta'])-z)>1e-12 or src[k]['reference_drifts']!=refs for k in ('p27_report','p32_report','p34_report','p36_report','p37_report','p38_report','p39_report','p40_report','p41_report','p42_report')):raise ValueError('Source zeta/reference mismatch')
    if p35['n_failed'] or p36['n_failed'] or p37['n_failed'] or p38['n_failed'] or p39['n_failed'] or p40['n_failed'] or p41['n_failed'] or p42['n_failed']:raise ValueError('P35/P36 has failed integrations')
    for report,keys in ((p34,('p23_report','p24_report','p27_report','p32_report')),(p35,('p23_report','p24_report','p27_report','p32_report','p34_report','p34_tests','p34_summary','p34_verifications')),(p36,tuple(p36['source_sha256'])),(p37,tuple(p37['source_sha256'])),(p38,tuple(p38['source_sha256'])),(p39,tuple(p39['source_sha256'])),(p40,tuple(p40['source_sha256'])),(p41,tuple(p41['source_sha256'])),(p42,tuple(p42['source_sha256']))):
        for k in keys:
            if digest(paths[k])!=report['source_sha256'][k]:raise ValueError('Source SHA mismatch '+k)
    r42=readcsv(paths['p42_crossings']);pr42=readcsv(paths['p42_predictions']);sm42=readcsv(paths['p42_summary']);v42=readcsv(paths['p42_verifications']);ev42=readcsv(paths['p42_extended_verifications'])
    if len(r42)!=len(p42['crossings']) or len(pr42)!=len(p42['predictions']) or len(sm42)!=len(p42['summary']) or len(v42)!=len(p42['verifications']) or len(ev42)!=len(p42['extended_verifications']):raise ValueError('P42 CSV/report count mismatch')
    for c,j in zip(r42,p42['crossings']):
        for k in ('edge_index','scale','angle_a','angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P42 crossing CSV/report mismatch '+k)
    if any(not yes(r['reproduced']) or not yes(r['all_windows_consistent']) for r in v42+ev42):raise ValueError('P42 horizon verification failure')
    r35=readcsv(paths['p35_refinements']);r36=readcsv(paths['p36_crossings']);v35=readcsv(paths['p35_verifications']);v36=readcsv(paths['p36_verifications']);r37=readcsv(paths['p37_crossings']);v37=readcsv(paths['p37_verifications']);pr37=readcsv(paths['p37_predictions']);sm37=readcsv(paths['p37_summary']);r38=readcsv(paths['p38_crossings']);v38=readcsv(paths['p38_verifications']);pr38=readcsv(paths['p38_predictions']);sm38=readcsv(paths['p38_summary']);r39=readcsv(paths['p39_crossings']);v39=readcsv(paths['p39_verifications']);pr39=readcsv(paths['p39_predictions']);sm39=readcsv(paths['p39_summary']);r40=readcsv(paths['p40_crossings']);v40=readcsv(paths['p40_verifications']);pr40=readcsv(paths['p40_predictions']);sm40=readcsv(paths['p40_summary']);r41=readcsv(paths['p41_crossings']);v41=readcsv(paths['p41_verifications']);pr41=readcsv(paths['p41_predictions']);sm41=readcsv(paths['p41_summary']);dg41=readcsv(paths['p41_diagnostics'])
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
    if len(v39)!=12 or len(v40)!=12 or len(v41)!=12:raise ValueError('Expected 12 P39 and 12 P40 long checks')
    if any(not yes(v['reproduced']) or not yes(v['all_windows_consistent']) for v in v35+v36+v37+v38+v39+v40+v41):raise ValueError('P35/P36 long verification incomplete')
    if any(r['status']!='refined' for r in r35+r36+r37+r38+r39+r40+r41+r42):raise ValueError('Unresolved source crossings')
    if len(r39)!=len(p39['crossings']) or len(pr39)!=len(p39['predictions']) or len(sm39)!=len(p39['summary']):raise ValueError('P39 CSV/report count mismatch')
    for c,j in zip(r39,p39['crossings']):
        for k in ('edge_index','scale','angle_a','angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P39 crossing CSV/report mismatch '+k)
    if len(r40)!=len(p40['crossings']) or len(pr40)!=len(p40['predictions']) or len(sm40)!=len(p40['summary']):raise ValueError('P40 CSV/report count mismatch')
    for c,j in zip(r40,p40['crossings']):
        for k in ('edge_index','scale','angle_a','angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P40 crossing CSV/report mismatch '+k)
    if len(r41)!=len(p41['crossings']) or len(pr41)!=len(p41['predictions']) or len(sm41)!=len(p41['summary']) or len(dg41)!=len(p41['diagnostics']):raise ValueError('P41 CSV/report count mismatch')
    for c,j in zip(r41,p41['crossings']):
        for k in ('edge_index','scale','angle_a','angle_b'):
            if abs(float(c[k])-float(j[k]))>1e-11:raise ValueError('P41 crossing CSV/report mismatch '+k)
    target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
    seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
    x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
    if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('Base direction mismatch')
    perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
    directions={int(r['edge_index']):int(r['direction_index']) for r in p32['rows']}
    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    config={'source_sha256':{k:digest(v) for k,v in paths.items()},'settings':{k:v for k,v in vars(a).items() if k not in ('resume','output_dir')}}
    cp=out/'p43_checkpoint.json';cf=out/'p43_config.json'
    if a.resume and cp.exists() and (not cf.exists() or json.loads(cf.read_text())!=config):raise ValueError('Resume configuration/source mismatch')
    if not a.resume and cp.exists():raise ValueError('Checkpoint exists: use --resume or new output directory')
    save(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];byjob={r['job_id']:r for r in records}
    def run(e,s,angle,mode):
        jid=f'e{e}_s{s:.17g}_a{angle:.17g}_{mode}'
        if jid in byjob:return byjob[jid]
        r=dict(job_id=jid,edge_index=e,scale=float(s),angle_degrees=float(angle),mode=mode,status='failed',verdict='failed')
        try:
            if mode=='extended':
                from types import SimpleNamespace
                aa=SimpleNamespace(**vars(a));aa.verify_observe=a.extended_observe
                r.update(simulate(x0,p,base,perps,directions[e],s,angle,refs,z,aa,'long'))
            else:r.update(simulate(x0,p,base,perps,directions[e],s,angle,refs,z,a,'long' if mode=='long' else 'grid'))
        except Exception as exc:r['error']=repr(exc)
        records.append(r);byjob[jid]=r;save(cp,records)
        print(f'{len(records):04d} edge={e} scale={s:.12g} angle={angle:.10g} {mode}: {r["verdict"]}',flush=True)
        return r
    calibration=[];predictions=[];crossings=[];checks=[];extended_checks=[];summary=[];rows=[];selection=[];diagnostics=[];gap_diagnostics=[]
    def publish():
        report=dict(scope='P43: frozen geometry-aware uncertainty versus empirical P42 baseline at untouched P43 sites',zeta=z,reference_drifts=refs,source_sha256=config['source_sha256'],settings=vars(a),selection=selection,calibration=calibration,diagnostics=diagnostics,gap_diagnostics=gap_diagnostics,rows=rows,predictions=predictions,crossings=crossings,verifications=checks,extended_verifications=extended_checks,summary=summary,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Historical leave-one-out residuals are diagnostic, not out-of-sample performance.','P40-P42 sites were adaptively selected and enter P43 historical data; P43 sites are chosen using P35-P42 geometry only.','Geometry-only intervals calibrated on two reserved P38 sites per map and empirical baseline are deterministic descriptive bounds, not confidence intervals.', 'The geometry proxy can be small even where interpolation error is large; P43 explicitly tests this failure mode.','P43 holdouts do not update predictions, training, or calibration; empirical baseline and geometry model are frozen before integration.','Finite-time selected 2D slices; scans may miss remote or subgrid crossings.'])
        save(out/'p43_report.json',report)
        csvout(out/'p43_diagnostics.csv',diagnostics,['edge_index','source','scale','method','observed_midpoint','loo_prediction','loo_absolute_error','nearest_anchor_distance','local_gap_width','local_slope_change','status'])
        csvout(out/'p43_gap_diagnostics.csv',gap_diagnostics,['edge_index','gap_index','left_scale','right_scale','mid_scale','width','slope_left','slope_right','slope_change','score','selected'])
        csvout(out/'p43_predictions.csv',predictions,['edge_index','row_index','scale','method','prediction_degrees','numerical_lower','numerical_upper','calibration_max_error_degrees','calibrated_lower','calibrated_upper','observed_lower','observed_upper','observed_midpoint','absolute_error_degrees','numerical_interval_intersects_observed','calibrated_interval_intersects_observed','baseline_allowance_degrees','baseline_lower','baseline_upper','baseline_interval_intersects_observed','crossing_status'])
        csvout(out/'p43_crossings.csv',crossings,['edge_index','row_index','scale','selection_reason','scan_min','scan_max','n_scan_samples','n_crossings','n_ambiguous','angle_a','angle_b','class_a','class_b','width_degrees','levels_completed','midpoint_degrees','status'])
        csvout(out/'p43_verifications.csv',checks,['edge_index','row_index','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
        csvout(out/'p43_extended_verifications.csv',extended_checks,['edge_index','row_index','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id','observe'])
        csvout(out/'p43_summary.csv',summary,['edge_index','method','n_training','n_calibration','calibration_max_error_degrees','n_holdout_rows','n_refined_crossings','n_numerical_intersections','n_calibrated_intersections','median_abs_holdout_error_degrees','max_abs_holdout_error_degrees','n_long_checks','n_long_reproduced','n_extended_checks','n_extended_reproduced','n_baseline_intersections','n_complete','n_failed'])
    for e in (30,34):
        if a.only_edge is not None and e!=a.only_edge:continue
        old=anchors35(r35,e);cal=anchors36(r36,e)
        prior=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P37',class_a=r['class_a'],class_b=r['class_b']) for r in r37 if int(r['edge_index'])==e and r['status']=='refined']
        p38pts=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P38',class_a=r['class_a'],class_b=r['class_b'],row_index=int(r['row_index'])) for r in r38 if int(r['edge_index'])==e and r['status']=='refined']
        if len(old)!=4 or len(cal)!=3 or len(prior)!=2 or len(p38pts)!=4:raise ValueError('Expected complete P35-P38 crossings')
        if len({(v['class_a'],v['class_b']) for v in old+cal+prior+p38pts})!=1:raise ValueError('Crossing orientation mismatch')
        # P41 crossing outcomes are permitted as P43 training anchors; P42 sites remain held out.
        # P38 reserved rows 1/3 stay calibration-only, excluded from training.
        p39pts=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P39',class_a=r['class_a'],class_b=r['class_b'],row_index=int(r['row_index'])) for r in r39 if int(r['edge_index'])==e and r['status']=='refined']
        p41pts=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P41',class_a=r['class_a'],class_b=r['class_b'],row_index=int(r['row_index'])) for r in r41 if int(r['edge_index'])==e and r['status']=='refined']
        p40pts=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P40',class_a=r['class_a'],class_b=r['class_b'],row_index=int(r['row_index'])) for r in r40 if int(r['edge_index'])==e and r['status']=='refined']
        p42pts=[dict(scale=float(r['scale']),low=float(r['angle_a']),high=float(r['angle_b']),source='P42',class_a=r['class_a'],class_b=r['class_b'],row_index=int(r['row_index'])) for r in r42 if int(r['edge_index'])==e and r['status']=='refined']
        if len(p42pts)!=2:raise ValueError('Expected two complete P42 crossings per map')
        if len(p39pts)!=3 or len(p40pts)!=(2 if e==30 else 4) or len(p41pts)!=(2 if e==30 else 4):raise ValueError('Expected complete P39/P40 crossing sets')
        train=sorted(old+cal+prior+[v for v in p38pts if v['row_index'] in (0,2)]+p39pts+p40pts+p41pts+p42pts,key=lambda v:v['scale'])
        expected=20 if e==30 else 24
        if len(train)!=expected or len({round(v['scale'],14) for v in train})!=expected:raise ValueError('Unexpected or duplicated training anchors')
        if len({(v['class_a'],v['class_b']) for v in train+p38pts})!=1:raise ValueError('Crossing orientation mismatch')
        # Frozen P40 observed errors form an empirical error floor. Historical
        # leave-one-out errors diagnose geometry; only the four nearest
        # INTERIOR anchor LOO residuals are used for a spatial envelope.
        historic={m:[float(r['absolute_error_degrees']) for r in pr40+pr41+pr42 if int(r['edge_index'])==e and r['method']==m and r['crossing_status']=='refined'] for m in METHODS}
        if any(len(historic[m])!=(6 if e==30 else 10) for m in METHODS):raise ValueError('Incomplete P40 prediction errors')
        loo={m:[] for m in METHODS}
        xs=np.array([v['scale'] for v in train]);ys=np.array([(v['low']+v['high'])/2 for v in train]);slopes=np.diff(ys)/np.diff(xs)
        for i,v in enumerate(train):
            left=max(0,i-1);right=min(len(train)-2,i)
            near=min(abs(xs[i]-xs[j]) for j in range(len(train)) if j!=i)
            gap=max(xs[i]-xs[left],xs[right+1]-xs[i]) if 0<i<len(train)-1 else float('nan')
            slope_change=abs(slopes[i]-slopes[i-1]) if 0<i<len(train)-1 else float('nan')
            for m in METHODS:
                if i in (0,len(train)-1):
                    diagnostics.append(dict(edge_index=e,source=v['source'],scale=v['scale'],method=m,observed_midpoint=ys[i],status='endpoint_excluded_no_extrapolation'))
                    continue
                q=prediction(m,train[:i]+train[i+1:],v['scale']);err=abs(q['center']-ys[i]);loo[m].append((float(xs[i]),float(err)))
                diagnostics.append(dict(edge_index=e,source=v['source'],scale=v['scale'],method=m,observed_midpoint=ys[i],loo_prediction=q['center'],loo_absolute_error=err,nearest_anchor_distance=near,local_gap_width=gap,local_slope_change=slope_change,status='interior_LOO_diagnostic'))
        # Freeze a geometry-aware deterministic envelope before P43 outcomes.
        # For each site, proxy = |local quadratic - linear| + half the
        # spread of linear/quadratic/PCHIP predictions + a small bracket floor.
        # Its scale is calibrated on reserved P38 rows 1/3 (never training).
        # Prior P40-P42 errors provide an independent empirical comparison,
        # not an automatic floor for the geometry-only interval.
        def geometry_proxy(s):
            q={m:prediction(m,train,s) for m in METHODS}
            centers=[q[m]['center'] for m in METHODS]
            bracket=max(q[m]['numerical_high']-q[m]['numerical_low'] for m in METHODS)
            return max(1e-7,abs(q['quadratic']['center']-q['linear']['center']) + 0.5*(max(centers)-min(centers)) + 0.5*bracket)
        reserved=[v for v in p38pts if v['row_index'] in (1,3)]
        # A curvature proxy may vanish locally; the ratio exposes this
        # failure rather than silently claiming reliable coverage.
        geometry_factors={}
        for m in METHODS:
            ratios=[]
            for v in reserved:
                observed=(v['low']+v['high'])/2
                err=abs(prediction(m,train,v['scale'])['center']-observed)
                ratios.append(err/geometry_proxy(v['scale']))
            geometry_factors[m]=max(ratios)
            calibration.append(dict(edge_index=e,method=m,geometry_factor=geometry_factors[m],geometry_calibration_sites=2,empirical_max_observed_error_degrees=max(historic[m]),max_historical_loo_error_degrees=max(err for _,err in loo[m]),source='P38_reserved_geometry_ratio_vs_P40_P42_empirical_baseline',n_original_calibration=len(historic[m]),n_training=expected))
        # Gap ranking is fixed before P43 integration; score balances gap width
        # and slope changes, while enforcing radial separation where possible.
        gaps=list(range(len(train)-1));gap_scores={}
        for gi in gaps:
            left=slopes[max(0,gi-1)];right=slopes[min(len(slopes)-1,gi+1)]
            curvature=abs(right-left);width=float(xs[gi+1]-xs[gi]);score=width*(1+min(3.,curvature/max(1.,abs(slopes[gi]))))
            gap_scores[gi]=score
        count=a.edge34_rows if e==34 else a.edge30_rows
        chosen=[]
        for gi in sorted(gaps,key=lambda k:(-gap_scores[k],k)):
            if len(chosen)>=count:break
            if all(abs(gi-k)>1 for k in chosen):chosen.append(gi)
        for gi in sorted(gaps,key=lambda k:(-gap_scores[k],k)):
            if len(chosen)>=count:break
            if gi not in chosen:chosen.append(gi)
        if len(chosen)!=count:raise ValueError('Insufficient unused gaps')
        for gi in gaps:
            gap_diagnostics.append(dict(edge_index=e,gap_index=gi,left_scale=float(xs[gi]),right_scale=float(xs[gi+1]),mid_scale=float((xs[gi]+xs[gi+1])/2),width=float(xs[gi+1]-xs[gi]),slope_left=float(slopes[max(0,gi-1)]),slope_right=float(slopes[min(len(slopes)-1,gi+1)]),slope_change=float(abs(slopes[max(0,gi-1)]-slopes[min(len(slopes)-1,gi+1)])),score=float(gap_scores[gi]),selected=gi in chosen))
        for ri,gi in enumerate(sorted(chosen)):
            s=float((xs[gi]+xs[gi+1])/2)
            reason='historical_gap_width_and_slope_change_rank'
            selection.append(dict(edge_index=e,row_index=ri,scale=s,reason=reason,source_gap_index=gi,source_gap_width=float(xs[gi+1]-xs[gi]),geometry_proxy_degrees=geometry_proxy(s),geometry_factors=geometry_factors))
            pp={m:prediction(m,train,s) for m in METHODS}
            # Geometry-only allowance: frozen curvature proxy times reserved-calibration ratio.
            # P43 outcomes cannot change this allowance.
            margins={m:geometry_factors[m]*geometry_proxy(s) for m in METHODS}
            # Empirical baseline uses historical P40-P42 errors plus four nearest LOO residuals.
            baseline={m:max(max(historic[m]),*(err for _,err in sorted(loo[m],key=lambda item:abs(item[0]-s))[:4])) for m in METHODS}
            if a.diagnostic_only:continue
            low=min(q['numerical_low']-max(margins[m],baseline[m]) for m,q in pp.items())-a.scan_margin
            high=max(q['numerical_high']+max(margins[m],baseline[m]) for m,q in pp.items())+a.scan_margin
            n=max(3,int(math.ceil((high-low)/a.scan_step))+1)
            angles=np.linspace(low,high,n)
            samples=[dict(angle=float(ang),verdict=run(e,s,float(ang),'scan')['verdict']) for ang in angles]
            pairs=[(l,r) for l,r in zip(samples,samples[1:]) if l['verdict'] in refs and r['verdict'] in refs and l['verdict']!=r['verdict']]
            namb=sum(v['verdict'] not in refs for v in samples)
            status='single_crossing' if len(pairs)==1 and not namb else ('no_crossing' if not pairs and not namb else ('multiple_crossings' if len(pairs)>1 and not namb else 'ambiguous_scan'))
            rows.append(dict(edge_index=e,row_index=ri,scale=s,selection_reason=reason,source_gap_index=gi,scan_min=low,scan_max=high,n_scan_samples=n,n_crossings=len(pairs),n_ambiguous=namb,status=status,samples=samples))
            if status!='single_crossing':
                crossings.append(dict(edge_index=e,row_index=ri,scale=s,selection_reason=reason,scan_min=low,scan_max=high,n_scan_samples=n,n_crossings=len(pairs),n_ambiguous=namb,status=status))
                for m,q in pp.items():predictions.append(dict(edge_index=e,row_index=ri,scale=s,method=m,prediction_degrees=q['center'],numerical_lower=q['numerical_low'],numerical_upper=q['numerical_high'],calibration_max_error_degrees=margins[m],baseline_allowance_degrees=baseline[m],baseline_lower=q['numerical_low']-baseline[m],baseline_upper=q['numerical_high']+baseline[m],calibrated_lower=q['numerical_low']-margins[m],calibrated_upper=q['numerical_high']+margins[m],crossing_status=status))
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
                predictions.append(dict(edge_index=e,row_index=ri,scale=s,method=m,prediction_degrees=q['center'],numerical_lower=nl,numerical_upper=nu,calibration_max_error_degrees=margins[m],baseline_allowance_degrees=baseline[m],baseline_lower=q['numerical_low']-baseline[m],baseline_upper=q['numerical_high']+baseline[m],calibrated_lower=cl,calibrated_upper=cu,observed_lower=lo['angle'],observed_upper=hi['angle'],observed_midpoint=mid,absolute_error_degrees=abs(mid-q['center']),numerical_interval_intersects_observed=not (nu<lo['angle'] or nl>hi['angle']),calibrated_interval_intersects_observed=not (cu<lo['angle'] or cl>hi['angle']),baseline_interval_intersects_observed=not (nu+baseline[m]<lo['angle'] or nl-baseline[m]>hi['angle']),crossing_status=status))
            publish()
            if status=='refined':
                for side,pt in (('a',lo),('b',hi)):
                    r=run(e,s,pt['angle'],'long')
                    checks.append(dict(edge_index=e,row_index=ri,side=side,scale=s,angle_degrees=pt['angle'],expected_class=pt['verdict'],observed_class=r['verdict'],reproduced=r['status']=='complete' and r['verdict']==pt['verdict'],all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
                    publish()
                    if a.extended_observe>0:
                        x=run(e,s,pt['angle'],'extended')
                        extended_checks.append(dict(edge_index=e,row_index=ri,side=side,scale=s,angle_degrees=pt['angle'],expected_class=pt['verdict'],observed_class=x['verdict'],reproduced=x['status']=='complete' and x['verdict']==pt['verdict'],all_windows_consistent=x.get('all_windows_consistent',False),status=x['status'],job_id=x['job_id'],observe=a.extended_observe))
                        publish()
        for m in METHODS:
            pr=[v for v in predictions if v['edge_index']==e and v['method']==m];cr=[v for v in crossings if v['edge_index']==e];ve=[v for v in checks if v['edge_index']==e];tr=[v for v in records if v['edge_index']==e]
            valid=[v for v in pr if v['crossing_status']=='refined']
            summary.append(dict(edge_index=e,method=m,n_training=len(train),n_calibration=len(historic[m]),calibration_max_error_degrees=max(historic[m]),n_holdout_rows=len(cr),n_refined_crossings=len(valid),n_numerical_intersections=sum(v['numerical_interval_intersects_observed'] for v in valid),n_calibrated_intersections=sum(v['calibrated_interval_intersects_observed'] for v in valid),median_abs_holdout_error_degrees=float(np.median([v['absolute_error_degrees'] for v in valid])) if valid else None,max_abs_holdout_error_degrees=max((v['absolute_error_degrees'] for v in valid),default=None),n_long_checks=len(ve),n_long_reproduced=sum(v['reproduced'] and v['all_windows_consistent'] for v in ve),n_extended_checks=sum(v['edge_index']==e for v in extended_checks),n_extended_reproduced=sum(v['edge_index']==e and v['reproduced'] and v['all_windows_consistent'] for v in extended_checks),n_baseline_intersections=sum(v.get('baseline_interval_intersects_observed',False) for v in valid),n_complete=sum(v['status']=='complete' for v in tr),n_failed=sum(v['status']!='complete' for v in tr)))
        publish()
    print(f'P43 complete: {len(crossings)} new radial rows, {len(checks)} long checks, {len(records)} integrations; failures={sum(r["status"]!="complete" for r in records)}',flush=True)
if __name__=='__main__':main()

