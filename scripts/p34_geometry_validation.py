#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p34_geometry_validation.py

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
from p00_model_validation import Parameters
from p32_boundary_continuation import simulate,digest,save,csvout

def loadcsv(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def midpoint(c):return (float(c['angle_a'])+float(c['angle_b']))/2

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 defaults={'p23_report':'results/author_longtime_verification/p23_report.json','p24_report':'results/author_basin_attraction/p24_report.json','p27_report':'results/author_local_boundary_atlas/p27_report.json','p32_report':'results/author_boundary_continuation/p32_report.json','p33_crossings':'results/author_expanded_boundary_search/p33_crossings.csv','p33_rows':'results/author_expanded_boundary_search/p33_rows.csv','p33_summary':'results/author_expanded_boundary_search/p33_summary.csv','p33_verifications':'results/author_expanded_boundary_search/p33_verifications.csv'}
 for k,v in defaults.items():ap.add_argument('--'+k.replace('_','-'),default=v)
 ap.add_argument('--p33-report',default=None,help='Optional P33 JSON report for additional source-hash verification')
 ap.add_argument('--output-dir',default='results/author_geometry_validation')
 ap.add_argument('--only-edge',type=int,choices=[30,34]);ap.add_argument('--tests-per-map',type=int,default=8)
 ap.add_argument('--scan-halfwidth',type=float,default=.09375,help='Degrees either side of linear prediction')
 ap.add_argument('--angular-step',type=float,default=.015625);ap.add_argument('--refine-levels',type=int,default=3)
 ap.add_argument('--long-pairs-per-map',type=int,default=2)
 ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
 ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
 ap.add_argument('--rtol',type=float,default=1e-9);ap.add_argument('--atol',type=float,default=1e-11)
 ap.add_argument('--verify-observe',type=float,default=3200.);ap.add_argument('--verify-dt',type=float,default=.25)
 ap.add_argument('--verify-rtol',type=float,default=1e-11);ap.add_argument('--verify-atol',type=float,default=1e-13)
 ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
 ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 if a.tests_per_map<1 or a.scan_halfwidth<=0 or a.angular_step<=0 or a.refine_levels<0 or a.long_pairs_per_map<0 or a.windows<4:ap.error('Invalid sampling settings')
 paths={k:Path(getattr(a,k)) for k in defaults}
 p23,p24,p27,p32=(json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report'))
 z=float(p32['zeta']);refs=p32['reference_drifts'];rows=loadcsv(paths['p33_rows']);raw=loadcsv(paths['p33_crossings']);p33summary=loadcsv(paths['p33_summary']);v33=loadcsv(paths['p33_verifications'])
 if any(int(r['n_failed']) for r in p33summary) or not all(r['reproduced'].lower()=='true' for r in v33):raise ValueError('P33 summary/verification contains failure')
 if abs(float(p27['zeta'])-z)>1e-12 or p27['reference_drifts']!=refs:raise ValueError('P27/P32 reference mismatch')
 for k in ('p23_report','p24_report','p27_report'):
  if p32['source_sha256'][k]!=digest(paths[k]):raise ValueError('P32 upstream hash mismatch: '+k)
 if a.p33_report:
  rep=json.loads(Path(a.p33_report).read_text())
  if abs(float(rep['zeta'])-z)>1e-12 or rep['reference_drifts']!=refs or rep.get('n_failed',0):raise ValueError('P33 report mismatch/failures')
  for k in ('p23_report','p24_report','p27_report','p32_report'):
   if rep['source_sha256'][k]!=digest(paths[k]):raise ValueError('P33 report upstream hash mismatch '+k)
  # The optional report can independently cross-check the uploaded CSV contents.
  if len(rep['crossings'])!=len(raw) or len(rep['rows'])!=len(rows):raise ValueError('P33 report/CSV row count mismatch')
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
 if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('Direction mismatch')
 perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
 directions={int(r['edge_index']):int(r['direction_index']) for r in p32['rows']}
 # Preserve multiplicity: only fit rows with exactly one resolved P33 crossing; never join ambiguous branches.
 usable={}
 for e in (30,34):
  rr=[r for r in raw if int(r['edge_index'])==e and r['status']=='refined' and r['class_a'] in refs and r['class_b'] in refs and r['class_a']!=r['class_b']]
  byscale={}
  for r in rr:byscale.setdefault(round(float(r['scale']),13),[]).append(r)
  usable[e]=sorted([(float(rs[0]['scale']),midpoint(rs[0]),float(rs[0]['width_degrees']),rs[0]) for rs in byscale.values() if len(rs)==1])
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 cfg=dict(source_sha256={k:digest(v) for k,v in paths.items()},p33_report_sha256=digest(a.p33_report) if a.p33_report else None,settings={k:v for k,v in vars(a).items() if k not in ('resume','output_dir')})
 cp=out/'p34_checkpoint.json';cf=out/'p34_config.json'
 if a.resume and cp.exists() and (not cf.exists() or json.loads(cf.read_text())!=cfg):raise ValueError('Resume source/settings mismatch')
 if not a.resume and cp.exists():raise ValueError('Checkpoint exists: use --resume or new output directory')
 save(cf,cfg);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];byjob={r['job_id']:r for r in records}
 def run(e,s,ang,mode='scan'):
  jid=f'e{e}_s{s:.15g}_a{ang:.15g}_{mode}'
  if jid in byjob:return byjob[jid]
  r=dict(job_id=jid,edge_index=e,scale=float(s),angle_degrees=float(ang),mode=mode,status='failed',verdict='failed')
  try:r.update(simulate(x0,p,base,perps,directions[e],s,ang,refs,z,a,'long' if mode=='long' else 'grid'))
  except Exception as ex:r['error']=repr(ex)
  records.append(r);byjob[jid]=r;save(cp,records)
  print(f'{len(records):04d} edge={e} scale={s:.11g} angle={ang:.8g} {mode} {r["verdict"]}',flush=True)
  return r
 tests=[];checks=[];summaries=[];geometry=[]
 def publish():
  report=dict(scope='P34 independently sampled radial interpolation and local finite-time transition geometry',zeta=z,reference_drifts=refs,source_sha256=cfg['source_sha256'],p33_report_sha256=cfg['p33_report_sha256'],settings=vars(a),geometry=geometry,tests=tests,verifications=checks,summary=summaries,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Linear interpolation and finite-difference slopes describe sampled crossing midpoints, not established smooth boundary curves.','Independent radial tests are selected deterministically within observed P33 gaps; crossings outside local scan can be missed.','A crossing bracket is finite angular resolution, not physical boundary thickness.','Finite-time drift classifications on selected two-dimensional slices; no global six-dimensional basin or asymptotic claim.'])
  save(out/'p34_report.json',report)
  csvout(out/'p34_geometry.csv',geometry,['edge_index','scale','angle_midpoint','width_degrees','slope_left_deg_per_scale','slope_right_deg_per_scale','curvature_proxy_deg_per_scale2'])
  csvout(out/'p34_tests.csv',tests,['edge_index','test_index','scale','source_scale_left','source_scale_right','predicted_angle','observed_angle','prediction_error_degrees','bracket_width_degrees','n_crossings','status','class_a','class_b','angle_a','angle_b'])
  csvout(out/'p34_verifications.csv',checks,['edge_index','test_index','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
  csvout(out/'p34_summary.csv',summaries,['edge_index','n_source_rows','n_test_rows','n_single_crossing','n_no_crossing','n_multi_crossing','n_ambiguous','median_abs_prediction_error_degrees','max_abs_prediction_error_degrees','n_long_checks','n_reproduced','n_complete','n_failed'])
 for e in (30,34):
  if a.only_edge is not None and e!=a.only_edge:continue
  points=usable[e]
  if len(points)<3:raise ValueError(f'Insufficient single-crossing rows for map {e}')
  for i,(s,ang,w,_) in enumerate(points):
   left=(ang-points[i-1][1])/(s-points[i-1][0]) if i else None
   right=(points[i+1][1]-ang)/(points[i+1][0]-s) if i+1<len(points) else None
   curv=2*(right-left)/(points[i+1][0]-points[i-1][0]) if left is not None and right is not None else None
   geometry.append(dict(edge_index=e,scale=s,angle_midpoint=ang,width_degrees=w,slope_left_deg_per_scale=left,slope_right_deg_per_scale=right,curvature_proxy_deg_per_scale2=curv))
  # Independent test scales: select widest P33 gaps first, then place 1 midpoint in each chosen gap.
  gaps=sorted([(points[i+1][0]-points[i][0],i) for i in range(len(points)-1)],reverse=True)
  selected=sorted(gaps[:a.tests_per_map],key=lambda q:q[1])
  maptests=[]
  for idx,(_,i) in enumerate(selected):
   s0,y0,_,_=points[i];s1,y1,_,_=points[i+1];s=(s0+s1)/2;pred=(y0+y1)/2
   grid=[pred-a.scan_halfwidth+k*a.angular_step for k in range(int(math.floor(2*a.scan_halfwidth/a.angular_step+1e-10))+1)]
   samples=[(float(ang),run(e,s,float(ang))) for ang in grid]
   found=[];amb=sum(r['verdict'] not in refs for _,r in samples)
   for (ang0,r0),(ang1,r1) in zip(samples,samples[1:]):
    if r0['verdict'] not in refs or r1['verdict'] not in refs or r0['verdict']==r1['verdict']:continue
    lo=(ang0,r0);hi=(ang1,r1);status='refined'
    for _ in range(a.refine_levels):
     mid=(lo[0]+hi[0])/2;rr=run(e,s,mid,'refine')
     if rr['verdict'] not in refs:status='ambiguous_midpoint';break
     if rr['verdict']==lo[1]['verdict']:lo=(mid,rr)
     elif rr['verdict']==hi[1]['verdict']:hi=(mid,rr)
     else:status='unexpected_class';break
    found.append((lo,hi,status))
   status='ambiguous_samples' if amb else ('no_observed_crossing' if not found else ('single_crossing' if len(found)==1 and found[0][2]=='refined' else 'multiple_or_unresolved'))
   selected_cross=min(found,key=lambda q:abs((q[0][0]+q[1][0])/2-pred)) if found else None
   obs=(selected_cross[0][0]+selected_cross[1][0])/2 if selected_cross else None
   t=dict(edge_index=e,test_index=idx,scale=s,source_scale_left=s0,source_scale_right=s1,predicted_angle=pred,observed_angle=obs,prediction_error_degrees=obs-pred if obs is not None else None,bracket_width_degrees=selected_cross[1][0]-selected_cross[0][0] if selected_cross else None,n_crossings=len(found),status=status,class_a=selected_cross[0][1]['verdict'] if selected_cross else None,class_b=selected_cross[1][1]['verdict'] if selected_cross else None,angle_a=selected_cross[0][0] if selected_cross else None,angle_b=selected_cross[1][0] if selected_cross else None)
   tests.append(t);maptests.append((t,selected_cross));publish()
  # Verify spatially spread, independently integrated endpoint pairs; do not use source P33 verifications as new evidence.
  candidates=[(t,c) for t,c in maptests if t['status']=='single_crossing']
  if candidates:
   indices=sorted(set(np.linspace(0,len(candidates)-1,min(a.long_pairs_per_map,len(candidates))).round().astype(int).tolist())) if a.long_pairs_per_map else []
   for k in indices:
    t,c=candidates[k]
    for side,(ang,r) in zip(('a','b'),c[:2]):
     vr=run(e,t['scale'],ang,'long')
     checks.append(dict(edge_index=e,test_index=t['test_index'],side=side,scale=t['scale'],angle_degrees=ang,expected_class=r['verdict'],observed_class=vr['verdict'],reproduced=vr['status']=='complete' and vr['verdict']==r['verdict'],all_windows_consistent=vr.get('all_windows_consistent',False),status=vr['status'],job_id=vr['job_id']))
  tt=[t for t in tests if t['edge_index']==e];cc=[v for v in checks if v['edge_index']==e];rr=[r for r in records if r['edge_index']==e];errs=[abs(t['prediction_error_degrees']) for t in tt if t['prediction_error_degrees'] is not None]
  summaries.append(dict(edge_index=e,n_source_rows=len(points),n_test_rows=len(tt),n_single_crossing=sum(t['status']=='single_crossing' for t in tt),n_no_crossing=sum(t['status']=='no_observed_crossing' for t in tt),n_multi_crossing=sum(t['n_crossings']>1 for t in tt),n_ambiguous=sum('ambiguous' in t['status'] or 'unresolved' in t['status'] for t in tt),median_abs_prediction_error_degrees=float(np.median(errs)) if errs else None,max_abs_prediction_error_degrees=max(errs) if errs else None,n_long_checks=len(cc),n_reproduced=sum(v['reproduced'] for v in cc),n_complete=sum(r['status']=='complete' for r in rr),n_failed=sum(r['status']!='complete' for r in rr)))
  publish()
 print(f'P34 finished: {len(tests)} independent radial tests, {len(checks)} long checks, {len(records)} new integrations; failures={sum(r["status"]!="complete" for r in records)}',flush=True)
if __name__=='__main__':main()

