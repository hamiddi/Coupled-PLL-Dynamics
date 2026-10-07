#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p35_boundary_resolution.py

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

def readcsv(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def truth(x):return str(x).lower()=='true'
def spread(n,k):
 if k>=n:return list(range(n))
 return sorted(set(int(round(x)) for x in np.linspace(0,n-1,k)))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 defaults={'p23_report':'results/author_longtime_verification/p23_report.json','p24_report':'results/author_basin_attraction/p24_report.json','p27_report':'results/author_local_boundary_atlas/p27_report.json','p32_report':'results/author_boundary_continuation/p32_report.json','p34_report':'results/author_geometry_validation/p34_report.json','p34_tests':'results/author_geometry_validation/p34_tests.csv','p34_summary':'results/author_geometry_validation/p34_summary.csv','p34_verifications':'results/author_geometry_validation/p34_verifications.csv'}
 for key,val in defaults.items():ap.add_argument('--'+key.replace('_','-'),default=val)
 ap.add_argument('--output-dir',default='results/author_boundary_resolution')
 ap.add_argument('--only-edge',type=int,choices=[30,34]);ap.add_argument('--pairs-per-map',type=int,default=4)
 ap.add_argument('--fine-levels',type=int,default=5,help='Bisections beyond P34 bracket')
 ap.add_argument('--ultralong-pairs-per-map',type=int,default=2)
 ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
 ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
 ap.add_argument('--rtol',type=float,default=1e-9);ap.add_argument('--atol',type=float,default=1e-11)
 ap.add_argument('--verify-observe',type=float,default=3200.);ap.add_argument('--ultralong-observe',type=float,default=6400.)
 ap.add_argument('--verify-dt',type=float,default=.25);ap.add_argument('--verify-rtol',type=float,default=1e-11);ap.add_argument('--verify-atol',type=float,default=1e-13)
 ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
 ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 if a.pairs_per_map<1 or a.fine_levels<0 or a.ultralong_pairs_per_map<0 or a.windows<4 or any(x<=0 for x in (a.observe,a.dt,a.rtol,a.atol,a.verify_observe,a.ultralong_observe,a.verify_dt,a.verify_rtol,a.verify_atol)):ap.error('Invalid settings')
 paths={k:Path(getattr(a,k)) for k in defaults};src={k:json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report','p34_report')}
 p23,p24,p27,p32,p34=(src[k] for k in ('p23_report','p24_report','p27_report','p32_report','p34_report'))
 z=float(p34['zeta']);refs=p34['reference_drifts'];p34tests=readcsv(paths['p34_tests']);p34sum=readcsv(paths['p34_summary']);p34checks=readcsv(paths['p34_verifications'])
 if any(int(r['n_failed']) for r in p34sum) or p34['n_failed'] or any(not truth(r['reproduced']) for r in p34checks):raise ValueError('P34 failures or unreproduced verification')
 if z!=float(p32['zeta']) or p32['reference_drifts']!=refs or p27['reference_drifts']!=refs:raise ValueError('P32/P34 references differ')
 for k in ('p23_report','p24_report','p27_report','p32_report'):
  if p34['source_sha256'][k]!=digest(paths[k]):raise ValueError('P34 source hash mismatch: '+k)
 if len(p34tests)!=len(p34['tests']):raise ValueError('P34 tests CSV/report row count mismatch')
 for x,y in zip(p34tests,p34['tests']):
  for key in ('edge_index','test_index','scale','angle_a','angle_b'):
   if abs(float(x[key])-float(y[key]))>1e-12:raise ValueError('P34 tests CSV/report mismatch: '+key)
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
 if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('P27 direction mismatch')
 perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 directions={int(r['edge_index']):int(r['direction_index']) for r in p32['rows']}
 p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 cfg=dict(source_sha256={k:digest(v) for k,v in paths.items()},settings={k:v for k,v in vars(a).items() if k not in ('resume','output_dir')})
 cp=out/'p35_checkpoint.json';cf=out/'p35_config.json'
 if a.resume and cp.exists() and (not cf.exists() or json.loads(cf.read_text())!=cfg):raise ValueError('Resume source/settings mismatch')
 if not a.resume and cp.exists():raise ValueError('Existing checkpoint: use --resume or another output dir')
 save(cf,cfg);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];byjob={r['job_id']:r for r in records}
 def run(e,s,ang,mode):
  jid=f'e{e}_s{s:.17g}_a{ang:.17g}_{mode}'
  if jid in byjob:return byjob[jid]
  r=dict(job_id=jid,edge_index=e,scale=float(s),angle_degrees=float(ang),mode=mode,status='failed',verdict='failed')
  try:
   # P32 simulate exposes one long mode; use a mode-specific shallow args clone for 6400.
   if mode=='ultralong':
    from argparse import Namespace
    aa=Namespace(**vars(a));aa.verify_observe=a.ultralong_observe
    r.update(simulate(x0,p,base,perps,directions[e],s,ang,refs,z,aa,'long'))
   else:r.update(simulate(x0,p,base,perps,directions[e],s,ang,refs,z,a,'long' if mode=='long' else 'grid'))
  except Exception as exc:r['error']=repr(exc)
  records.append(r);byjob[jid]=r;save(cp,records)
  print(f'{len(records):04d} e={e} s={s:.12g} a={ang:.10g} {mode}: {r["verdict"]}',flush=True)
  return r
 refinements=[];checks=[];slopes=[];summaries=[]
 def publish():
  rep=dict(scope='P35 finite-resolution bracket refinement and finite-time observation-horizon robustness',zeta=z,reference_drifts=refs,source_sha256=cfg['source_sha256'],settings=vars(a),refinements=refinements,verifications=checks,slopes=slopes,summary=summaries,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Selected P34 brackets only; no proof of global boundary continuity or smoothness.','Interval slope bounds propagate angular sampling uncertainty, not model or classification uncertainty.','Endpoint class reproduction at finite horizons does not establish asymptotic attractor membership.','The observed transitions concern selected two-dimensional initial-condition slices.'])
  save(out/'p35_report.json',rep)
  csvout(out/'p35_refinements.csv',refinements,['edge_index','test_index','scale','p34_angle_a','p34_angle_b','fine_angle_a','fine_angle_b','p34_width_degrees','fine_width_degrees','width_ratio','p34_midpoint','fine_midpoint','midpoint_shift_degrees','class_a','class_b','levels_completed','status'])
  csvout(out/'p35_verifications.csv',checks,['edge_index','test_index','side','horizon','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
  csvout(out/'p35_slopes.csv',slopes,['edge_index','left_test_index','right_test_index','scale_left','scale_right','p34_slope','p35_slope','p34_slope_lower','p34_slope_upper','p35_slope_lower','p35_slope_upper','intervals_overlap','p34_slope_interval_width','p35_slope_interval_width'])
  csvout(out/'p35_summary.csv',summaries,['edge_index','n_selected','n_refined','n_unresolved','median_width_ratio','max_midpoint_shift_degrees','n_slope_pairs','n_slope_interval_overlaps','n_long_checks','n_long_reproduced','n_ultralong_checks','n_ultralong_reproduced','n_complete','n_failed'])
 for e in (30,34):
  if a.only_edge is not None and e!=a.only_edge:continue
  eligible=sorted((r for r in p34tests if int(r['edge_index'])==e and r['status']=='single_crossing' and r['class_a'] in refs and r['class_b'] in refs and r['class_a']!=r['class_b']),key=lambda r:float(r['scale']))
  if not eligible:raise ValueError(f'No resolved P34 crossings for map {e}')
  selected=[eligible[i] for i in spread(len(eligible),a.pairs_per_map)]
  selected_ultra={int(selected[i]['test_index']) for i in spread(len(selected),min(a.ultralong_pairs_per_map,len(selected)))} if a.ultralong_pairs_per_map else set()
  for t in selected:
   s=float(t['scale']);lo=float(t['angle_a']);hi=float(t['angle_b']);ca=t['class_a'];cb=t['class_b'];completed=0;status='refined'
   for level in range(a.fine_levels):
    mid=(lo+hi)/2;r=run(e,s,mid,'refine')
    if r['status']!='complete' or r['verdict'] not in (ca,cb):status='unresolved_midpoint';break
    if r['verdict']==ca:lo=mid
    else:hi=mid
    completed+=1
   row=dict(edge_index=e,test_index=int(t['test_index']),scale=s,p34_angle_a=float(t['angle_a']),p34_angle_b=float(t['angle_b']),fine_angle_a=lo,fine_angle_b=hi,p34_width_degrees=float(t['angle_b'])-float(t['angle_a']),fine_width_degrees=hi-lo,width_ratio=(float(t['angle_b'])-float(t['angle_a']))/(hi-lo),p34_midpoint=(float(t['angle_a'])+float(t['angle_b']))/2,fine_midpoint=(lo+hi)/2,midpoint_shift_degrees=(lo+hi-float(t['angle_a'])-float(t['angle_b']))/2,class_a=ca,class_b=cb,levels_completed=completed,status=status)
   refinements.append(row);publish()
   if status!='refined':continue
   for side,ang,expected in (('a',lo,ca),('b',hi,cb)):
    for mode in (('long','ultralong') if int(t['test_index']) in selected_ultra else ('long',)):
     vr=run(e,s,ang,mode)
     checks.append(dict(edge_index=e,test_index=int(t['test_index']),side=side,horizon=a.ultralong_observe if mode=='ultralong' else a.verify_observe,scale=s,angle_degrees=ang,expected_class=expected,observed_class=vr['verdict'],reproduced=vr['status']=='complete' and vr['verdict']==expected,all_windows_consistent=vr.get('all_windows_consistent',False),status=vr['status'],job_id=vr['job_id']))
     publish()
  good=sorted((r for r in refinements if r['edge_index']==e and r['status']=='refined'),key=lambda r:r['scale'])
  for left,right in zip(good,good[1:]):
   ds=right['scale']-left['scale']
   def bounds(prefix):
    al=left[prefix+'_angle_a'];bl=left[prefix+'_angle_b'];ar=right[prefix+'_angle_a'];br=right[prefix+'_angle_b']
    return ((ar-bl)/ds,(br-al)/ds,((ar+br)-(al+bl))/(2*ds))
   l0,u0,m0=bounds('p34');l1,u1,m1=bounds('fine')
   slopes.append(dict(edge_index=e,left_test_index=left['test_index'],right_test_index=right['test_index'],scale_left=left['scale'],scale_right=right['scale'],p34_slope=m0,p35_slope=m1,p34_slope_lower=l0,p34_slope_upper=u0,p35_slope_lower=l1,p35_slope_upper=u1,intervals_overlap=max(l0,l1)<=min(u0,u1)+1e-10,p34_slope_interval_width=u0-l0,p35_slope_interval_width=u1-l1))
  ee=[r for r in refinements if r['edge_index']==e];vv=[r for r in checks if r['edge_index']==e];tt=[r for r in records if r['edge_index']==e];ss=[r for r in slopes if r['edge_index']==e]
  vlong=[r for r in vv if r['horizon']==a.verify_observe];vultra=[r for r in vv if r['horizon']==a.ultralong_observe]
  summaries.append(dict(edge_index=e,n_selected=len(ee),n_refined=len(good),n_unresolved=len(ee)-len(good),median_width_ratio=float(np.median([r['width_ratio'] for r in good])) if good else None,max_midpoint_shift_degrees=max((abs(r['midpoint_shift_degrees']) for r in good),default=None),n_slope_pairs=len(ss),n_slope_interval_overlaps=sum(r['intervals_overlap'] for r in ss),n_long_checks=len(vlong),n_long_reproduced=sum(r['reproduced'] for r in vlong),n_ultralong_checks=len(vultra),n_ultralong_reproduced=sum(r['reproduced'] for r in vultra),n_complete=sum(r['status']=='complete' for r in tt),n_failed=sum(r['status']!='complete' for r in tt)))
  publish()
 print(f'P35 completed: {len(refinements)} selected brackets, {len(checks)} horizon checks, {len(records)} integrations, failures={sum(r["status"]!="complete" for r in records)}',flush=True)
if __name__=='__main__':main()

