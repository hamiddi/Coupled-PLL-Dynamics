#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p50_high_resolution_boundary_verification.py

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
from p32_boundary_continuation import simulate,save,csvout,digest

def rows(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def require(v,msg):
 if not v:raise ValueError(msg)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 defaults={'p23_report':'results/author_longtime_verification/p23_report.json','p24_report':'results/author_basin_attraction/p24_report.json','p27_report':'results/author_local_boundary_atlas/p27_report.json','p32_report':'results/author_boundary_continuation/p32_report.json','p48_report':'results/author_consolidated_validation/p48_report.json','p49_report':'results/author_basin_geometry_resilience/p49_report.json','p49_grid':'results/author_basin_geometry_resilience/p49_grid.csv','p49_baselines':'results/author_basin_geometry_resilience/p49_baselines.csv','p49_occupancy':'results/author_basin_geometry_resilience/p49_occupancy.csv','p49_horizon_checks':'results/author_basin_geometry_resilience/p49_horizon_checks.csv'}
 for k,v in defaults.items():ap.add_argument('--'+k.replace('_','-'),default=v)
 ap.add_argument('--output-dir',default='results/author_high_resolution_boundary_verification')
 ap.add_argument('--only-edge',type=int,choices=(30,34));ap.add_argument('--rows-per-edge',type=int,default=4)
 ap.add_argument('--subdivisions',type=int,default=2,help='New evenly spaced interior points per P49 grid interval')
 ap.add_argument('--bisect-levels',type=int,default=7)
 ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
 ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
 ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13)
 ap.add_argument('--verify-observe',type=float,default=3200.);ap.add_argument('--extended-observe',type=float,default=6400.)
 ap.add_argument('--verify-dt',type=float,default=.25);ap.add_argument('--verify-rtol',type=float,default=1e-12);ap.add_argument('--verify-atol',type=float,default=1e-14)
 ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
 ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
 a=ap.parse_args()
 require(1<=a.rows_per_edge<=4 and 1<=a.subdivisions<=6 and 1<=a.bisect_levels<=12 and a.windows>=4,'Invalid design settings')
 require(all(getattr(a,k)>0 for k in ('observe','dt','rtol','atol','verify_observe','extended_observe','verify_dt','verify_rtol','verify_atol')),'Invalid integration settings')
 paths={k:Path(getattr(a,k)) for k in defaults}
 for k,p in paths.items():require(p.is_file(),f'Missing {k}: {p}')
 p23,p24,p27,p32,p48,p49=(json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report','p48_report','p49_report'))
 require(p49['n_failed']==0 and p49['n_complete']==360,'P49 must be complete 360/360')
 require(p48['n_failed']==0 and p48['n_complete']==156,'P48 must be complete 156/156')
 for k in ('p23_report','p24_report','p27_report','p32_report','p48_report'):
  require(p49['source_sha256'][k]==digest(paths[k]),f'P49 upstream hash mismatch: {k}')
 for k in ('p23_report','p24_report','p27_report','p32_report'):
  require(p48['source_sha256'][k]==digest(paths[k]),f'P48 upstream hash mismatch: {k}')
 require(p49['reference_drifts']==p48['reference_drifts']==p27['reference_drifts'],'Reference mismatch')
 grid=rows(paths['p49_grid']);baselines=rows(paths['p49_baselines']);occ=rows(paths['p49_occupancy']);horiz=rows(paths['p49_horizon_checks'])
 require(len(grid)==120 and len(baselines)==16 and len(occ)==8 and len(horiz)==32,'P49 CSV counts mismatch')
 require(all(r['status']=='complete' and r['all_windows_consistent']=='True' for r in grid+baselines),'P49 grid/baseline incomplete or inconsistent')
 require(all(r['reproduced']=='True' and r['windows_consistent']=='True' for r in horiz),'P49 horizon verification incomplete')
 selected=[]
 for e in (30,34):
  if a.only_edge is not None and e!=a.only_edge:continue
  selected.extend(sorted((r for r in occ if int(r['edge_index'])==e),key=lambda r:int(r['row_index']))[:a.rows_per_edge])
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(float(r['zeta'])-.899)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
 require(np.allclose(base,p27['base_direction'],atol=1e-12),'Direction mismatch')
 perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 z=float(p49['zeta']);refs=p49['reference_drifts'];p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 source_sha={k:digest(v) for k,v in paths.items()}
 settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')}
 config=dict(source_sha256=source_sha,settings=settings,selected=selected)
 cp=out/'p50_checkpoint.json';cf=out/'p50_config.json'
 if a.resume and cf.exists():require(json.loads(cf.read_text())==config,'Resume configuration mismatch')
 if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
 selected_ids={(int(r['edge_index']),int(r['row_index'])) for r in selected}
 frozen={key:sorted((r for r in grid if (int(r['edge_index']),int(r['row_index']))==key),key=lambda r:float(r['angle_degrees'])) for key in selected_ids}
 require(all(len(v)==15 for v in frozen.values()),'Missing frozen P49 grid rows')
 anomalies=[]
 for key in sorted(selected_ids):
  gg=frozen[key];changes=sum(gg[i]['verdict']!=gg[i-1]['verdict'] for i in range(1,len(gg)))
  bb=[r for r in baselines if (int(r['edge_index']),int(r['row_index']))==key]
  same=len({r['verdict'] for r in bb})==1
  anomalies.append(dict(edge_index=key[0],row_index=key[1],p49_adjacent_changes=changes,p49_baseline_same_class=same,priority=changes>1 or same))
 if a.diagnostic_only:
  save(out/'p50_preflight.json',dict(status='PASS',scope='Frozen P49 input validation; no P50 scientific integrations',n_rows=len(selected),n_frozen_grid=sum(map(len,frozen.values())),n_planned_initial=sum((len(v)-1)*a.subdivisions+len(v) for v in frozen.values()),anomalies=anomalies,source_sha256=source_sha))
  print(f'P50 preflight PASS: {len(selected)} rows, {sum(map(len,frozen.values()))} P49 grid points, {sum(x["priority"] for x in anomalies)} priority rows');return
 save(cf,config)
 records=json.loads(cp.read_text()) if a.resume and cp.exists() else []
 byid={r['job_id']:r for r in records}
 require(len(byid)==len(records),'Duplicate checkpoint job IDs')
 def run(job_id,e,ri,s,theta,kind,mode='grid',obs=None):
  if job_id in byid:return byid[job_id]
  di=int(next(r['direction_index'] for r in p32['rows'] if int(r['edge_index'])==e))
  aa=argparse.Namespace(**vars(a));aa.verify_observe=obs if obs is not None else a.verify_observe
  rec=dict(job_id=job_id,kind=kind,edge_index=e,row_index=ri,scale=s,angle_degrees=theta,status='failed',verdict='failed')
  try:rec.update(simulate(x0,p,base,perps,di,s,theta,refs,z,aa,mode))
  except Exception as exc:rec['error']=repr(exc)
  records.append(rec);byid[job_id]=rec;save(cp,records)
  print(f'{len(records)} {job_id}: {rec["verdict"]}',flush=True)
  return rec
 scans=[];crossings=[];verification=[];sensitivity=[]
 for key in sorted(frozen):
  e,ri=key;gg=frozen[key];s=float(gg[0]['scale']);points=[]
  for i,g in enumerate(gg):
   theta=float(g['angle_degrees']);r=run(f'e{e}_r{ri}_anchor_{i:02d}',e,ri,s,theta,'anchor');points.append(r)
   if i+1<len(gg):
    nxt=float(gg[i+1]['angle_degrees'])
    for j in range(1,a.subdivisions+1):
     th=theta+(nxt-theta)*j/(a.subdivisions+1)
     points.append(run(f'e{e}_r{ri}_dense_{i:02d}_{j}',e,ri,s,th,'dense'))
  points.sort(key=lambda r:r['angle_degrees'])
  for i in range(len(points)-1):
   left,right=points[i],points[i+1]
   if left['verdict'] not in refs or right['verdict'] not in refs:continue
   if left['verdict']==right['verdict']:continue
   lo,hi=left,right
   for level in range(a.bisect_levels):
    mid=(lo['angle_degrees']+hi['angle_degrees'])/2
    r=run(f'e{e}_r{ri}_change_{i:02d}_bisect_{level}',e,ri,s,mid,'bisect')
    if r['verdict'] not in refs:break
    if r['verdict']==lo['verdict']:lo=r
    elif r['verdict']==hi['verdict']:hi=r
    else:break
   resolved=lo['verdict'] in refs and hi['verdict'] in refs and lo['verdict']!=hi['verdict']
   item=dict(edge_index=e,row_index=ri,scale=s,change_index=i,angle_a=lo['angle_degrees'],angle_b=hi['angle_degrees'],width_degrees=hi['angle_degrees']-lo['angle_degrees'],left_class=lo['verdict'],right_class=hi['verdict'],resolved=resolved)
   crossings.append(item)
   for side,endpoint in (('left',lo),('right',hi)):
    for name,ob in (('long',a.verify_observe),('extended',a.extended_observe)):
     r=run(f'e{e}_r{ri}_change_{i:02d}_{name}_{side}',e,ri,s,endpoint['angle_degrees'],name,'long',ob)
     verification.append(dict(edge_index=e,row_index=ri,change_index=i,side=side,horizon=name,expected=endpoint['verdict'],observed=r['verdict'],reproduced=r['verdict']==endpoint['verdict'] and r['verdict'] in refs,all_windows_consistent=r.get('all_windows_consistent',False),status=r['status']))
   # Two tighter-tolerance checks on each side of the refined transition, same observation horizon.
   for side,endpoint in (('left',lo),('right',hi)):
    aa=argparse.Namespace(**vars(a));aa.rtol=a.verify_rtol;aa.atol=a.verify_atol;aa.dt=a.verify_dt
    jid=f'e{e}_r{ri}_change_{i:02d}_tight_{side}'
    if jid in byid:r=byid[jid]
    else:
     di=int(next(rr['direction_index'] for rr in p32['rows'] if int(rr['edge_index'])==e))
     r=dict(job_id=jid,kind='tight',edge_index=e,row_index=ri,scale=s,angle_degrees=endpoint['angle_degrees'],status='failed',verdict='failed')
     try:r.update(simulate(x0,p,base,perps,di,s,endpoint['angle_degrees'],refs,z,aa,'grid'))
     except Exception as exc:r['error']=repr(exc)
     records.append(r);byid[jid]=r;save(cp,records);print(f'{len(records)} {jid}: {r["verdict"]}',flush=True)
    sensitivity.append(dict(edge_index=e,row_index=ri,change_index=i,side=side,expected=endpoint['verdict'],observed=r['verdict'],reproduced=r['verdict']==endpoint['verdict'] and r['verdict'] in refs,all_windows_consistent=r.get('all_windows_consistent',False),status=r['status']))
  scans.append(dict(edge_index=e,row_index=ri,scale=s,n_p49_grid=len(gg),n_p50_dense=len(points),n_p49_changes=next(x['p49_adjacent_changes'] for x in anomalies if (x['edge_index'],x['row_index'])==key),n_p50_changes=sum(points[i]['verdict']!=points[i-1]['verdict'] and points[i]['verdict'] in refs and points[i-1]['verdict'] in refs for i in range(1,len(points))),n_unresolved=sum(r['verdict'] not in refs for r in points),n_refined=sum(c['edge_index']==e and c['row_index']==ri for c in crossings),p49_baseline_same_class=next(x['p49_baseline_same_class'] for x in anomalies if (x['edge_index'],x['row_index'])==key)))
 common=['job_id','kind','edge_index','row_index','scale','angle_degrees','status','verdict','all_windows_consistent','observe','dt','rtol','atol','error']
 csvout(out/'p50_samples.csv',[r for r in records if r['kind'] in ('anchor','dense','bisect')],common)
 csvout(out/'p50_crossings.csv',crossings,['edge_index','row_index','scale','change_index','angle_a','angle_b','width_degrees','left_class','right_class','resolved'])
 csvout(out/'p50_verifications.csv',verification,['edge_index','row_index','change_index','side','horizon','expected','observed','reproduced','all_windows_consistent','status'])
 csvout(out/'p50_tolerance_checks.csv',sensitivity,['edge_index','row_index','change_index','side','expected','observed','reproduced','all_windows_consistent','status'])
 csvout(out/'p50_summary.csv',scans,['edge_index','row_index','scale','n_p49_grid','n_p50_dense','n_p49_changes','n_p50_changes','n_unresolved','n_refined','p49_baseline_same_class'])
 failed=sum(r['status']!='complete' for r in records)
 report=dict(scope='P50 local high-resolution finite-time verification of P49 multiple class changes',zeta=z,source_sha256=source_sha,settings=settings,anomalies=anomalies,summary=scans,crossings=crossings,verifications=verification,tolerance_checks=sensitivity,n_complete=len(records)-failed,n_failed=failed,n_refined=len(crossings),n_verifications=len(verification),n_tolerance_checks=len(sensitivity),limitations=['Only selected 1D angular rows on two 2D initial-condition slices; no global basin topology inference.','A finite grid can miss sub-grid transitions; refinement targets observed changes only.','Finite-time classifications and longer integrations do not establish asymptotic stability or fractal dimension.','P49 grid values were re-integrated with tighter tolerances; P49 classifications were not copied into P50 results.'])
 save(out/'p50_report.json',report)
 print(f'P50 finished: {len(records)} integrations; {failed} failures; {len(crossings)} refined transitions; {len(verification)} horizon checks',flush=True)
if __name__=='__main__':main()

