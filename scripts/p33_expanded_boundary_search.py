#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p33_expanded_boundary_search.py

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
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p32_boundary_continuation import simulate, digest, save, csvout

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 defaults={'p23_report':'results/author_longtime_verification/p23_report.json','p24_report':'results/author_basin_attraction/p24_report.json','p27_report':'results/author_local_boundary_atlas/p27_report.json','p28_report':'results/author_adaptive_angular_refinement/p28_report.json','p29_report':'results/author_boundary_convergence/p29_report.json','p30_report':'results/author_local_2d_boundary/p30_report.json','p31_report':'results/author_adaptive_2d_convergence/p31_report.json','p32_report':'results/author_boundary_continuation/p32_report.json'}
 for k,v in defaults.items():ap.add_argument('--'+k.replace('_','-'),default=v)
 ap.add_argument('--output-dir',default='results/author_expanded_boundary_search')
 ap.add_argument('--only-edge',type=int,choices=[30,34])
 ap.add_argument('--extension-degrees',type=float,default=.75,help='Extension beyond EACH original angular endpoint')
 ap.add_argument('--angular-step-degrees',type=float,default=.03125)
 ap.add_argument('--radial-levels',type=int,default=2,help='Nested radial midpoints across each crossing/no-crossing adjacent P32 row pair')
 ap.add_argument('--refine-levels',type=int,default=3)
 ap.add_argument('--long-pairs-per-map',type=int,default=2)
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
 if a.extension_degrees<=0 or a.angular_step_degrees<=0 or a.radial_levels<0 or a.refine_levels<0 or a.long_pairs_per_map<0 or a.windows<4 or any(v<=0 for v in (a.observe,a.dt,a.rtol,a.atol,a.verify_observe,a.verify_dt,a.verify_rtol,a.verify_atol)):ap.error('Invalid settings')
 paths={k:Path(getattr(a,k)) for k in defaults};src={k:json.loads(v.read_text()) for k,v in paths.items()}
 p23,p24,p27,p28,p29,p30,p31,p32=(src[k] for k in defaults)
 z=float(p32['zeta']);refs=p32['reference_drifts']
 if any(src[k].get('n_failed',0) for k in ('p30_report','p31_report','p32_report')):raise ValueError('Upstream experiment contains failed integrations')
 for k in ('p27_report','p28_report','p29_report','p30_report','p31_report'):
  if abs(float(src[k]['zeta'])-z)>1e-12 or src[k]['reference_drifts']!=refs:raise ValueError('zeta/reference mismatch '+k)
 for k in ('p23_report','p24_report','p27_report','p28_report','p29_report','p30_report','p31_report'):
  if p32['source_sha256'][k]!=digest(paths[k]):raise ValueError('P32 source hash mismatch '+k)
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
 if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('Base direction mismatch')
 perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 config=dict(source_sha256={k:digest(v) for k,v in paths.items()},settings={k:v for k,v in vars(a).items() if k not in ('resume','output_dir')})
 cp=out/'p33_checkpoint.json';cfg=out/'p33_config.json'
 if a.resume and cp.exists() and (not cfg.exists() or json.loads(cfg.read_text())!=config):raise ValueError('Resume source/settings mismatch')
 if not a.resume and cp.exists():raise ValueError('Checkpoint exists: use --resume or a new output directory')
 save(cfg,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];byjob={r['job_id']:r for r in records}
 def run(e,di,s,ang,mode='scan'):
  jid=f'e{e}_s{s:.15g}_a{ang:.15g}_{mode}'
  if jid in byjob:return byjob[jid]
  r=dict(job_id=jid,edge_index=e,direction_index=di,scale=float(s),angle_degrees=float(ang),mode=mode,status='failed',verdict='failed')
  try:r.update(simulate(x0,p,base,perps,di,s,ang,refs,z,a,'long' if mode=='long' else 'grid'))
  except Exception as exc:r['error']=repr(exc)
  records.append(r);byjob[jid]=r;save(cp,records)
  print(f'{len(records):04d} e={e} scale={s:.10g} angle={ang:.8g} {mode}: {r["verdict"]}',flush=True)
  return r
 # Reuse original P32 samples on unchanged radial rows, and all newly computed points on resume.
 original={(int(r['edge_index']),round(float(r['scale']),14),round(float(v['angle']),12)):v for r in p32['rows'] for v in r['samples']}
 p32rows={e:sorted((r for r in p32['rows'] if r['edge_index']==e),key=lambda r:r['scale']) for e in (30,34)}
 rows=[];crossings=[];verifications=[];summary=[]
 def publish():
  report=dict(scope='P33 expanded angular search and radial refinement around missing P32 crossings',zeta=z,reference_drifts=refs,source_sha256=config['source_sha256'],settings=vars(a),rows=rows,crossings=crossings,verifications=verifications,summary=summary,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Finite-time drift classification on selected 2D initial-condition slices only.','No crossing within an expanded interval does not prove absence of a boundary.','Multiple sampled crossings are retained; nested refinement cannot rule out subgrid crossings.','Crossings at different radial scales are not proven to belong to a single connected branch.'])
  save(out/'p33_report.json',report)
  csvout(out/'p33_rows.csv',rows,['edge_index','row_type','scale','p32_crossings','n_expanded_crossings','n_ambiguous','status','crossing_midpoints_degrees','leftmost_class','rightmost_class'])
  csvout(out/'p33_crossings.csv',crossings,['edge_index','row_type','scale','crossing_index','angle_a','angle_b','class_a','class_b','width_degrees','n_levels','status','outside_p32_window'])
  csvout(out/'p33_verifications.csv',verifications,['edge_index','row_type','scale','crossing_index','side','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
  csvout(out/'p33_summary.csv',summary,['edge_index','p32_missing_rows','recovered_missing_rows','still_missing_rows','adaptive_rows','adaptive_rows_with_crossings','n_crossings','outside_original_window','n_long_checks','n_reproduced','n_complete','n_failed'])
 def scan(e,di,s,typ,p32n,amin,amax,original_only=False):
  # All original P32 angular samples are reused, including P32 interleaved rows.
  # Expanded interval is sampled on the original P30 angular lattice.
  kmin=math.ceil((amin-a.extension_degrees-amin)/a.angular_step_degrees-1e-10)
  kmax=math.floor((amax+a.extension_degrees-amin)/a.angular_step_degrees+1e-10)
  grid=[round(amin+k*a.angular_step_degrees,12) for k in range(kmin,kmax+1)]
  # For existing P32 rows, reuse original samples; for new rows, scan original window too.
  old=[(float(v['angle']),v) for r in p32rows[e] if abs(float(r['scale'])-s)<1e-13 for v in r['samples']]
  byangle={round(x,12):v for x,v in old}
  for ang in grid:
   if round(ang,12) in byangle:continue
   r=run(e,di,s,ang)
   byangle[round(ang,12)]=dict(angle=ang,verdict=r['verdict'],job_id=r['job_id'])
  samples=sorted(byangle.values(),key=lambda x:x['angle'])
  found=[]
  for left,right in zip(samples,samples[1:]):
   if left['verdict'] not in refs or right['verdict'] not in refs or left['verdict']==right['verdict']:continue
   lo=left.copy();hi=right.copy();status='refined';done=0
   for _ in range(a.refine_levels):
    ang=(lo['angle']+hi['angle'])/2;r=run(e,di,s,ang,'refine');v=r['verdict']
    if v not in refs:status='ambiguous_midpoint';break
    if v==lo['verdict']:lo=dict(angle=ang,verdict=v,job_id=r['job_id'])
    elif v==hi['verdict']:hi=dict(angle=ang,verdict=v,job_id=r['job_id'])
    else:status='unexpected_class';break
    done+=1
   mid=(lo['angle']+hi['angle'])/2
   found.append(dict(edge_index=e,row_type=typ,scale=s,crossing_index=len(found),angle_a=lo['angle'],angle_b=hi['angle'],class_a=lo['verdict'],class_b=hi['verdict'],width_degrees=hi['angle']-lo['angle'],n_levels=done,status=status,outside_p32_window=not(amin<=mid<=amax),endpoints=[lo,hi]))
  crossings.extend({k:v for k,v in c.items() if k!='endpoints'} for c in found)
  namb=sum(v['verdict'] not in refs for v in samples)
  rows.append(dict(edge_index=e,row_type=typ,scale=s,p32_crossings=p32n,n_expanded_crossings=len(found),n_ambiguous=namb,status='ambiguous_samples' if namb else ('no_observed_crossing' if not found else 'observed_crossings'),crossing_midpoints_degrees=[(c['angle_a']+c['angle_b'])/2 for c in found],leftmost_class=samples[0]['verdict'],rightmost_class=samples[-1]['verdict'],samples=samples))
  publish();return found
 for e in (30,34):
  if a.only_edge is not None and a.only_edge!=e:continue
  oldrows=p32rows[e];di=int(oldrows[0]['direction_index']);angles=[v['angle'] for v in oldrows[0]['samples']];amin=min(angles);amax=max(angles)
  missing=[r for r in oldrows if r['n_crossings']==0]
  # Expand all original rows, not just missing ones: detect changes in crossing count and additional branches.
  mapcross=[]
  for r in oldrows:
   mapcross+=scan(e,di,float(r['scale']),'p32_row',int(r['n_crossings']),amin,amax)
  # Insert nested radial rows only at ORIGINAL P32 presence/no-presence interfaces.
  boundaries=[(x,y) for x,y in zip(oldrows,oldrows[1:]) if bool(x['n_crossings'])!=bool(y['n_crossings'])]
  adaptive=[]
  for x,y in boundaries:
   lo=float(x['scale']);hi=float(y['scale'])
   for level in range(1,a.radial_levels+1):
    # dyadic internal points, excluding points from previous levels
    for k in range(1,2**level,2):adaptive.append((lo+(hi-lo)*k/(2**level),f'adaptive_L{level}'))
  for s,typ in sorted(adaptive):mapcross+=scan(e,di,s,typ,None,amin,amax)
  valid=[c for c in mapcross if c['status']=='refined']
  # Spread verification across radial range, preferentially covering recovered original missing rows.
  recovered=[c for c in valid if c['row_type']=='p32_row' and any(abs(c['scale']-r['scale'])<1e-13 for r in missing)]
  selected=[]
  if recovered:selected.append(recovered[0]);
  if recovered and len(recovered)>1 and a.long_pairs_per_map>1:selected.append(recovered[-1])
  remaining=[c for c in valid if c not in selected]
  while len(selected)<a.long_pairs_per_map and remaining:
   if not selected:pick=remaining[len(remaining)//2]
   else:pick=max(remaining,key=lambda c:min(abs(c['scale']-v['scale']) for v in selected))
   selected.append(pick);remaining.remove(pick)
  for c in selected:
   for side,pt in zip(('a','b'),c['endpoints']):
    r=run(e,di,c['scale'],pt['angle'],'long')
    verifications.append(dict(edge_index=e,row_type=c['row_type'],scale=c['scale'],crossing_index=c['crossing_index'],side=side,angle_degrees=pt['angle'],expected_class=pt['verdict'],observed_class=r['verdict'],reproduced=r['verdict']==pt['verdict'] and r['status']=='complete',all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
  rr=[r for r in rows if r['edge_index']==e];cc=[c for c in crossings if c['edge_index']==e];vv=[v for v in verifications if v['edge_index']==e];tr=[r for r in records if r['edge_index']==e]
  recovered_count=sum(any(r['n_expanded_crossings'] for r in rr if r['row_type']=='p32_row' and abs(r['scale']-old['scale'])<1e-13) for old in missing)
  summary.append(dict(edge_index=e,p32_missing_rows=len(missing),recovered_missing_rows=recovered_count,still_missing_rows=len(missing)-recovered_count,adaptive_rows=len(adaptive),adaptive_rows_with_crossings=sum(r['n_expanded_crossings']>0 for r in rr if r['row_type'].startswith('adaptive')),n_crossings=len(cc),outside_original_window=sum(c['outside_p32_window'] for c in cc),n_long_checks=len(vv),n_reproduced=sum(v['reproduced'] for v in vv),n_complete=sum(r['status']=='complete' for r in tr),n_failed=sum(r['status']!='complete' for r in tr)))
  publish()
 print(f'P33 finished: {len(rows)} rows, {len(crossings)} observed transitions, {len(verifications)} long checks, {len(records)} new integrations; failed={sum(r["status"]!="complete" for r in records)}',flush=True)
if __name__=='__main__':main()

