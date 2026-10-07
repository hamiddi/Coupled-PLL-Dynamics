#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p32_boundary_continuation.py

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
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate,classify

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def csvout(path,rows,fields):
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def simulate(x0,p,base,perp,direction,s,ang,refs,z,a,mode):
 vec=math.cos(math.radians(ang))*base+math.sin(math.radians(ang))*perp[direction]
 x=x0+s*vec
 long=mode=='long';dt=a.verify_dt if long else a.dt;rt=a.verify_rtol if long else a.rtol;at=a.verify_atol if long else a.atol;obs=a.verify_observe if long else a.observe
 if a.transient:x=integrate(x,p,a.transient,dt,rt,at,a.max_abs)[1][-1]
 t,Y=integrate(x,p,obs,dt,rt,at,a.max_abs)
 windows=[]
 for k in range(a.windows):
  lo=int(k*(len(t)-1)/a.windows);hi=int((k+1)*(len(t)-1)/a.windows)+1
  windows.append(dict(window=k,**classify(t[lo:hi],Y[lo:hi],z,refs,a.drift_tolerance)))
 classes=[w['classification'] for w in windows]
 verdict=classes[-1] if len(set(classes[-4:]))==1 and classes[-1] in refs else 'ambiguous'
 return dict(status='complete',verdict=verdict,all_windows_consistent=len(set(classes))==1,windows=windows,final_state=Y[-1].tolist(),observe=obs,dt=dt,rtol=rt,atol=at)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 defaults={'p23_report':'results/author_longtime_verification/p23_report.json','p24_report':'results/author_basin_attraction/p24_report.json','p27_report':'results/author_local_boundary_atlas/p27_report.json','p28_report':'results/author_adaptive_angular_refinement/p28_report.json','p29_report':'results/author_boundary_convergence/p29_report.json','p30_report':'results/author_local_2d_boundary/p30_report.json','p31_report':'results/author_adaptive_2d_convergence/p31_report.json'}
 for k,v in defaults.items():ap.add_argument('--'+k.replace('_','-'),default=v)
 ap.add_argument('--output-dir',default='results/author_boundary_continuation')
 ap.add_argument('--only-edge',type=int,choices=[13,23,30,34])
 ap.add_argument('--midrows-per-map',type=int,default=3,help='Interleaved radial rows per map, spread across the six coarse radial gaps; 0 disables')
 ap.add_argument('--levels',type=int,default=3,help='Angular bisections on every observed coarse/interleaved angular change')
 ap.add_argument('--long-pairs-per-map',type=int,default=2,help='Spatially separated refined angular transitions per map; both endpoints checked')
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
 if a.midrows_per_map<0 or a.midrows_per_map>6 or a.levels<1 or a.long_pairs_per_map<0 or a.windows<4 or any(v<=0 for v in (a.observe,a.dt,a.rtol,a.atol,a.verify_observe,a.verify_dt,a.verify_rtol,a.verify_atol)):ap.error('Invalid settings')
 paths={k:Path(getattr(a,k)) for k in defaults};src={k:json.loads(v.read_text()) for k,v in paths.items()}
 p23,p24,p27,p28,p29,p30,p31=(src[k] for k in defaults)
 z=float(p30['zeta']);refs=p30['reference_drifts']
 if p30['n_failed'] or p31['n_failed']:raise ValueError('P30/P31 contains failed integrations')
 if p30['reference_drifts']!=p31['reference_drifts'] or p30['reference_drifts']!=p27['reference_drifts']:raise ValueError('Reference drifts differ')
 for k in ('p27_report','p28_report','p29_report','p31_report'):
  if abs(float(src[k]['zeta'])-z)>1e-12:raise ValueError('zeta mismatch '+k)
 for k in ('p23_report','p24_report','p27_report','p28_report','p29_report'):
  if p30['source_sha256'][k]!=digest(paths[k]):raise ValueError('P30 source hash mismatch '+k)
 for k in ('p23_report','p24_report','p27_report','p28_report','p29_report','p30_report'):
  if p31['source_sha256'][k]!=digest(paths[k]):raise ValueError('P31 source hash mismatch '+k)
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(r['zeta']-z)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
 if not np.allclose(base,p27['base_direction'],atol=1e-12):raise ValueError('Base direction mismatch')
 perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 config=dict(source_sha256={k:digest(v) for k,v in paths.items()},settings={k:v for k,v in vars(a).items() if k not in ('resume','output_dir')})
 cp=out/'p32_checkpoint.json';cfg=out/'p32_config.json'
 if a.resume and cp.exists() and (not cfg.exists() or json.loads(cfg.read_text())!=config):raise ValueError('Resume source/settings mismatch')
 if not a.resume and cp.exists():raise ValueError('Checkpoint exists: --resume or new output directory')
 save(cfg,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];byjob={r['job_id']:r for r in records}
 def run(e,di,s,ang,mode='grid'):
  # Canonical float coordinates ensure shared points are evaluated once.
  jid=f'e{e}_s{s:.15g}_a{ang:.15g}_{mode}'
  if jid in byjob:return byjob[jid]
  r=dict(job_id=jid,edge_index=e,direction_index=di,scale=float(s),angle_degrees=float(ang),mode=mode,status='failed',verdict='failed')
  try:r.update(simulate(x0,p,base,perps,di,s,ang,refs,z,a,mode))
  except Exception as exc:r['error']=repr(exc)
  records.append(r);byjob[jid]=r;save(cp,records)
  print(f'{len(records):04d} edge={e} scale={s:.10g} angle={ang:.8g} {mode}: {r["verdict"]}',flush=True)
  return r
 # Baseline P30 samples are reused, not re-integrated; P31 angular brackets are independently overlaid.
 p30grid={(int(r['edge_index']),int(r['scale_index']),int(r['angle_index'])):r for r in p30['grid']}
 p31angles=[r for r in p31['refinements'] if r['axis']=='angle' and r['status']=='resolved_on_sampled_line']
 rows=[];crossings=[];checks=[];summaries=[]
 def publish():
  report=dict(scope='P32 2D finite-time class-boundary continuation and spatial-resolution comparison',zeta=z,reference_drifts=refs,source_sha256=config['source_sha256'],settings=vars(a),rows=rows,crossings=crossings,verifications=checks,summary=summaries,trajectories=records,n_complete=sum(r['status']=='complete' for r in records),n_failed=sum(r['status']!='complete' for r in records),limitations=['Finite-time drift classes only; no asymptotic basin or uniqueness proof.','Each radial row is independently scanned; crossing order can change or crossings can disappear between rows.','Spatial refinement is finite and cannot rule out unsampled subgrid structure.','The two-dimensional slices do not characterize the full six-dimensional basin boundary.'])
  save(out/'p32_report.json',report)
  csvout(out/'p32_rows.csv',rows,['edge_index','direction_index','row_type','row_index','scale','n_samples','n_crossings','n_ambiguous','status','crossing_midpoints_degrees'])
  csvout(out/'p32_crossings.csv',crossings,['edge_index','direction_index','row_type','row_index','scale','crossing_index','angle_a','angle_b','class_a','class_b','width_degrees','n_levels','status','p31_nearest_midpoint_degrees','p31_nearest_distance_degrees'])
  csvout(out/'p32_verifications.csv',checks,['edge_index','row_type','row_index','crossing_index','side','scale','angle_degrees','expected_class','observed_class','reproduced','all_windows_consistent','status','job_id'])
  csvout(out/'p32_summary.csv',summaries,['edge_index','n_coarse_rows','n_interleaved_rows','n_coarse_crossings','n_interleaved_crossings','n_ambiguous_rows','n_no_crossing_rows','n_refined_crossings','n_long_checks','n_reproduced','n_complete','n_failed'])
 for sm in p30['summary']:
  e=int(sm['edge_index']);di=int(sm['direction_index'])
  if a.only_edge is not None and e!=a.only_edge:continue
  grid=[r for r in p30['grid'] if int(r['edge_index'])==e]
  scales=sorted(set(float(r['scale']) for r in grid));angles=sorted(set(float(r['angle_degrees']) for r in grid))
  # Deterministic spread across all six radial gaps, including the endpoints.
  gaps=[] if a.midrows_per_map==0 else sorted(set(int(round(v)) for v in np.linspace(0,5,a.midrows_per_map)))
  row_specs=[('coarse',i,s) for i,s in enumerate(scales)]+[('interleaved',i,(scales[i]+scales[i+1])/2) for i in gaps]
  row_specs.sort(key=lambda x:x[2]);mapcross=[]
  for typ,ri,s in row_specs:
   samples=[]
   for ai,ang in enumerate(angles):
    r=p30grid[(e,ri,ai)] if typ=='coarse' else run(e,di,s,ang)
    samples.append(dict(angle=ang,verdict=r['verdict'],job_id=r['job_id']))
   found=[]
   for ai in range(len(samples)-1):
    left,right=samples[ai],samples[ai+1]
    if left['verdict'] not in refs or right['verdict'] not in refs or left['verdict']==right['verdict']:continue
    lo,hi=left.copy(),right.copy();status='refined';done=0
    for level in range(a.levels):
     mid=(lo['angle']+hi['angle'])/2;r=run(e,di,s,mid,'refine');v=r['verdict']
     if v not in refs:status='ambiguous_midpoint';break
     if v==lo['verdict']:lo=dict(angle=mid,verdict=v,job_id=r['job_id'])
     elif v==hi['verdict']:hi=dict(angle=mid,verdict=v,job_id=r['job_id'])
     else:status='unexpected_class';break
     done+=1
    midpoint=(lo['angle']+hi['angle'])/2
    anchors=[(abs((float(r['angle_a'])+float(r['angle_b']))/2-midpoint),(float(r['angle_a'])+float(r['angle_b']))/2) for r in p31angles if int(r['edge_index'])==e and abs(float(r['scale_a'])-s)<1e-12]
    nearest=min(anchors) if anchors else (None,None)
    cross=dict(edge_index=e,direction_index=di,row_type=typ,row_index=ri,scale=s,crossing_index=len(found),angle_a=lo['angle'],angle_b=hi['angle'],class_a=lo['verdict'],class_b=hi['verdict'],width_degrees=hi['angle']-lo['angle'],n_levels=done,status=status,p31_nearest_midpoint_degrees=nearest[1],p31_nearest_distance_degrees=nearest[0],endpoints=[lo,hi])
    found.append(cross);mapcross.append(cross)
    crossings.append({k:v for k,v in cross.items() if k!='endpoints'})
   namb=sum(v['verdict'] not in refs for v in samples)
   rows.append(dict(edge_index=e,direction_index=di,row_type=typ,row_index=ri,scale=s,n_samples=len(samples),n_crossings=len(found),n_ambiguous=namb,status='ambiguous_samples' if namb else ('no_observed_crossing' if not found else 'observed_crossings'),crossing_midpoints_degrees=[(c['angle_a']+c['angle_b'])/2 for c in found],samples=samples))
   publish()
  # Select spatially distributed crossing brackets; long-check both endpoints.
  valid=[c for c in mapcross if c['status']=='refined']
  if valid and a.long_pairs_per_map:
   selected=[valid[i] for i in sorted(set(int(round(x)) for x in np.linspace(0,len(valid)-1,min(a.long_pairs_per_map,len(valid)))))]
   for c in selected:
    for side,pt in zip(('a','b'),c['endpoints']):
     r=run(e,di,c['scale'],pt['angle'],'long')
     checks.append(dict(edge_index=e,row_type=c['row_type'],row_index=c['row_index'],crossing_index=c['crossing_index'],side=side,scale=c['scale'],angle_degrees=pt['angle'],expected_class=pt['verdict'],observed_class=r['verdict'],reproduced=r['verdict']==pt['verdict'] and r['status']=='complete',all_windows_consistent=r.get('all_windows_consistent',False),status=r['status'],job_id=r['job_id']))
  rr=[r for r in rows if r['edge_index']==e];cc=[c for c in crossings if c['edge_index']==e];vv=[v for v in checks if v['edge_index']==e];tr=[r for r in records if r['edge_index']==e]
  summaries.append(dict(edge_index=e,n_coarse_rows=sum(r['row_type']=='coarse' for r in rr),n_interleaved_rows=sum(r['row_type']=='interleaved' for r in rr),n_coarse_crossings=sum(c['row_type']=='coarse' for c in cc),n_interleaved_crossings=sum(c['row_type']=='interleaved' for c in cc),n_ambiguous_rows=sum(r['n_ambiguous']>0 for r in rr),n_no_crossing_rows=sum(r['n_crossings']==0 for r in rr),n_refined_crossings=sum(c['status']=='refined' for c in cc),n_long_checks=len(vv),n_reproduced=sum(v['reproduced'] for v in vv),n_complete=sum(r['status']=='complete' for r in tr),n_failed=sum(r['status']!='complete' for r in tr)))
  publish()
 print(f'P32 finished: {len(rows)} rows, {len(crossings)} observed crossings, {len(checks)} long checks, {len(records)} new integrations; failed={sum(r["status"]!="complete" for r in records)}',flush=True)
if __name__=='__main__':main()

