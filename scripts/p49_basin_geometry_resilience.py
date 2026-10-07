#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p49_basin_geometry_resilience.py

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
from p32_boundary_continuation import simulate,save,csvout,digest

GRID_OFFSETS=(-.06,-.04,-.025,-.015,-.009,-.005,-.002,0.,.002,.005,.009,.015,.025,.04,.06)
ANGULAR_LEVELS=(.002,.006,.012)
RADIAL_LEVELS=(2e-6,8e-6,2e-5)

def loadcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def check(flag,msg):
 if not flag:raise ValueError(msg)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
 ap.add_argument('--p24-report',default='results/author_basin_attraction/p24_report.json')
 ap.add_argument('--p27-report',default='results/author_local_boundary_atlas/p27_report.json')
 ap.add_argument('--p32-report',default='results/author_boundary_continuation/p32_report.json')
 ap.add_argument('--p48-report',default='results/author_consolidated_validation/p48_report.json')
 ap.add_argument('--p48-crossings',default='results/author_consolidated_validation/p48_crossings.csv')
 ap.add_argument('--p48-verifications',default='results/author_consolidated_validation/p48_verifications.csv')
 ap.add_argument('--p48-extended-verifications',default='results/author_consolidated_validation/p48_extended_verifications.csv')
 ap.add_argument('--output-dir',default='results/author_basin_geometry_resilience')
 ap.add_argument('--only-edge',type=int,choices=(30,34))
 ap.add_argument('--rows-per-edge',type=int,default=4)
 ap.add_argument('--grid-points',type=int,default=15,help='Symmetric subset of frozen angular offsets; full experiment uses 15')
 ap.add_argument('--angular-levels',default=','.join(map(str,ANGULAR_LEVELS)))
 ap.add_argument('--radial-levels',default=','.join(map(str,RADIAL_LEVELS)))
 ap.add_argument('--baseline-offset',type=float,default=.004,help='Angular distance in degrees on either side of P48 crossing')
 ap.add_argument('--transient',type=float,default=400.)
 ap.add_argument('--observe',type=float,default=1600.)
 ap.add_argument('--windows',type=int,default=8)
 ap.add_argument('--dt',type=float,default=.5)
 ap.add_argument('--rtol',type=float,default=1e-9)
 ap.add_argument('--atol',type=float,default=1e-11)
 ap.add_argument('--verify-observe',type=float,default=3200.)
 ap.add_argument('--extended-observe',type=float,default=6400.)
 ap.add_argument('--verify-dt',type=float,default=.25)
 ap.add_argument('--verify-rtol',type=float,default=1e-11)
 ap.add_argument('--verify-atol',type=float,default=1e-13)
 ap.add_argument('--drift-tolerance',type=float,default=.08)
 ap.add_argument('--max-abs',type=float,default=1e7)
 ap.add_argument('--diagnostic-only',action='store_true')
 ap.add_argument('--resume',action='store_true')
 a=ap.parse_args()
 check(1<=a.rows_per_edge<=4 and 3<=a.grid_points<=15 and a.grid_points%2==1,'rows-per-edge 1..4; grid-points odd 3..15')
 check(a.windows>=4 and a.baseline_offset>0 and all(getattr(a,k)>0 for k in ('observe','dt','rtol','atol','verify_observe','extended_observe','verify_dt','verify_rtol','verify_atol')),'Invalid integration parameters')
 angles=[float(v) for v in a.angular_levels.split(',')];radials=[float(v) for v in a.radial_levels.split(',')]
 check(bool(angles) and bool(radials) and all(v>0 for v in angles+radials),'Perturbation magnitudes must be positive')
 paths={k:Path(getattr(a,k)) for k in ('p23_report','p24_report','p27_report','p32_report','p48_report','p48_crossings','p48_verifications','p48_extended_verifications')}
 for k,v in paths.items():check(v.is_file(),f'Missing source {k}: {v}')
 p23,p24,p27,p32,p48=(json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report','p48_report'))
 for k in ('p23_report','p24_report','p27_report','p32_report'):check(digest(paths[k])==p48['source_sha256'][k],f'P48 provenance mismatch: {k}')
 check(p48['n_failed']==0 and p48['n_complete']==156,'Expected complete P48 156/156 run')
 check(abs(float(p48['zeta'])-.899)<1e-12 and p48['reference_drifts']==p27['reference_drifts'],'P48/P27 reference mismatch')
 crossings=loadcsv(paths['p48_crossings']);vv=loadcsv(paths['p48_verifications']);ee=loadcsv(paths['p48_extended_verifications'])
 check(len(crossings)==8 and len(vv)==16 and len(ee)==16,'Incomplete P48 input CSVs')
 check(all(r['status']=='refined' and int(r['n_crossings'])==1 and int(r['n_ambiguous'])==0 for r in crossings),'P48 crossing not uniquely refined in local scan')
 check(all(r['reproduced'].lower()=='true' and r['all_windows_consistent'].lower()=='true' for r in vv+ee),'P48 endpoint verification failure')
 for c,j in zip(crossings,p48['crossings']):
  for k in ('edge_index','scale','angle_a','angle_b'):check(abs(float(c[k])-float(j[k]))<1e-11,'P48 CSV/report crossing mismatch: '+k)
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(float(r['zeta'])-.899)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 x0=np.asarray(seed['final_state'],float);base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base)
 check(np.allclose(base,p27['base_direction'],atol=1e-12),'Base direction mismatch')
 perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 check(all(abs(np.dot(base,v))<1e-9 and abs(np.linalg.norm(v)-1)<1e-9 for v in perps),'Invalid perpendicular axes')
 z=float(p48['zeta']);refs=p48['reference_drifts'];p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
 chosen=[c for c in crossings if a.only_edge is None or int(c['edge_index'])==a.only_edge]
 chosen=[c for e in (30,34) for c in sorted((c for c in chosen if int(c['edge_index'])==e),key=lambda c:float(c['scale']))[:a.rows_per_edge]]
 offsets=GRID_OFFSETS[(15-a.grid_points)//2:(15+a.grid_points)//2]
 design=[]
 for c in chosen:
  e=int(c['edge_index']);ri=int(c['row_index']);s=float(c['scale']);theta=float(c['midpoint_degrees']);di=int(next(r['direction_index'] for r in p32['rows'] if int(r['edge_index'])==e))
  check(0<=di<len(perps),f'Invalid direction index for edge {e}')
  prefix=f'e{e}_r{ri}'
  for j,d in enumerate(offsets):design.append(dict(job_id=f'{prefix}_grid_{j}',kind='grid',edge_index=e,row_index=ri,direction_index=di,scale=s,angle_degrees=theta+d,offset_degrees=d,baseline_side='',magnitude=abs(d),perturb_sign=0))
  for side in (-1,1):
   origin=theta+side*a.baseline_offset
   label='left' if side<0 else 'right'
   design.append(dict(job_id=f'{prefix}_baseline_{label}',kind='baseline',edge_index=e,row_index=ri,direction_index=di,scale=s,angle_degrees=origin,offset_degrees=side*a.baseline_offset,baseline_side=label,magnitude=0.,perturb_sign=0))
   for mag in angles:
    for sign in (-1,1):design.append(dict(job_id=f'{prefix}_angular_{label}_{mag:.9g}_{sign:+d}',kind='angular',edge_index=e,row_index=ri,direction_index=di,scale=s,angle_degrees=origin+sign*mag,offset_degrees=side*a.baseline_offset+sign*mag,baseline_side=label,magnitude=mag,perturb_sign=sign))
   for mag in radials:
    for sign in (-1,1):design.append(dict(job_id=f'{prefix}_radial_{label}_{mag:.9g}_{sign:+d}',kind='radial',edge_index=e,row_index=ri,direction_index=di,scale=s+sign*mag,angle_degrees=origin,offset_degrees=side*a.baseline_offset,baseline_side=label,magnitude=mag,perturb_sign=sign))
   for kind in ('long','extended'):design.append(dict(job_id=f'{prefix}_{kind}_{label}',kind=kind,edge_index=e,row_index=ri,direction_index=di,scale=s,angle_degrees=origin,offset_degrees=side*a.baseline_offset,baseline_side=label,magnitude=0.,perturb_sign=0))
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 config=dict(source_sha256={k:digest(v) for k,v in paths.items()},settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')},design=design)
 cfg=out/'p49_config.json';cp=out/'p49_checkpoint.json'
 if a.resume and cfg.exists():check(json.loads(cfg.read_text())==config,'Resume configuration/provenance mismatch')
 if not a.resume and cp.exists():raise ValueError('Existing checkpoint: --resume or choose another output directory')
 if a.diagnostic_only:
  save(out/'p49_preflight.json',dict(scope='P49 design and source validation only; no integrations',source_sha256=config['source_sha256'],n_sites=len(chosen),n_planned=len(design),counts={k:sum(d['kind']==k for d in design) for k in ('grid','baseline','angular','radial','long','extended')},sites=[dict(edge_index=c['edge_index'],row_index=c['row_index'],scale=c['scale'],crossing=c['midpoint_degrees']) for c in chosen]))
  print(f'P49 preflight PASS: {len(chosen)} P48 sites, {len(design)} planned integrations; no scientific results generated');return
 save(cfg,config)
 records=json.loads(cp.read_text()) if a.resume and cp.exists() else []
 byid={r['job_id']:r for r in records}
 check(len(byid)==len(records) and all(j in {d['job_id'] for d in design} for j in byid),'Checkpoint job identity mismatch')
 for d in design:
  if d['job_id'] in byid:continue
  mode='long' if d['kind'] in ('long','extended') else 'grid'
  aa=argparse.Namespace(**vars(a));aa.verify_observe=a.extended_observe if d['kind']=='extended' else a.verify_observe
  rec=dict(d,status='failed',verdict='failed')
  try:
   result=simulate(x0,p,base,perps,d['direction_index'],d['scale'],d['angle_degrees'],refs,z,aa,mode)
   rec.update(result)
   # The full final_state is intentionally not included in CSVs, but preserved in checkpoint/report.
  except Exception as ex:rec['error']=repr(ex)
  records.append(rec);byid[d['job_id']]=rec;save(cp,records)
  print(f'{len(records)}/{len(design)} {d["job_id"]}: {rec["verdict"]}',flush=True)
 # CSV outputs and summary can be regenerated deterministically from checkpoint after resume.
 grid=[r for r in records if r['kind']=='grid'];baseline=[r for r in records if r['kind']=='baseline'];pert=[r for r in records if r['kind'] in ('angular','radial')];horiz=[r for r in records if r['kind'] in ('long','extended')]
 base_by={(r['edge_index'],r['row_index'],r['baseline_side']):r for r in baseline}
 for r in pert:
  b=base_by[(r['edge_index'],r['row_index'],r['baseline_side'])]
  r['baseline_verdict']=b['verdict'];r['switched']=r['verdict']!=b['verdict'] if r['verdict'] in refs and b['verdict'] in refs else None
 for r in horiz:
  b=base_by[(r['edge_index'],r['row_index'],r['baseline_side'])]
  r['baseline_verdict']=b['verdict'];r['reproduced']=r['verdict']==b['verdict'] and r['verdict'] in refs;r['windows_consistent']=r.get('all_windows_consistent',False)
 occupancy=[];resilience=[]
 for c in chosen:
  e=int(c['edge_index']);ri=int(c['row_index']);gg=[r for r in grid if r['edge_index']==e and r['row_index']==ri]
  n0=sum(r['verdict']=='z0.899_C00' for r in gg);n1=sum(r['verdict']=='z0.899_C01' for r in gg);nu=len(gg)-n0-n1
  changes=sum(gg[i]['verdict']!=gg[i-1]['verdict'] and gg[i]['verdict'] in refs and gg[i-1]['verdict'] in refs for i in range(1,len(gg)))
  occupancy.append(dict(edge_index=e,row_index=ri,scale=float(c['scale']),boundary_angle=float(c['midpoint_degrees']),strip_halfwidth_degrees=max(abs(v) for v in offsets),n_grid=len(gg),n_C00=n0,n_C01=n1,n_unresolved=nu,fraction_C00=n0/len(gg),fraction_C01=n1/len(gg),n_adjacent_class_changes=changes,scope='selected angular strip, uniform angular samples; not global basin area'))
 for e in (30,34):
  for kind in ('angular','radial'):
   for mag in (angles if kind=='angular' else radials):
    sub=[r for r in pert if r['edge_index']==e and r['kind']==kind and r['magnitude']==mag]
    if not sub:continue
    resolved=[r for r in sub if r['switched'] is not None]
    resilience.append(dict(edge_index=e,kind=kind,magnitude=mag,n_trials=len(sub),n_resolved=len(resolved),n_switched=sum(r['switched'] for r in resolved),switch_fraction_resolved=(sum(r['switched'] for r in resolved)/len(resolved) if resolved else None),n_unresolved=len(sub)-len(resolved),scope='conditional finite-time class changes for deterministic perturbations'))
 nfailed=sum(r['status']!='complete' for r in records)
 report=dict(scope='P49 finite-time local basin occupancy and controlled perturbation resilience on frozen P48 sites',zeta=z,reference_drifts=refs,source_sha256=config['source_sha256'],settings=vars(a),sites=chosen,occupancy=occupancy,resilience=resilience,trajectories=records,n_complete=len(records)-nfailed,n_failed=nfailed,n_planned=len(design),limitations=['Angular occupancy is conditional on the narrow, boundary-centered sampled strip; it is not global basin area or a six-dimensional basin-volume estimate.','Perturbations are deterministic angular/radial displacements on the same selected two-dimensional initial-condition slices, not stochastic noise or generic six-dimensional perturbations.','Finite-time drift classifications do not prove asymptotic attraction or globally unique boundaries.','A single class change along a sampled angular row does not rule out additional unsampled crossings.','The P48 crossing midpoint and baseline-side offsets are frozen before P49; baseline classifications are independently re-integrated.'])
 save(out/'p49_report.json',report)
 common=['job_id','kind','edge_index','row_index','direction_index','scale','angle_degrees','offset_degrees','baseline_side','magnitude','perturb_sign','status','verdict','all_windows_consistent','observe','error']
 csvout(out/'p49_grid.csv',grid,common)
 csvout(out/'p49_baselines.csv',baseline,common)
 csvout(out/'p49_perturbations.csv',pert,common+['baseline_verdict','switched'])
 csvout(out/'p49_horizon_checks.csv',horiz,common+['baseline_verdict','reproduced','windows_consistent'])
 csvout(out/'p49_occupancy.csv',occupancy,['edge_index','row_index','scale','boundary_angle','strip_halfwidth_degrees','n_grid','n_C00','n_C01','n_unresolved','fraction_C00','fraction_C01','n_adjacent_class_changes','scope'])
 csvout(out/'p49_resilience.csv',resilience,['edge_index','kind','magnitude','n_trials','n_resolved','n_switched','switch_fraction_resolved','n_unresolved','scope'])
 print(f'P49 finished: {len(records)}/{len(design)} integrations, {nfailed} failures; {len(occupancy)} occupancy rows; {len(resilience)} resilience rows',flush=True)
if __name__=='__main__':main()

