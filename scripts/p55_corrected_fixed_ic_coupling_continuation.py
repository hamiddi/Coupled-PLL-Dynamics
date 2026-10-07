#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p55_corrected_fixed_ic_coupling_continuation.py

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
import argparse,csv,json,math,hashlib
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate,classify

def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def save(p,o):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rr,ff):
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=ff,extrasaction='ignore');w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not x:raise ValueError(m)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--p23-report',default='results/author_longtime_verification/p23_report.json')
 ap.add_argument('--p24-report',default='results/author_basin_attraction/p24_report.json')
 ap.add_argument('--p27-report',default='results/author_local_boundary_atlas/p27_report.json')
 ap.add_argument('--p32-report',default='results/author_boundary_continuation/p32_report.json')
 ap.add_argument('--p51-report',default='results/author_uncertainty_exponent/p51_report.json')
 ap.add_argument('--p51-integrations',default='results/author_uncertainty_exponent/p51_integrations.csv')
 ap.add_argument('--output-dir',default='results/author_corrected_fixed_ic_coupling_continuation')
 ap.add_argument('--zetas',default='0.899,0.901,0.903')
 ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
 ap.add_argument('--long-observe',type=float,default=3200.);ap.add_argument('--extended-observe',type=float,default=6400.)
 ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
 ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13)
 ap.add_argument('--drift-tolerance',type=float,default=.08);ap.add_argument('--max-abs',type=float,default=1e7)
 ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
 a=ap.parse_args(); zs=[float(x) for x in a.zetas.split(',') if x.strip()]
 paths={k:Path(v) for k,v in {'p23_report':a.p23_report,'p24_report':a.p24_report,'p27_report':a.p27_report,'p32_report':a.p32_report,'p51_report':a.p51_report,'p51_integrations':a.p51_integrations}.items()}
 for k,p in paths.items():req(p.is_file(),f'Missing {k}: {p}')
 p23=json.loads(paths['p23_report'].read_text());p24=json.loads(paths['p24_report'].read_text());p27=json.loads(paths['p27_report'].read_text());p32=json.loads(paths['p32_report'].read_text());p51=json.loads(paths['p51_report'].read_text());p51i=readcsv(paths['p51_integrations'])
 req(p51['n_failed']==0 and p51['n_complete']==1608,'P51 must be complete 1608/1608')
 available=sorted(float(c['zeta']) for c in p23['comparisons']);req(all(any(abs(z-q)<1e-12 for q in available) for z in zs),f'Each zeta needs frozen P23 references; available={available}')
 req(any(abs(z-.899)<1e-12 for z in zs),'zeta=0.899 required as selection source')
 # P54-verified exact P51 geometry at zeta=.899: the center is the P23 seed final_state.
 # P53 incorrectly used the P24 target final_state; P55 explicitly prevents that regression.
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(float(r['zeta'])-.899)<1e-12 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 center=np.asarray(seed['final_state'],float)
 wrong_center=np.asarray(target['final_state'],float)
 req(np.linalg.norm(center-wrong_center)>1.0,'P51 and P24 centers unexpectedly coincide; audit historical inputs')
 base=np.asarray(target['perturb_direction'],float);base/=np.linalg.norm(base);perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 req(np.allclose(base,p27['base_direction'],atol=1e-12),'Base direction mismatch')
 direction={int(r['edge_index']):int(r['direction_index']) for r in p32['rows']};req(all(e in direction for e in (30,34)),'Missing edge directions')
 selected=[tuple(x) for x in p51['selected']]
 # Select midpoint sample of every contiguous valid class band in each P51 scan row.
 frozen=[]
 for e,ri,s in selected:
  e=int(e);ri=int(ri);s=float(s)
  rr=[r for r in p51i if r['kind']=='scan' and int(r['edge_index'])==e and int(r['row_index'])==ri and r['status']=='complete' and r['verdict']!='ambiguous']
  rr=sorted(rr,key=lambda r:float(r['angle_degrees']));req(len(rr)==81,f'Expected 81 P51 scan points for e{e} r{ri}, got {len(rr)}')
  runs=[];start=0
  for j in range(1,len(rr)+1):
   if j==len(rr) or rr[j]['verdict']!=rr[start]['verdict']:
    runs.append(rr[start:j]);start=j
  for bi,run in enumerate(runs):
   r=run[len(run)//2];th=float(r['angle_degrees']);di=direction[e];vec=math.cos(math.radians(th))*base+math.sin(math.radians(th))*perps[di];x=center+s*vec
   frozen.append(dict(ic_id=f'e{e}_r{ri}_b{bi:02d}',edge_index=e,row_index=ri,band_index=bi,source_class=r['verdict'],source_job_id=r['job_id'],scale=s,angle_degrees=th,band_n_samples=len(run),x0=x.tolist()))
 req(len(frozen)>=32,'Unexpectedly few frozen band representatives')
 # P23 coupling-specific drift references only; x0 never changes with zeta.
 contexts={}
 for z in zs:
  comp=next(c for c in p23['comparisons'] if abs(float(c['zeta'])-z)<1e-12);refs=dict(zip(comp['clusters'],comp['mean_detector_drifts']));c01=next(k for k in refs if k.endswith('C01'));c00=next(k for k in refs if k.endswith('C00'));contexts[z]=dict(refs=refs,c00=c00,c01=c01)
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);source_sha={k:sha(v) for k,v in paths.items()}
 settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')};config=dict(source_sha256=source_sha,settings=settings,frozen_ics=frozen)
 cp=out/'p55_checkpoint.json';cf=out/'p55_config.json'
 if a.resume and cf.exists():req(json.loads(cf.read_text())==config,'Resume configuration mismatch')
 if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
 if a.diagnostic_only:
  n=len(frozen)*len(zs)*3;save(out/'p55_preflight.json',dict(status='PASS',n_fixed_initial_conditions=len(frozen),zetas=zs,primary_integrations=len(frozen)*len(zs),long_integrations=len(frozen)*len(zs),extended_integrations=len(frozen)*len(zs),planned_integrations=n,source_sha256=source_sha));print(f'P55 preflight PASS: {len(frozen)} fixed ICs x {len(zs)} zetas x 3 horizons = {n} integrations');return
 save(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];by={r['job_id']:r for r in records};req(len(by)==len(records),'Duplicate checkpoint IDs')
 def run(ic,z,hname,obs):
  jid=f'{ic["ic_id"]}_z{z:.3f}_{hname}'
  if jid in by:return by[jid]
  ctx=contexts[z];p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.);x=np.asarray(ic['x0'],float).copy()
  rec=dict(job_id=jid,ic_id=ic['ic_id'],edge_index=ic['edge_index'],row_index=ic['row_index'],band_index=ic['band_index'],source_class=ic['source_class'],scale=ic['scale'],angle_degrees=ic['angle_degrees'],zeta=z,horizon=hname,observe=obs,status='failed',verdict='failed',all_windows_consistent=False)
  try:
   if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
   t,Y=integrate(x,p,obs,a.dt,a.rtol,a.atol,a.max_abs);wins=[]
   for k in range(a.windows):
    lo=int(k*(len(t)-1)/a.windows);hi=int((k+1)*(len(t)-1)/a.windows)+1;wins.append(dict(window=k,**classify(t[lo:hi],Y[lo:hi],z,ctx['refs'],a.drift_tolerance)))
   cl=[w['classification'] for w in wins];v=cl[-1] if len(set(cl[-4:]))==1 and cl[-1] in ctx['refs'] else 'ambiguous';rec.update(status='complete',verdict=v,all_windows_consistent=len(set(cl))==1,final_drift=wins[-1]['drift'],distance_to_C00=wins[-1]['distances'][ctx['c00']],distance_to_C01=wins[-1]['distances'][ctx['c01']])
  except Exception as ex:rec['error']=repr(ex)
  records.append(rec);by[jid]=rec;save(cp,records);print(f'{len(records):04d} {ic["ic_id"]} z={z:.3f} {hname}: {rec["verdict"]}',flush=True);return rec
 for ic in frozen:
  for z in zs:
   run(ic,z,'primary',a.observe);run(ic,z,'long',a.long_observe);run(ic,z,'extended',a.extended_observe)
 # summarize primary coupling response and horizon persistence
 summ=[]
 for ic in frozen:
  row=dict(ic_id=ic['ic_id'],edge_index=ic['edge_index'],row_index=ic['row_index'],band_index=ic['band_index'],source_class=ic['source_class'],scale=ic['scale'],angle_degrees=ic['angle_degrees'])
  prim=[];persist=True
  for z in zs:
   p=by[f'{ic["ic_id"]}_z{z:.3f}_primary'];l=by[f'{ic["ic_id"]}_z{z:.3f}_long'];e=by[f'{ic["ic_id"]}_z{z:.3f}_extended'];row[f'z{z:.3f}_primary']=p['verdict'];row[f'z{z:.3f}_long']=l['verdict'];row[f'z{z:.3f}_extended']=e['verdict'];prim.append(p['verdict']);persist &= (p['verdict']==l['verdict']==e['verdict'])
  row['n_coupling_changes']=sum(prim[j]!=prim[j-1] for j in range(1,len(prim)));row['coupling_sensitive']=row['n_coupling_changes']>0;row['all_horizons_agree']=persist;summ.append(row)
 agg=[]
 for z in zs:
  rr=[by[f'{ic["ic_id"]}_z{z:.3f}_primary'] for ic in frozen];ctx=contexts[z];valid=[r for r in rr if r['verdict'] in ctx['refs']]
  agg.append(dict(zeta=z,n_fixed_ics=len(frozen),n_valid=len(valid),n_ambiguous=len(rr)-len(valid),n_C00=sum(r['verdict']==ctx['c00'] for r in valid),n_C01=sum(r['verdict']==ctx['c01'] for r in valid),fraction_C00=sum(r['verdict']==ctx['c00'] for r in valid)/len(valid) if valid else math.nan,fraction_C01=sum(r['verdict']==ctx['c01'] for r in valid)/len(valid) if valid else math.nan))
 failed=sum(r['status']!='complete' for r in records);n_sensitive=sum(r['coupling_sensitive'] for r in summ);n_persist=sum(r['all_horizons_agree'] for r in summ)
 # Mandatory same-coupling control: every exact P51 state must reproduce its P51 label at zeta=.899.
 control_rows=[]
 for ic in frozen:
  vals=[by[f'{ic["ic_id"]}_z0.899_{h}']['verdict'] for h in ('primary','long','extended')]
  primary_match=(vals[0]==ic['source_class']); all_match=all(v==ic['source_class'] for v in vals)
  control_rows.append(dict(ic_id=ic['ic_id'],source_class=ic['source_class'],zeta_0p899_primary=vals[0],zeta_0p899_long=vals[1],zeta_0p899_extended=vals[2],primary_matches_p51=primary_match,all_horizons_match_p51=all_match))
 control_primary_pass=all(r['primary_matches_p51'] for r in control_rows)
 control_all_horizons_pass=all(r['all_horizons_match_p51'] for r in control_rows)
 control_pass=control_primary_pass and control_all_horizons_pass and failed==0
 csvout(out/'p55_same_zeta_control.csv',control_rows,['ic_id','source_class','zeta_0p899_primary','zeta_0p899_long','zeta_0p899_extended','primary_matches_p51','all_horizons_match_p51'])
 # fixed IC output proves exact six-dimensional states used across all zetas
 fic=[]
 for ic in frozen:
  r={k:ic[k] for k in ('ic_id','edge_index','row_index','band_index','source_class','source_job_id','scale','angle_degrees','band_n_samples')};r.update({f'x{i+1}':ic['x0'][i] for i in range(6)});fic.append(r)
 csvout(out/'p55_fixed_initial_conditions.csv',fic,['ic_id','edge_index','row_index','band_index','source_class','source_job_id','scale','angle_degrees','band_n_samples','x1','x2','x3','x4','x5','x6'])
 csvout(out/'p55_integrations.csv',records,['job_id','ic_id','edge_index','row_index','band_index','source_class','scale','angle_degrees','zeta','horizon','observe','status','verdict','all_windows_consistent','final_drift','distance_to_C00','distance_to_C01','error'])
 sf=['ic_id','edge_index','row_index','band_index','source_class','scale','angle_degrees']+[f'z{z:.3f}_{h}' for z in zs for h in ('primary','long','extended')]+['n_coupling_changes','coupling_sensitive','all_horizons_agree']
 csvout(out/'p55_ic_summary.csv',summ,sf);csvout(out/'p55_coupling_summary.csv',agg,['zeta','n_fixed_ics','n_valid','n_ambiguous','n_C00','n_C01','fraction_C00','fraction_C01'])
 report=dict(scope='P55 corrected controlled coupling continuation using P54-verified exact P51 initial states',zetas=zs,source_sha256=source_sha,settings=settings,n_fixed_initial_conditions=len(frozen),selection_rule='Midpoint P51 scan sample from every contiguous valid class band in each of the eight frozen P51 rows.',n_integrations=len(records),n_complete=len(records)-failed,n_failed=failed,p51_center_source='P23 seed final_state verified by P54',p51_center_vs_p24_target_l2=float(np.linalg.norm(center-wrong_center)),same_zeta_control_pass=control_pass,same_zeta_primary_pass=control_primary_pass,same_zeta_all_horizons_pass=control_all_horizons_pass,n_same_zeta_control_failures=sum(not r['all_horizons_match_p51'] for r in control_rows),interpretation_status=('CONTROL_PASS: coupling comparison eligible for interpretation' if control_pass else 'CONTROL_FAIL: do not interpret coupling comparison'),n_coupling_sensitive_ics=(n_sensitive if control_pass else None),n_all_horizons_agree=n_persist,coupling_summary=(agg if control_pass else []),limitations=['Only P23-supported zeta values 0.899, 0.901, 0.903 are classified.','Initial conditions are selected from P51 finite-resolution bands at zeta=0.899; bands narrower than the P51 grid may be absent.','A class change across sampled zeta values localizes coupling sensitivity only to the interval, not a critical coupling value.','Finite-time drift classes do not prove asymptotic attractor membership or global basin topology.'])
 save(out/'p55_report.json',report);print(f'P55 finished: fixed ICs={len(frozen)}, integrations={len(records)}, failures={failed}, coupling-sensitive ICs={n_sensitive}, all-horizon-agree={n_persist}/{len(frozen)}')
if __name__=='__main__':main()

