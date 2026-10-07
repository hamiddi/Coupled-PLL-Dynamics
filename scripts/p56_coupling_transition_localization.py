#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p56_coupling_transition_localization.py

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
import argparse,csv,json,hashlib,math
from pathlib import Path
import numpy as np
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate

def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def save(p,o):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rr,ff):
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=ff,extrasaction='ignore');w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not x:raise ValueError(m)
def orient(d,zero_tol):
 d1,d2=map(float,d)
 if d1 < -zero_tol and d2 > zero_tol:return 'M' # physical (-,+)
 if d1 > zero_tol and d2 < -zero_tol:return 'P' # physical (+,-)
 return 'ambiguous'
def drift(t,Y):
 dt=float(t[-1]-t[0]);
 if dt<=0:return [math.nan,math.nan]
 # detector phases phi1=theta2-theta1, phi2=zeta*theta1-theta2 are stored through states
 # Here zeta is handled by caller because phi2 depends on it.
 return dt

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--p55-fixed-ics',default='results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv')
 ap.add_argument('--p55-control',default='results/author_corrected_fixed_ic_coupling_continuation/p55_same_zeta_control.csv')
 ap.add_argument('--p55-report',default='results/author_corrected_fixed_ic_coupling_continuation/p55_report.json')
 ap.add_argument('--output-dir',default='results/author_coupling_transition_localization')
 ap.add_argument('--zeta-start',type=float,default=.899);ap.add_argument('--zeta-stop',type=float,default=.903);ap.add_argument('--zeta-step',type=float,default=.0001)
 ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
 ap.add_argument('--long-observe',type=float,default=3200.);ap.add_argument('--extended-observe',type=float,default=6400.)
 ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
 ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13);ap.add_argument('--max-abs',type=float,default=1e7)
 ap.add_argument('--zero-tol',type=float,default=.05,help='minimum absolute detector drift for orientation assignment')
 ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
 a=ap.parse_args(); paths={k:Path(v) for k,v in {'p55_fixed_ics':a.p55_fixed_ics,'p55_control':a.p55_control,'p55_report':a.p55_report}.items()}
 for k,p in paths.items():req(p.is_file(),f'Missing {k}: {p}')
 rep=json.loads(paths['p55_report'].read_text());req(rep.get('same_zeta_control_pass') is True,'P55 same-zeta control must PASS')
 ics=readcsv(paths['p55_fixed_ics']);ctl=readcsv(paths['p55_control']);req(len(ics)==46,'Expected 46 P55 fixed ICs');req(len(ctl)==46 and all(r['all_horizons_match_p51']=='True' for r in ctl),'P55 control CSV must be 46/46 PASS')
 zs=[];z=a.zeta_start
 while z<=a.zeta_stop+1e-12:zs.append(round(z,10));z+=a.zeta_step
 req(len(zs)>=3 and abs(zs[0]-.899)<1e-10 and abs(zs[-1]-.903)<1e-10,'Default scientific scope requires endpoints 0.899 and 0.903')
 frozen=[]
 for r in ics:
  x=[float(r[f'x{i}']) for i in range(1,7)]; frozen.append(dict(ic_id=r['ic_id'],edge_index=int(r['edge_index']),row_index=int(r['row_index']),band_index=int(r['band_index']),source_class=r['source_class'],scale=float(r['scale']),angle_degrees=float(r['angle_degrees']),x0=x))
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 source_sha={k:sha(v) for k,v in paths.items()};settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')};config=dict(source_sha256=source_sha,settings=settings,n_ics=len(frozen),zetas=zs)
 cf=out/'p56_config.json';cp=out/'p56_checkpoint.json'
 if a.resume and cf.exists():req(json.loads(cf.read_text())==config,'Resume configuration mismatch')
 if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
 if a.diagnostic_only:
  n=len(frozen)*len(zs);save(out/'p56_preflight.json',dict(status='PASS',n_fixed_initial_conditions=len(frozen),n_zeta=len(zs),zeta_start=zs[0],zeta_stop=zs[-1],zeta_step=a.zeta_step,planned_primary_integrations=n,note='Long/extended endpoint validation added only for detected primary orientation transitions.'));print(f'P56 preflight PASS: {len(frozen)} fixed ICs x {len(zs)} zetas = {n} primary integrations');return
 save(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];by={r['job_id']:r for r in records};req(len(by)==len(records),'Duplicate checkpoint IDs')
 def run(ic,z,hname,obs):
  jid=f'{ic["ic_id"]}_z{z:.4f}_{hname}'
  if jid in by:return by[jid]
  p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.);x=np.asarray(ic['x0'],float).copy();rec=dict(job_id=jid,ic_id=ic['ic_id'],edge_index=ic['edge_index'],row_index=ic['row_index'],band_index=ic['band_index'],source_class=ic['source_class'],zeta=z,horizon=hname,observe=obs,status='failed',orientation='failed',all_windows_consistent=False)
  try:
   if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
   t,Y=integrate(x,p,obs,a.dt,a.rtol,a.atol,a.max_abs); wins=[]
   for k in range(a.windows):
    lo=int(k*(len(t)-1)/a.windows);hi=int((k+1)*(len(t)-1)/a.windows)+1;tt=t[lo:hi];YY=Y[lo:hi];dur=float(tt[-1]-tt[0]);phi1=YY[:,3]-YY[:,0];phi2=z*YY[:,0]-YY[:,3];d=[float((phi1[-1]-phi1[0])/dur),float((phi2[-1]-phi2[0])/dur)];wins.append(dict(window=k,drift=d,orientation=orient(d,a.zero_tol)))
   oo=[w['orientation'] for w in wins];v=oo[-1] if len(set(oo[-4:]))==1 and oo[-1] in ('M','P') else 'ambiguous';rec.update(status='complete',orientation=v,all_windows_consistent=len(set(oo))==1,final_drift=wins[-1]['drift'],min_abs_final_drift=min(abs(x) for x in wins[-1]['drift']))
  except Exception as ex:rec['error']=repr(ex)
  records.append(rec);by[jid]=rec;save(cp,records);print(f'{len(records):04d} {ic["ic_id"]} z={z:.4f} {hname}: {rec["orientation"]}',flush=True);return rec
 # Dense primary sweep.
 for ic in frozen:
  for z in zs:run(ic,z,'primary',a.observe)
 # Detect adjacent primary orientation changes; ambiguous points are reported, not silently bridged.
 transitions=[]
 for ic in frozen:
  rr=[by[f'{ic["ic_id"]}_z{z:.4f}_primary'] for z in zs]
  for j in range(1,len(rr)):
   L,R=rr[j-1],rr[j]
   if L['orientation'] in ('M','P') and R['orientation'] in ('M','P') and L['orientation']!=R['orientation']:
    transitions.append(dict(transition_id=f'{ic["ic_id"]}_t{len([q for q in transitions if q["ic_id"]==ic["ic_id"]]):02d}',ic_id=ic['ic_id'],edge_index=ic['edge_index'],row_index=ic['row_index'],band_index=ic['band_index'],zeta_left=L['zeta'],zeta_right=R['zeta'],orientation_left=L['orientation'],orientation_right=R['orientation'],bracket_width=R['zeta']-L['zeta']))
 # Validate both endpoints of every detected transition at longer horizons.
 for tr in transitions:
  ic=next(q for q in frozen if q['ic_id']==tr['ic_id'])
  for z in (tr['zeta_left'],tr['zeta_right']):
   run(ic,z,'long',a.long_observe);run(ic,z,'extended',a.extended_observe)
 # Summaries.
 rows=[]
 for ic in frozen:
  rr=[by[f'{ic["ic_id"]}_z{z:.4f}_primary'] for z in zs]; oo=[r['orientation'] for r in rr]; trs=[t for t in transitions if t['ic_id']==ic['ic_id']]
  rows.append(dict(ic_id=ic['ic_id'],edge_index=ic['edge_index'],row_index=ic['row_index'],band_index=ic['band_index'],source_class=ic['source_class'],orientation_z0p899=oo[0],orientation_z0p903=oo[-1],n_primary_M=sum(x=='M' for x in oo),n_primary_P=sum(x=='P' for x in oo),n_primary_ambiguous=sum(x=='ambiguous' for x in oo),n_adjacent_orientation_changes=len(trs),first_transition_left=(trs[0]['zeta_left'] if trs else ''),first_transition_right=(trs[0]['zeta_right'] if trs else '')))
 coup=[]
 for z in zs:
  rr=[by[f'{ic["ic_id"]}_z{z:.4f}_primary'] for ic in frozen];coup.append(dict(zeta=z,n_fixed_ics=len(frozen),n_M=sum(r['orientation']=='M' for r in rr),n_P=sum(r['orientation']=='P' for r in rr),n_ambiguous=sum(r['orientation']=='ambiguous' for r in rr),fraction_M=sum(r['orientation']=='M' for r in rr)/len(rr),fraction_P=sum(r['orientation']=='P' for r in rr)/len(rr)))
 # transition endpoint persistence
 for tr in transitions:
  ic=next(q for q in frozen if q['ic_id']==tr['ic_id']); vals=[]
  for side,z in [('left',tr['zeta_left']),('right',tr['zeta_right'])]:
   pv=by[f'{ic["ic_id"]}_z{z:.4f}_primary']['orientation'];lv=by[f'{ic["ic_id"]}_z{z:.4f}_long']['orientation'];ev=by[f'{ic["ic_id"]}_z{z:.4f}_extended']['orientation'];tr[f'{side}_primary']=pv;tr[f'{side}_long']=lv;tr[f'{side}_extended']=ev;tr[f'{side}_persistent']=(pv==lv==ev)
  tr['both_endpoints_persistent']=tr['left_persistent'] and tr['right_persistent']
 failed=sum(r['status']!='complete' for r in records)
 csvout(out/'p56_integrations.csv',records,['job_id','ic_id','edge_index','row_index','band_index','source_class','zeta','horizon','observe','status','orientation','all_windows_consistent','final_drift','min_abs_final_drift','error'])
 csvout(out/'p56_transition_brackets.csv',transitions,['transition_id','ic_id','edge_index','row_index','band_index','zeta_left','zeta_right','orientation_left','orientation_right','bracket_width','left_primary','left_long','left_extended','left_persistent','right_primary','right_long','right_extended','right_persistent','both_endpoints_persistent'])
 csvout(out/'p56_ic_summary.csv',rows,['ic_id','edge_index','row_index','band_index','source_class','orientation_z0p899','orientation_z0p903','n_primary_M','n_primary_P','n_primary_ambiguous','n_adjacent_orientation_changes','first_transition_left','first_transition_right'])
 csvout(out/'p56_coupling_summary.csv',coup,['zeta','n_fixed_ics','n_M','n_P','n_ambiguous','fraction_M','fraction_P'])
 report=dict(scope='P56 fixed-IC dense coupling sweep using physical detector-drift orientation, not coupling-specific cluster labels',source_sha256=source_sha,settings=settings,n_fixed_initial_conditions=len(frozen),n_zeta=len(zs),n_primary_integrations=len(frozen)*len(zs),n_detected_orientation_transitions=len(transitions),n_transition_endpoints_persistent=sum(t.get('both_endpoints_persistent',False) for t in transitions),n_integrations=len(records),n_complete=len(records)-failed,n_failed=failed,orientation_definition={'M':'detector drift (-,+)','P':'detector drift (+,-)','ambiguous':f'not a stable orientation or |drift| <= {a.zero_tol}'},label_warning='P23 C00/C01 cluster labels permute physical drift orientation across zeta: (-,+) is C00 at 0.899, C01 at 0.901, C00 at 0.903. P56 therefore does not use C00/C01 as cross-zeta physical identities.',limitations=['Grid spacing brackets orientation changes but does not locate a bifurcation or asymptotic critical coupling.','Finite-time drift orientation does not prove asymptotic attractor identity.','A 0.0001 zeta grid can miss narrower alternating windows.','Only the 46 P55/P51 representative initial conditions are tested.'])
 save(out/'p56_report.json',report);print(f'P56 finished: primary={len(frozen)*len(zs)}, transitions={len(transitions)}, total integrations={len(records)}, failures={failed}')
if __name__=='__main__':main()

