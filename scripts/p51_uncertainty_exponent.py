#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p51_uncertainty_exponent.py

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
 if not v: raise ValueError(msg)
def truth(v): return str(v).lower()=='true'

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 defaults={
  'p23_report':'results/author_longtime_verification/p23_report.json',
  'p24_report':'results/author_basin_attraction/p24_report.json',
  'p27_report':'results/author_local_boundary_atlas/p27_report.json',
  'p32_report':'results/author_boundary_continuation/p32_report.json',
  'p49_grid':'results/author_basin_geometry_resilience/p49_grid.csv',
  'p50_report':'results/author_high_resolution_boundary_verification/p50_report.json',
  'p50_crossings':'results/author_high_resolution_boundary_verification/p50_crossings.csv',
  'p50_summary':'results/author_high_resolution_boundary_verification/p50_summary.csv'}
 for k,v in defaults.items(): ap.add_argument('--'+k.replace('_','-'),default=v)
 ap.add_argument('--output-dir',default='results/author_uncertainty_exponent')
 ap.add_argument('--eps-deg',default='0.0005,0.001,0.002,0.004,0.008',help='Comma-separated angular pair separations in degrees')
 ap.add_argument('--pairs-per-row',type=int,default=12)
 ap.add_argument('--scan-points',type=int,default=81,help='Odd number of uniformly spaced points per P50 row')
 ap.add_argument('--transient',type=float,default=400.); ap.add_argument('--observe',type=float,default=1600.)
 ap.add_argument('--windows',type=int,default=8); ap.add_argument('--dt',type=float,default=.5)
 ap.add_argument('--rtol',type=float,default=1e-11); ap.add_argument('--atol',type=float,default=1e-13)
 ap.add_argument('--drift-tolerance',type=float,default=.08); ap.add_argument('--max-abs',type=float,default=1e7)
 ap.add_argument('--diagnostic-only',action='store_true'); ap.add_argument('--resume',action='store_true')
 a=ap.parse_args()
 eps=sorted(set(float(x) for x in a.eps_deg.split(',') if x.strip()))
 require(len(eps)>=3 and min(eps)>0,'Need at least three positive epsilon scales')
 require(a.pairs_per_row>=4 and a.scan_points>=15 and a.scan_points%2==1,'Invalid sampling design')
 require(all(getattr(a,k)>0 for k in ('transient','observe','dt','rtol','atol')) and a.windows>=4,'Invalid integration settings')
 paths={k:Path(getattr(a,k)) for k in defaults}
 for k,p in paths.items(): require(p.is_file(),f'Missing {k}: {p}')
 p23,p24,p27,p32,p50=(json.loads(paths[k].read_text()) for k in ('p23_report','p24_report','p27_report','p32_report','p50_report'))
 require(p50['n_failed']==0 and p50['n_complete']==890 and p50['n_refined']==42,'P50 must be complete: 890/890, 42 refined transitions')
 for k in ('p23_report','p24_report','p27_report','p32_report'):
  require(p50['source_sha256'][k]==digest(paths[k]),f'P50 upstream hash mismatch: {k}')
 cross=rows(paths['p50_crossings']); summ=rows(paths['p50_summary']); p49grid=rows(paths['p49_grid'])
 require(len(cross)==42 and all(truth(r['resolved']) for r in cross),'P50 crossings must contain 42 resolved transitions')
 require(len(summ)==8 and sum(int(r['n_p50_changes']) for r in summ)==42,'P50 summary mismatch')
 require(len(p49grid)==120,'P49 grid count mismatch')
 # Reconstruct validated initial-condition family exactly as prior experiments.
 target=next(r for r in p24['trajectories'] if r['job_id']==p27['source_p24_target'])
 seed=next(r for r in p23['trajectories'] if abs(float(r['zeta'])-.899)<1e-10 and r['seed_id']==p27['source_p23_seed'] and r['source_cluster']==target['source_cluster'])
 x0=np.asarray(seed['final_state'],float); base=np.asarray(target['perturb_direction'],float); base/=np.linalg.norm(base)
 require(np.allclose(base,p27['base_direction'],atol=1e-12),'Direction mismatch')
 perps=[np.asarray(v,float) for v in p27['perpendicular_axes']]
 z=float(p50['zeta']); refs=p50.get('reference_drifts',p27['reference_drifts'])
 # P50 report does not repeat refs in some package versions; use frozen P27 refs and zeta.
 refs=p27['reference_drifts']
 p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
 selected=sorted((int(r['edge_index']),int(r['row_index']),float(r['scale'])) for r in summ)
 domains={}
 for e,ri,s in selected:
  g=sorted(float(r['angle_degrees']) for r in p49grid if int(r['edge_index'])==e and int(r['row_index'])==ri)
  require(len(g)==15,f'Missing P49 domain for edge {e} row {ri}')
  domains[(e,ri)]=(min(g),max(g))
 maxeps=max(eps)
 require(all(domains[k][1]-domains[k][0] > 2*maxeps for k in domains),'Epsilon too large for one or more domains')
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 source_sha={k:digest(v) for k,v in paths.items()}
 settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')}; settings['eps_deg_parsed']=eps
 config=dict(source_sha256=source_sha,settings=settings,selected=selected,domains={f'{k[0]}_{k[1]}':v for k,v in domains.items()})
 cp=out/'p51_checkpoint.json'; cf=out/'p51_config.json'
 if a.resume and cf.exists(): require(json.loads(cf.read_text())==config,'Resume configuration mismatch')
 if not a.resume and cp.exists(): raise ValueError('Existing checkpoint; use --resume or another output directory')
 if a.diagnostic_only:
  planned_scan=len(selected)*a.scan_points; planned_pairs=len(selected)*len(eps)*a.pairs_per_row*2
  save(out/'p51_preflight.json',dict(status='PASS',scope='Frozen P50/P49 validation; no P51 scientific integrations',n_rows=len(selected),eps_deg=eps,pairs_per_row=a.pairs_per_row,scan_points=a.scan_points,planned_scan_integrations=planned_scan,planned_pair_integrations=planned_pairs,planned_total=planned_scan+planned_pairs,source_sha256=source_sha))
  print(f'P51 preflight PASS: {len(selected)} rows; planned {planned_scan+planned_pairs} integrations ({planned_scan} scan + {planned_pairs} pair endpoints)'); return
 save(cf,config)
 records=json.loads(cp.read_text()) if a.resume and cp.exists() else []; byid={r['job_id']:r for r in records}
 require(len(byid)==len(records),'Duplicate checkpoint IDs')
 def run(jid,e,ri,s,theta,kind):
  if jid in byid:return byid[jid]
  di=int(next(r['direction_index'] for r in p32['rows'] if int(r['edge_index'])==e))
  rec=dict(job_id=jid,kind=kind,edge_index=e,row_index=ri,scale=s,angle_degrees=theta,status='failed',verdict='failed')
  try: rec.update(simulate(x0,p,base,perps,di,s,theta,refs,z,a,'grid'))
  except Exception as exc: rec['error']=repr(exc)
  records.append(rec);byid[jid]=rec;save(cp,records);print(f'{len(records)} {jid}: {rec["verdict"]}',flush=True);return rec
 # A. Independent 81-point (default) multiresolution scan across each frozen P49 angular domain.
 scan_summary=[]
 for e,ri,s in selected:
  lo,hi=domains[(e,ri)]; pts=[]
  for j,th in enumerate(np.linspace(lo,hi,a.scan_points)):
   pts.append(run(f'scan_e{e}_r{ri}_{j:03d}',e,ri,s,float(th),'scan'))
  valid=[r['verdict'] in refs for r in pts]
  changes=sum(valid[j] and valid[j-1] and pts[j]['verdict']!=pts[j-1]['verdict'] for j in range(1,len(pts)))
  old=next(r for r in summ if int(r['edge_index'])==e and int(r['row_index'])==ri)
  scan_summary.append(dict(edge_index=e,row_index=ri,scale=s,domain_min_deg=lo,domain_max_deg=hi,n_points=a.scan_points,n_valid=sum(valid),n_ambiguous=len(valid)-sum(valid),p49_changes=int(old['n_p49_changes']),p50_changes=int(old['n_p50_changes']),p51_changes=changes))
 # B. Deterministic stratified epsilon-pair experiment. Same center set at all eps, safely inside max-eps margin.
 pair_rows=[]; phi=(math.sqrt(5)-1)/2
 for e,ri,s in selected:
  lo,hi=domains[(e,ri)]; inner_lo=lo+maxeps/2; inner_hi=hi-maxeps/2
  # quasi-uniform deterministic centers; no outcome-dependent placement
  centers=[inner_lo+(inner_hi-inner_lo)*(((j+.5)*phi)%1.0) for j in range(a.pairs_per_row)]
  for ei,ep in enumerate(eps):
   for j,c in enumerate(centers):
    L=run(f'pair_e{e}_r{ri}_q{ei}_p{j:02d}_L',e,ri,s,c-ep/2,'pair')
    R=run(f'pair_e{e}_r{ri}_q{ei}_p{j:02d}_R',e,ri,s,c+ep/2,'pair')
    valid=L['verdict'] in refs and R['verdict'] in refs
    pair_rows.append(dict(edge_index=e,row_index=ri,scale=s,epsilon_deg=ep,pair_index=j,center_deg=c,left_angle_deg=c-ep/2,right_angle_deg=c+ep/2,left_class=L['verdict'],right_class=R['verdict'],valid=valid,uncertain=valid and L['verdict']!=R['verdict']))
 # Aggregate uncertainty fractions overall and by edge.
 frac=[]
 groups=[('all',None)]+[(f'edge{e}',e) for e in (30,34)]
 for label,edge in groups:
  for ep in eps:
   rr=[r for r in pair_rows if r['epsilon_deg']==ep and (edge is None or r['edge_index']==edge)]
   vv=[r for r in rr if r['valid']]; uu=sum(r['uncertain'] for r in vv)
   frac.append(dict(group=label,epsilon_deg=ep,n_pairs=len(rr),n_valid=len(vv),n_uncertain=uu,uncertain_fraction=(uu/len(vv) if vv else math.nan)))
 # Descriptive log-log slopes, only when every epsilon has 0<f<1 and >=3 points.
 fits=[]
 for label,_ in groups:
  rr=[r for r in frac if r['group']==label and math.isfinite(r['uncertain_fraction']) and 0<r['uncertain_fraction']<1]
  if len(rr)>=3:
   x=np.log([r['epsilon_deg'] for r in rr]); y=np.log([r['uncertain_fraction'] for r in rr]); coef=np.polyfit(x,y,1); pred=np.polyval(coef,x); ssr=float(np.sum((y-pred)**2)); sst=float(np.sum((y-y.mean())**2)); r2=1-ssr/sst if sst>0 else math.nan
   fits.append(dict(group=label,n_scales=len(rr),alpha=float(coef[0]),intercept=float(coef[1]),r_squared=r2,fit_status='descriptive_fit'))
  else: fits.append(dict(group=label,n_scales=len(rr),alpha=math.nan,intercept=math.nan,r_squared=math.nan,fit_status='insufficient_nonzero_interior_fractions'))
 failed=sum(r['status']!='complete' for r in records)
 csvout(out/'p51_scan_summary.csv',scan_summary,['edge_index','row_index','scale','domain_min_deg','domain_max_deg','n_points','n_valid','n_ambiguous','p49_changes','p50_changes','p51_changes'])
 csvout(out/'p51_pair_results.csv',pair_rows,['edge_index','row_index','scale','epsilon_deg','pair_index','center_deg','left_angle_deg','right_angle_deg','left_class','right_class','valid','uncertain'])
 csvout(out/'p51_uncertainty_fractions.csv',frac,['group','epsilon_deg','n_pairs','n_valid','n_uncertain','uncertain_fraction'])
 csvout(out/'p51_exponent_fits.csv',fits,['group','n_scales','alpha','intercept','r_squared','fit_status'])
 common=['job_id','kind','edge_index','row_index','scale','angle_degrees','status','verdict','all_windows_consistent','observe','dt','rtol','atol','error']
 csvout(out/'p51_integrations.csv',records,common)
 report=dict(scope='P51 multiscale finite-time basin-boundary complexity and descriptive uncertainty-exponent experiment',zeta=z,source_sha256=source_sha,settings=settings,selected=selected,domains={f'{k[0]}_{k[1]}':v for k,v in domains.items()},scan_summary=scan_summary,uncertainty_fractions=frac,exponent_fits=fits,n_complete=len(records)-failed,n_failed=failed,n_integrations=len(records),limitations=['Uncertainty exponents are descriptive finite-scale estimates, not proof of a fractal basin boundary.','Sampling is restricted to eight one-dimensional angular cuts on two initial-condition slices.','Finite-time classifications do not establish asymptotic attractor membership.','The deterministic pair design estimates sensitivity for the sampled domains and should not be interpreted as a global switching probability.'])
 save(out/'p51_report.json',report)
 print(f'P51 finished: {len(records)} integrations; {failed} failures; scan transitions={sum(r["p51_changes"] for r in scan_summary)}',flush=True)
 for r in fits: print(f'  {r["group"]}: alpha={r["alpha"]} R2={r["r_squared"]} ({r["fit_status"]})',flush=True)
if __name__=='__main__': main()

