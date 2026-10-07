#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p57_high_resolution_coupling_threshold_refinement.py

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
    if d1 < -zero_tol and d2 > zero_tol:return 'M'
    if d1 > zero_tol and d2 < -zero_tol:return 'P'
    return 'ambiguous'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p55-fixed-ics',default='results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv')
    ap.add_argument('--p56-report',default='results/author_coupling_transition_localization/p56_report.json')
    ap.add_argument('--p56-brackets',default='results/author_coupling_transition_localization/p56_transition_brackets.csv')
    ap.add_argument('--output-dir',default='results/author_high_resolution_coupling_threshold_refinement')
    ap.add_argument('--levels',type=int,default=10)
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.)
    ap.add_argument('--long-observe',type=float,default=3200.);ap.add_argument('--extended-observe',type=float,default=6400.)
    ap.add_argument('--windows',type=int,default=8);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--zero-tol',type=float,default=.05)
    ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); paths={k:Path(v) for k,v in {'p55_fixed_ics':a.p55_fixed_ics,'p56_report':a.p56_report,'p56_brackets':a.p56_brackets}.items()}
    for k,p in paths.items():req(p.is_file(),f'Missing {k}: {p}')
    rep=json.loads(paths['p56_report'].read_text());req(rep.get('n_detected_orientation_transitions')==23 and rep.get('n_transition_endpoints_persistent')==23,'P56 must contain 23/23 persistent transitions')
    ics={r['ic_id']:r for r in readcsv(paths['p55_fixed_ics'])}; br=readcsv(paths['p56_brackets']);req(len(br)==23,'Expected exactly 23 P56 transition brackets')
    for r in br:
        req(r['ic_id'] in ics,f'Missing IC {r["ic_id"]}')
        req(r['orientation_left']=='P' and r['orientation_right']=='M','P57 default scope requires P->M P56 brackets')
        req(r['both_endpoints_persistent']=='True','Every P56 bracket must have persistent endpoints')
        req(abs(float(r['zeta_left'])-.899)<1e-12 and abs(float(r['zeta_right'])-.8991)<1e-12,'Expected P56 brackets [0.8990,0.8991]')
    frozen={k:np.array([float(v[f'x{i}']) for i in range(1,7)],float) for k,v in ics.items()}
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    source_sha={k:sha(v) for k,v in paths.items()};settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')};config=dict(source_sha256=source_sha,settings=settings,n_brackets=len(br))
    cf=out/'p57_config.json';cp=out/'p57_checkpoint.json'
    if a.resume and cf.exists():req(json.loads(cf.read_text())==config,'Resume configuration mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
    planned=len(br)*a.levels + len(br)*4
    target=1e-4/(2**a.levels)
    if a.diagnostic_only:
        save(out/'p57_preflight.json',dict(status='PASS',n_brackets=len(br),levels=a.levels,initial_width=1e-4,target_width=target,planned_bisection_integrations=len(br)*a.levels,planned_endpoint_validation_integrations=len(br)*4,planned_total_integrations=planned))
        print(f'P57 preflight PASS: {len(br)} brackets x {a.levels} bisections + {len(br)*4} endpoint validations = {planned} integrations; target width={target:.12g}')
        return
    save(cf,config);records=json.loads(cp.read_text()) if a.resume and cp.exists() else [];by={r['job_id']:r for r in records};req(len(by)==len(records),'Duplicate checkpoint IDs')
    def run(ic_id,z,hname,obs,tag):
        jid=f'{ic_id}_{tag}_z{z:.12f}_{hname}'
        if jid in by:return by[jid]
        p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.);x=frozen[ic_id].copy();rec=dict(job_id=jid,ic_id=ic_id,tag=tag,zeta=z,horizon=hname,observe=obs,status='failed',orientation='failed',all_windows_consistent=False)
        try:
            if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
            t,Y=integrate(x,p,obs,a.dt,a.rtol,a.atol,a.max_abs);wins=[]
            for k in range(a.windows):
                lo=int(k*(len(t)-1)/a.windows);hi=int((k+1)*(len(t)-1)/a.windows)+1;tt=t[lo:hi];YY=Y[lo:hi];dur=float(tt[-1]-tt[0]);phi1=YY[:,3]-YY[:,0];phi2=z*YY[:,0]-YY[:,3];d=[float((phi1[-1]-phi1[0])/dur),float((phi2[-1]-phi2[0])/dur)];wins.append((d,orient(d,a.zero_tol)))
            oo=[q[1] for q in wins];o=oo[-1] if len(set(oo[-4:]))==1 and oo[-1] in ('M','P') else 'ambiguous';rec.update(status='complete',orientation=o,all_windows_consistent=len(set(oo))==1,final_drift=wins[-1][0],min_abs_final_drift=min(abs(q) for q in wins[-1][0]))
        except Exception as ex:rec['error']=repr(ex)
        records.append(rec);by[jid]=rec;save(cp,records);print(f'{len(records):04d} {ic_id} {tag} z={z:.12f} {hname}: {rec["orientation"]}',flush=True);return rec
    refined=[];ambiguous_count=0
    for bi,b in enumerate(br):
        ic=b['ic_id'];L=float(b['zeta_left']);R=float(b['zeta_right']);oL='P';oR='M';history=[];status='refined'
        for level in range(1,a.levels+1):
            mid=(L+R)/2.;rr=run(ic,mid,'primary',a.observe,f'b{bi:02d}_l{level:02d}');om=rr['orientation'];history.append(dict(level=level,zeta=mid,orientation=om,left_before=L,right_before=R))
            if om==oL:L=mid
            elif om==oR:R=mid
            else:
                status='ambiguous_midpoint';ambiguous_count+=1;break
        refined.append(dict(transition_id=b['transition_id'],ic_id=ic,edge_index=int(b['edge_index']),row_index=int(b['row_index']),band_index=int(b['band_index']),status=status,zeta_left=L,zeta_right=R,orientation_left=oL,orientation_right=oR,bracket_width=R-L,zeta_midpoint=(L+R)/2.,levels_completed=len(history),history=json.dumps(history,separators=(',',':'))))
    # Long/extended validation only for successfully refined final endpoints.
    for i,r in enumerate(refined):
        if r['status']!='refined':continue
        for side in ('left','right'):
            z=r[f'zeta_{side}'];expected=r[f'orientation_{side}'];lv=run(r['ic_id'],z,'long',a.long_observe,f'final{i:02d}_{side}');ev=run(r['ic_id'],z,'extended',a.extended_observe,f'final{i:02d}_{side}');r[f'{side}_long']=lv['orientation'];r[f'{side}_extended']=ev['orientation'];r[f'{side}_persistent']=(expected==lv['orientation']==ev['orientation'])
        r['both_endpoints_persistent']=r.get('left_persistent',False) and r.get('right_persistent',False)
    failed=sum(r['status']!='complete' for r in records)
    csvout(out/'p57_integrations.csv',records,['job_id','ic_id','tag','zeta','horizon','observe','status','orientation','all_windows_consistent','final_drift','min_abs_final_drift','error'])
    csvout(out/'p57_refined_thresholds.csv',refined,['transition_id','ic_id','edge_index','row_index','band_index','status','zeta_left','zeta_right','orientation_left','orientation_right','bracket_width','zeta_midpoint','levels_completed','left_long','left_extended','left_persistent','right_long','right_extended','right_persistent','both_endpoints_persistent','history'])
    mids=[r['zeta_midpoint'] for r in refined if r['status']=='refined'];widths=[r['bracket_width'] for r in refined if r['status']=='refined']
    summary=dict(n_refined=len(mids),n_ambiguous_midpoints=ambiguous_count,threshold_midpoint_min=min(mids) if mids else None,threshold_midpoint_max=max(mids) if mids else None,threshold_midpoint_mean=float(np.mean(mids)) if mids else None,threshold_midpoint_std=float(np.std(mids,ddof=1)) if len(mids)>1 else 0.,threshold_midpoint_range=(max(mids)-min(mids)) if mids else None,max_final_bracket_width=max(widths) if widths else None,n_both_endpoints_persistent=sum(bool(r.get('both_endpoints_persistent')) for r in refined))
    save(out/'p57_threshold_summary.json',summary)
    report=dict(scope='P57 high-resolution bisection refinement of P56 physical P->M orientation changes with exact fixed P55 initial conditions',source_sha256=source_sha,settings=settings,n_input_brackets=len(br),n_refined=summary['n_refined'],n_ambiguous_midpoints=ambiguous_count,n_integrations=len(records),n_complete=len(records)-failed,n_failed=failed,n_both_endpoints_persistent=summary['n_both_endpoints_persistent'],target_width=target,threshold_summary=summary,orientation_definition={'M':'detector drift (-,+)','P':'detector drift (+,-)'},limitations=['Bisection localizes a finite-time orientation boundary inside each supplied P56 bracket; it does not prove a dynamical bifurcation or asymptotic critical coupling.','Bisection assumes only that the maintained bracket endpoints retain opposite orientations; it does not establish uniqueness or monotonicity inside the bracket. Narrow additional alternating windows can remain unresolved.','Only 23 P56-switching representative initial conditions are refined.','Threshold dispersion across initial conditions should be interpreted as finite-time regime-selection boundary motion unless invariant-set continuation independently establishes a common bifurcation.'])
    save(out/'p57_report.json',report);print(f'P57 finished: refined={summary["n_refined"]}/{len(br)}, persistent={summary["n_both_endpoints_persistent"]}, total integrations={len(records)}, failures={failed}')
if __name__=='__main__':main()

