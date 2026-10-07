#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p60_coupling_axis_topology_audit.py

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
import argparse,csv,json,hashlib
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
def orient(d1,d2,zt=.05):
    if d1 < -zt and d2 > zt:return 'M'
    if d1 > zt and d2 < -zt:return 'P'
    return 'ambiguous'
def classify(t,Y,zeta):
    p1=Y[:,3]-Y[:,0];p2=zeta*Y[:,0]-Y[:,3];dur=float(t[-1]-t[0]);d1=float((p1[-1]-p1[0])/dur);d2=float((p2[-1]-p2[0])/dur)
    return orient(d1,d2),d1,d2

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p55-fixed-ics',default='results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv')
    ap.add_argument('--p57-thresholds',default='results/author_high_resolution_coupling_threshold_refinement/p57_refined_thresholds.csv')
    ap.add_argument('--p59-metrics',default='results/author_critical_transition_early_warning/p59_metrics.csv')
    ap.add_argument('--output-dir',default='results/author_coupling_axis_topology_audit')
    ap.add_argument('--below',type=float,default=4e-7,help='scan this far below P57 left endpoint')
    ap.add_argument('--above',type=float,default=1e-7,help='scan this far above P57 left endpoint')
    ap.add_argument('--n-points',type=int,default=41)
    ap.add_argument('--transient',type=float,default=400.);ap.add_argument('--observe',type=float,default=1600.);ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--verify-horizons',default='3200,6400');ap.add_argument('--rtol',type=float,default=1e-11);ap.add_argument('--atol',type=float,default=1e-13);ap.add_argument('--max-abs',type=float,default=1e7)
    ap.add_argument('--diagnostic-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); paths={'p55_fixed_ics':Path(a.p55_fixed_ics),'p57_thresholds':Path(a.p57_thresholds),'p59_metrics':Path(a.p59_metrics)}
    for k,p in paths.items():req(p.is_file(),f'Missing {k}: {p}')
    ics={r['ic_id']:r for r in readcsv(paths['p55_fixed_ics'])}; thr={r['ic_id']:r for r in readcsv(paths['p57_thresholds'])}; m59=readcsv(paths['p59_metrics'])
    affected=sorted({r['ic_id'] for r in m59 if r.get('status')=='complete' and r.get('orientation')!='P'})
    req(affected,'No P59 non-P points found'); req(all(i in ics and i in thr for i in affected),'Affected IC missing from P55/P57')
    X={k:np.array([float(v[f'x{i}']) for i in range(1,7)],float) for k,v in ics.items()}; horizons=[float(x) for x in a.verify_horizons.split(',')]
    req(a.n_points>=5 and a.below>0 and a.above>0,'Invalid scan geometry')
    offsets=np.linspace(-a.below,a.above,a.n_points)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cp=out/'p60_checkpoint.json';cf=out/'p60_config.json'
    config=dict(source_sha256={k:sha(v) for k,v in paths.items()},affected_ics=affected,settings={k:v for k,v in vars(a).items() if k not in ('resume','diagnostic_only','output_dir')},offsets=offsets.tolist())
    if a.resume and cf.exists():req(json.loads(cf.read_text())==config,'Resume configuration mismatch')
    if not a.resume and cp.exists():raise ValueError('Existing checkpoint; use --resume or another output directory')
    if a.diagnostic_only:
        save(out/'p60_preflight.json',dict(status='PASS',affected_ics=affected,n_affected=len(affected),scan_points_per_ic=a.n_points,primary_integrations=len(affected)*a.n_points,offset_min=float(offsets.min()),offset_max=float(offsets.max()),step=float(offsets[1]-offsets[0]),verification_horizons=horizons,n_p59_non_p_points=sum(r.get('status')=='complete' and r.get('orientation')!='P' for r in m59)))
        print(f'P60 preflight PASS: {len(affected)} affected ICs x {a.n_points} = {len(affected)*a.n_points} primary integrations; long verification added per detected transition');return
    save(cf,config); records=json.loads(cp.read_text()) if a.resume and cp.exists() else []; by={r['job_id']:r for r in records};req(len(by)==len(records),'Duplicate checkpoint IDs')
    # primary dense scans
    total=len(affected)*a.n_points
    for ic in affected:
        zl=float(thr[ic]['zeta_left'])
        for j,off in enumerate(offsets):
            z=zl+float(off);jid=f'primary_{ic}_{j:03d}'
            if jid in by:continue
            rec=dict(job_id=jid,phase='primary',ic_id=ic,edge_index=int(thr[ic]['edge_index']),row_index=int(thr[ic]['row_index']),band_index=int(thr[ic]['band_index']),zeta=z,zeta_left=zl,offset_from_left=float(off),observe=a.observe,status='failed')
            try:
                p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.);x=X[ic].copy()
                if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
                t,Y=integrate(x,p,a.observe,a.dt,a.rtol,a.atol,a.max_abs);o,d1,d2=classify(t,Y,z);rec.update(status='complete',orientation=o,drift1=d1,drift2=d2)
            except Exception as ex:rec['error']=repr(ex)
            records.append(rec);by[jid]=rec;save(cp,records);print(f'primary {sum(r["phase"]=="primary" for r in records):03d}/{total} {ic} off={off:+.3e}: {rec.get("orientation","failed")}',flush=True)
    # detect transitions from primary scan
    transitions=[]
    for ic in affected:
        rr=sorted([r for r in records if r['phase']=='primary' and r['ic_id']==ic and r['status']=='complete'],key=lambda r:r['zeta'])
        for k,(L,R) in enumerate(zip(rr[:-1],rr[1:])):
            if L.get('orientation') in ('P','M') and R.get('orientation') in ('P','M') and L['orientation']!=R['orientation']:
                transitions.append(dict(transition_id=f'{ic}_t{k:02d}',ic_id=ic,zeta_left=float(L['zeta']),zeta_right=float(R['zeta']),orientation_left=L['orientation'],orientation_right=R['orientation'],width=float(R['zeta'])-float(L['zeta'])))
    # verify both endpoints at longer horizons
    for tr in transitions:
        for side in ('left','right'):
            z=tr['zeta_'+side]; expected=tr['orientation_'+side]
            for h in horizons:
                jid=f'verify_{tr["transition_id"]}_{side}_{h:g}'
                if jid in by:continue
                rec=dict(job_id=jid,phase='verify',transition_id=tr['transition_id'],ic_id=tr['ic_id'],zeta=z,side=side,expected_orientation=expected,observe=h,status='failed')
                try:
                    p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.);x=X[tr['ic_id']].copy()
                    if a.transient:x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
                    t,Y=integrate(x,p,h,a.dt,a.rtol,a.atol,a.max_abs);o,d1,d2=classify(t,Y,z);rec.update(status='complete',orientation=o,drift1=d1,drift2=d2,reproduced=(o==expected))
                except Exception as ex:rec['error']=repr(ex)
                records.append(rec);by[jid]=rec;save(cp,records);print(f'verify {tr["transition_id"]} {side} h={h:g}: {rec.get("orientation","failed")}',flush=True)
    fields=['job_id','phase','transition_id','ic_id','edge_index','row_index','band_index','zeta','zeta_left','offset_from_left','side','expected_orientation','observe','status','orientation','drift1','drift2','reproduced','error']
    csvout(out/'p60_integrations.csv',records,fields)
    # transition verification summary
    vrows=[]
    for tr in transitions:
        vr=[r for r in records if r['phase']=='verify' and r.get('transition_id')==tr['transition_id']]
        q=dict(**tr,n_verify=len(vr),n_verify_complete=sum(r['status']=='complete' for r in vr),n_verify_reproduced=sum(bool(r.get('reproduced')) for r in vr),persistent=bool(vr) and all(r['status']=='complete' and r.get('reproduced') for r in vr))
        vrows.append(q)
    csvout(out/'p60_transition_brackets.csv',vrows,['transition_id','ic_id','zeta_left','zeta_right','orientation_left','orientation_right','width','n_verify','n_verify_complete','n_verify_reproduced','persistent'])
    # per IC topology
    irows=[]
    for ic in affected:
        rr=sorted([r for r in records if r['phase']=='primary' and r['ic_id']==ic and r['status']=='complete'],key=lambda r:r['zeta']); seq=[]
        for r in rr:
            if not seq or r['orientation']!=seq[-1]:seq.append(r['orientation'])
        tv=[v for v in vrows if v['ic_id']==ic]
        irows.append(dict(ic_id=ic,n_primary=len(rr),n_ambiguous=sum(r['orientation']=='ambiguous' for r in rr),n_transitions=len(tv),orientation_sequence='->'.join(seq),n_persistent=sum(v['persistent'] for v in tv),all_transitions_persistent=bool(tv) and all(v['persistent'] for v in tv)))
    csvout(out/'p60_ic_summary.csv',irows,['ic_id','n_primary','n_ambiguous','n_transitions','orientation_sequence','n_persistent','all_transitions_persistent'])
    summary=dict(n_affected_ics=len(affected),affected_ics=affected,n_p59_non_p_points=sum(r.get('status')=='complete' and r.get('orientation')!='P' for r in m59),n_primary_planned=total,n_primary_complete=sum(r['phase']=='primary' and r['status']=='complete' for r in records),n_primary_failed=sum(r['phase']=='primary' and r['status']!='complete' for r in records),n_transitions=len(vrows),n_persistent_transitions=sum(v['persistent'] for v in vrows),n_ics_with_multiple_transitions=sum(r['n_transitions']>1 for r in irows),n_ics_with_reentrant_sequence=sum(r['n_transitions']>=2 for r in irows),verification_integrations=sum(r['phase']=='verify' for r in records),total_integrations=len(records),scan_offset_range=[float(offsets.min()),float(offsets.max())],scan_step=float(offsets[1]-offsets[0]))
    save(out/'p60_summary.json',summary)
    save(out/'p60_report.json',dict(scope='P60 targeted coupling-axis topology audit of the P59 non-P/re-entrant points',source_sha256=config['source_sha256'],settings=config['settings'],summary=summary,interpretation_guardrails=['A repeated P/M sequence along coupling is finite-resolution evidence of re-entrant finite-time regime selection, not proof of fractal parameter-space structure.','Persistent endpoint reproduction at 3200 and 6400 supports robustness of detected windows but does not prove asymptotic attractor topology.','P60 is intentionally restricted to the ICs that produced unexpected non-P points in P59 and is not a global coupling-axis survey.']))
    print(f'P60 finished: {summary["n_primary_complete"]}/{total} primary complete; {summary["n_transitions"]} transitions; {summary["n_persistent_transitions"]} persistent; {summary["n_ics_with_multiple_transitions"]}/{len(affected)} ICs with multiple transitions')
if __name__=='__main__':main()

