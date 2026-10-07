#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p14_extended_floquet.py

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
import argparse,csv,json,os,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment
from p00_model_validation import Parameters
from p12_author_cycle_shooting import basis
from p13_pseudo_arclength import packed,correct,entry


def ordered_multipliers(row, previous=None):
    """Match nontrivial eigenvalues by minimum complex-plane displacement.

    Matching is diagnostic; near collisions it is ambiguous and is flagged.
    """
    vals=np.array([complex(*v) for v in row['multipliers']],complex)
    neutral=int(np.argmin(abs(vals-1)))
    non=np.delete(vals,neutral)
    if previous is None:
        non=non[np.lexsort((non.imag,non.real,-abs(non)))]
        return non,False,0.0
    cost=abs(previous[:,None]-non[None,:])/(1+abs(previous[:,None]))
    ri,ci=linear_sum_assignment(cost)
    ordered=np.empty_like(non);ordered[ri]=non[ci]
    ambiguity=False
    for i,j in zip(ri,ci):
        sorted_cost=np.sort(cost[i]);
        if len(sorted_cost)>1 and sorted_cost[1]-sorted_cost[0]<0.02:ambiguity=True
    return ordered,ambiguity,float(np.max(cost[ri,ci]))


def decorate(row, previous=None):
    vals,ambig,dist=ordered_multipliers(row,previous)
    row['tracked_nontrivial_multipliers']=[[float(v.real),float(v.imag)] for v in vals]
    row['tracked_moduli']=[float(abs(v)) for v in vals]
    row['tracked_angles_rad']=[float(np.angle(v)) for v in vals]
    row['tracking_ambiguous']=ambig
    row['tracking_max_relative_step']=dist
    return vals


def write_atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)


def export(out,result):
    write_atomic(out/'p14_report.json',result)
    cols=['step','zeta','period','periodicity_error','neutral_error','unstable_count',
          'max_nontrivial_modulus','tracking_ambiguous','tracking_max_relative_step',
          'zeta_turn_candidate','unit_crossing_candidates','status']
    with (out/'p14_branch.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader()
        for r in result['orbits']:
            x=dict(r);x['unit_crossing_candidates']=';'.join(map(str,r.get('unit_crossing_candidates',[])))
            w.writerow({k:x.get(k,'') for k in cols})
    with (out/'p14_multipliers.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['step','zeta','mode','real','imag','modulus','angle_rad','tracking_ambiguous'])
        for r in result['orbits']:
            for j,(re,im) in enumerate(r['tracked_nontrivial_multipliers']):
                w.writerow([r['step'],r['zeta'],j,re,im,r['tracked_moduli'][j],r['tracked_angles_rad'][j],r['tracking_ambiguous']])
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5.5))
        for j in range(5):
            ax.plot([r['zeta'] for r in result['orbits']],
                    [r['tracked_moduli'][j] for r in result['orbits']],'.-',label=f'mode {j}')
        ax.axhline(1,color='black',ls='--',lw=1,label='unit circle')
        ax.set_yscale('log');ax.set_xlabel('Coupling ζ');ax.set_ylabel('Nontrivial Floquet modulus (log)')
        ax.set_title('P14: individually tracked Floquet multipliers')
        ax.legend(ncol=3);fig.tight_layout();fig.savefig(out/'p14_floquet.png',dpi=300);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5));ax.plot([r['zeta'] for r in result['orbits']],
             [r['period'] for r in result['orbits']],'.-')
        ax.set(xlabel='Coupling ζ',ylabel='Period',title='P14: restarted periodic-orbit continuation')
        fig.tight_layout();fig.savefig(out/'p14_period.png',dpi=300);plt.close(fig)
    except Exception as exc:print('Plot unavailable:',exc,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p13-report',type=Path,required=True,help='Full P13 JSON report; final two orbits are restart seeds')
    ap.add_argument('--steps',type=int,default=120)
    ap.add_argument('--ds',type=float,default=.35)
    ap.add_argument('--ds-max',type=float,default=.50)
    ap.add_argument('--ds-min',type=float,default=.002)
    ap.add_argument('--zeta-stop',type=float,default=.90)
    ap.add_argument('--rtol',type=float,default=3e-9)
    ap.add_argument('--atol',type=float,default=3e-11)
    ap.add_argument('--maxfev',type=int,default=45)
    ap.add_argument('--output-dir',type=Path,default=Path('results/author_extended_floquet'))
    args=ap.parse_args()
    if not args.p13_report.exists():ap.error('P13 report not found')
    if args.steps<1 or args.ds_min<=0 or args.ds_max<args.ds_min or args.ds<=0:ap.error('Invalid continuation settings')
    src=json.loads(args.p13_report.read_text());old=src.get('orbits',[])
    good=[r for r in old if r.get('status')!='UNRESOLVED' and 'initial_state' in r]
    if len(good)<2:ap.error('P13 report must contain at least two verified periodic orbits')
    a=src['assumptions'];omega=float(a['omega']);W=float(a['W'])
    p=src['parameters'];base=Parameters(a=float(p['a']),b=float(p['b']),c=float(p['c']),d=float(p['d']),
        delta1=float(p['delta1']),delta2=float(p['delta2']),omega=omega,W=W,zeta=float(good[-1]['zeta']))
    hopf=src['hopf'];scale=float(src['settings']['zeta_scale'])
    # Reconstruct the same phase section used by P13 from its frozen P11 report.
    ye,u,v=basis(hopf,base)
    x0,x1=[packed(r,scale) for r in good[-2:]]
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    rows=[];prev=None
    for k,seed in enumerate(good[-2:]):
        row=entry(packed(seed,scale),scale,base,args.rtol,args.atol,hopf['zeta_hopf'],k)
        prev=decorate(row,prev);row['zeta_turn_candidate']=False;row['unit_crossing_candidates']=[]
        rows.append(row)
    ds=args.ds;failures=[];events=[]
    result=dict(scope='Harb representative normalized benchmark; not physical K0 calibration',
       source_p13=str(args.p13_report),parameters=p,assumptions=a,hopf=hopf,
       settings=dict(steps=args.steps,ds=args.ds,ds_max=args.ds_max,ds_min=args.ds_min,zeta_stop=args.zeta_stop,zeta_scale=scale),
       orbits=rows,failed_attempts=failures,candidate_events=events,
       limitations=['Mode matching near collisions is ambiguous and flagged',
                    'Unit-circle brackets are candidates, not independently verified secondary bifurcations',
                    'Finite continuation cannot exclude other attractors',
                    'No physical K0 mapping; omega and W are test settings'])
    export(out,result)
    if x1[7]/scale<=args.zeta_stop:
        print('Restart already at/below zeta stop; no continuation needed.',flush=True);return
    for k in range(args.steps):
        tangent=(x1-x0)/np.linalg.norm(x1-x0)
        attempt=0;sol=None
        while ds>=args.ds_min*(1-1e-10):
            attempt+=1;pred=x1+ds*tangent
            if not 0<pred[7]/scale<.99999:
                failures.append(dict(step=k+2,ds=ds,reason='predictor outside domain'));ds*=.5;continue
            candidate,info=correct(pred,tangent,ye,v,base,scale,args.rtol,args.atol,1e-10,args.maxfev)
            if info['success'] and np.linalg.norm(candidate-x1)>.05*ds:
                sol=candidate;break
            failures.append(dict(step=k+2,ds=ds,reason=info['message'],residual=info['residual']));ds*=.5
        if sol is None:
            print('Stopping: corrector failed at minimum step',flush=True);break
        row=entry(sol,scale,base,args.rtol,args.atol,hopf['zeta_hopf'],k+2,
                  dict(**info,ds=ds,attempts=attempt))
        if row['status']=='UNRESOLVED':
            failures.append(dict(step=k+2,ds=ds,reason='orbit or neutral Floquet validation failed'));break
        vals=decorate(row,prev)
        oldmods=np.array(rows[-1]['tracked_moduli']);newmods=abs(vals)
        crossings=[j for j in range(5) if (oldmods[j]-1)*(newmods[j]-1)<0]
        row['unit_crossing_candidates']=crossings
        row['zeta_turn_candidate']=bool((row['zeta']-rows[-1]['zeta'])*(rows[-1]['zeta']-rows[-2]['zeta'])<0)
        if crossings or row['zeta_turn_candidate'] or row['unstable_count']!=rows[-1]['unstable_count']:
            events.append(dict(step=k+2,zeta_interval=[rows[-1]['zeta'],row['zeta']],
                 modes=crossings,turn=row['zeta_turn_candidate'],
                 unstable_count_change=row['unstable_count']-rows[-1]['unstable_count'],
                 tracking_ambiguous=row['tracking_ambiguous'],status='CANDIDATE_ONLY'))
        rows.append(row);prev=vals
        print(f"step {k+2}: zeta={row['zeta']:.10f} T={row['period']:.7f} "
              f"max|mu|={row['max_nontrivial_modulus']:.5g} unstable={row['unstable_count']} "
              f"events={crossings} ds={ds:.4g}",flush=True)
        x0,x1=x1,sol
        export(out,result)
        if row['zeta']<=args.zeta_stop:break
        if attempt==1:ds=min(args.ds_max,ds*1.25)
    print(f"P14: {len(rows)} records (including 2 restart seeds); "
          f"{len(failures)} failed attempts; {len(events)} candidate events. Output: {out}",flush=True)
    if len(rows)<3:sys.exit(2)
if __name__=='__main__':main()

