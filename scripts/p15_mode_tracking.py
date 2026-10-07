#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p15_mode_tracking.py

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
import argparse, csv, json, os
from pathlib import Path
from dataclasses import replace
import numpy as np
from p00_model_validation import Parameters
from p12_author_cycle_shooting import basis
from p13_pseudo_arclength import packed, correct, entry


def analyze(row, previous=None, tiny_threshold=1e-6):
    mu=np.array([complex(*v) for v in row['multipliers']]); neutral=np.argmin(abs(mu-1))
    non=np.delete(mu,neutral)
    large=non[np.argsort(-abs(non))[:2]]
    remainder=non[np.argsort(-abs(non))[2:]]
    realmode=remainder[np.argmax(abs(remainder))]
    tiny=np.delete(remainder,np.argmax(abs(remainder)))
    large=large[np.argsort(-large.imag)]
    if abs(large[0].imag)<1e-7 and abs(large[1].imag)<1e-7:
        large=large[np.argsort(-large.real)]
    shape='REAL_PAIR' if max(abs(large.imag))<1e-6 else 'COMPLEX_PAIR'
    bigmods=sorted([float(abs(v)) for v in large],reverse=True)
    tinyvals=sorted([float(abs(v)) for v in tiny],reverse=True)
    tiny_resolved=bool(min(tinyvals)>tiny_threshold and tinyvals[0]/tinyvals[-1]>10)
    output=dict(step=row['step'],zeta=row['zeta'],period=row['period'],
      periodicity_error=row['periodicity_error'],neutral_error=row['neutral_error'],
      unstable_count=row['unstable_count'],large_pair_shape=shape,
      large_pair_real=[float(v.real) for v in large],
      large_pair_imag=[float(v.imag) for v in large],
      large_pair_moduli=bigmods,large_pair_max_modulus=bigmods[0],
      resolved_real_multiplier=float(realmode.real),
      resolved_real_modulus=float(abs(realmode)),
      tiny_cluster_moduli=tinyvals,tiny_cluster_individually_resolved=tiny_resolved,
      tiny_cluster_note='Unordered; do not infer individual mode identity or sign',
      neutral_multiplier=[float(mu[neutral].real),float(mu[neutral].imag)],
      status=row['status'])
    if previous:
        output['large_pair_shape_change']=shape!=previous['large_pair_shape']
        output['large_pair_unit_crossing_candidate']=bool((previous['large_pair_max_modulus']-1)*(bigmods[0]-1)<0)
        output['real_mode_unit_crossing_candidate']=bool((previous['resolved_real_modulus']-1)*(abs(realmode)-1)<0)
        output['zeta_turn_candidate']=False
    else:
        output.update(large_pair_shape_change=False,large_pair_unit_crossing_candidate=False,
                      real_mode_unit_crossing_candidate=False,zeta_turn_candidate=False)
    return output


def export(out,report):
    out.mkdir(parents=True,exist_ok=True)
    path=out/'p15_report.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,path)
    fields=['step','zeta','period','unstable_count','large_pair_shape',
            'large_pair_max_modulus','resolved_real_multiplier','resolved_real_modulus',
            'tiny_cluster_moduli','large_pair_shape_change','large_pair_unit_crossing_candidate',
            'real_mode_unit_crossing_candidate','zeta_turn_candidate','periodicity_error','neutral_error']
    with (out/'p15_branch.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in report['diagnostics']:
            w.writerow({k:json.dumps(r[k]) if isinstance(r.get(k),list) else r.get(k,'') for k in fields})
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        d=report['diagnostics'];z=[r['zeta'] for r in d]
        fig,ax=plt.subplots(figsize=(9,5.5))
        ax.plot(z,[r['large_pair_max_modulus'] for r in d],'.-',label='dominant pair (max)')
        ax.plot(z,[r['resolved_real_modulus'] for r in d],'.-',label='resolved real mode')
        ax.axhline(1,color='black',ls='--',lw=1,label='unit circle')
        ax.set_yscale('log');ax.set_xlabel('Coupling ζ');ax.set_ylabel('Floquet modulus (log)')
        ax.set_title('P15: robust spectral groups (tiny modes excluded)');ax.legend()
        fig.tight_layout();fig.savefig(out/'p15_floquet.png',dpi=300);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        for r in d:
            for re,im in zip(r['large_pair_real'],r['large_pair_imag']):
                ax.scatter(re,im,c=r['zeta'],cmap='viridis',vmin=min(z),vmax=max(z),s=18)
        ax.axhline(0,color='black',lw=.6);ax.set(xlabel='Re μ',ylabel='Im μ',title='P15: dominant Floquet pair trajectory')
        fig.tight_layout();fig.savefig(out/'p15_complex_plane.png',dpi=300);plt.close(fig)
    except Exception as exc:print('Plot warning:',exc,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p14-report',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=0,help='0 = reanalysis only; positive = extend from P14 final two orbits')
    ap.add_argument('--zeta-stop',type=float,default=.88)
    ap.add_argument('--ds',type=float,default=.35);ap.add_argument('--ds-max',type=float,default=.50)
    ap.add_argument('--ds-min',type=float,default=.002)
    ap.add_argument('--rtol',type=float,default=3e-9);ap.add_argument('--atol',type=float,default=3e-11)
    ap.add_argument('--maxfev',type=int,default=50)
    ap.add_argument('--output-dir',type=Path,default=Path('results/author_mode_tracking'))
    a=ap.parse_args();src=json.loads(a.p14_report.read_text());orbits=list(src['orbits'])
    if len(orbits)<2:ap.error('P14 report needs at least two verified orbit records')
    if any('multipliers' not in r for r in orbits):ap.error('P14 report missing raw Floquet multipliers')
    diagnostics=[];events=[]
    for r in orbits:
        d=analyze(r,diagnostics[-1] if diagnostics else None)
        if len(diagnostics)>1:
            d['zeta_turn_candidate']=bool((d['zeta']-diagnostics[-1]['zeta'])*(diagnostics[-1]['zeta']-diagnostics[-2]['zeta'])<0)
        diagnostics.append(d)
        if any(d[k] for k in ('large_pair_shape_change','large_pair_unit_crossing_candidate',
                              'real_mode_unit_crossing_candidate','zeta_turn_candidate')):
            events.append(dict(step=r['step'],zeta=r['zeta'],
                               shape_change=d['large_pair_shape_change'],
                               unit_crossing_large=d['large_pair_unit_crossing_candidate'],
                               unit_crossing_real=d['real_mode_unit_crossing_candidate'],
                               turn=d['zeta_turn_candidate'],status='SPECTRAL_EVENT_NOT_BIFURCATION'))
    report=dict(scope=src['scope'],source_p14=str(a.p14_report),parameters=src['parameters'],
                assumptions=src['assumptions'],hopf=src['hopf'],
                settings=dict(steps=a.steps,zeta_stop=a.zeta_stop,ds=a.ds,ds_max=a.ds_max),
                orbits=orbits,diagnostics=diagnostics,spectral_events=events,failed_attempts=[],
                limitations=['Tiny Floquet multipliers are numerically unresolved and treated as an unordered cluster',
                             'Dominant-pair real/complex morphology changes are NOT unit-circle bifurcations',
                             'Finite continuation does not establish multistability or exclude other attractors',
                             'No physical K0 calibration; omega and W remain test assumptions'])
    export(a.output_dir,report)
    if a.steps<=0:
        print(f'P15 reanalysis: {len(orbits)} orbits; {len(events)} spectral events; output {a.output_dir}',flush=True);return
    p=src['parameters'];ass=src['assumptions'];base=Parameters(a=p['a'],b=p['b'],c=p['c'],d=p['d'],
         delta1=p['delta1'],delta2=p['delta2'],omega=ass['omega'],W=ass['W'],zeta=orbits[-1]['zeta'])
    scale=float(src['settings']['zeta_scale']);ye,u,v=basis(src['hopf'],base)
    x0,x1=[packed(r,scale) for r in orbits[-2:]];ds=a.ds
    if orbits[-1]['zeta']<=a.zeta_stop:
        print('Already reached target; reanalysis completed without new continuation.');return
    for k in range(a.steps):
        tangent=(x1-x0)/np.linalg.norm(x1-x0);sol=None;attempt=0
        while ds>=a.ds_min*(1-1e-10):
            attempt+=1;pred=x1+ds*tangent
            if not 0<pred[7]/scale<.99999:
                report['failed_attempts'].append(dict(step=k,ds=ds,reason='outside coupling domain'));ds*=.5;continue
            candidate,info=correct(pred,tangent,ye,v,base,scale,a.rtol,a.atol,1e-10,a.maxfev)
            if info['success'] and np.linalg.norm(candidate-x1)>.05*ds:sol=candidate;break
            report['failed_attempts'].append(dict(step=k,ds=ds,reason=info['message']));ds*=.5
        if sol is None:print('Stopped: correction failed');break
        row=entry(sol,scale,base,a.rtol,a.atol,src['hopf']['zeta_hopf'],orbits[-1]['step']+1,
                  dict(**info,ds=ds,attempts=attempt))
        if row['status']=='UNRESOLVED':
            report['failed_attempts'].append(dict(step=k,reason='Floquet validation failed'));break
        d=analyze(row,diagnostics[-1]);d['zeta_turn_candidate']=bool((d['zeta']-diagnostics[-1]['zeta'])*(diagnostics[-1]['zeta']-diagnostics[-2]['zeta'])<0)
        orbits.append(row);diagnostics.append(d)
        if any(d[key] for key in ('large_pair_shape_change','large_pair_unit_crossing_candidate',
                                  'real_mode_unit_crossing_candidate','zeta_turn_candidate')):
            events.append(dict(step=row['step'],zeta=row['zeta'],shape_change=d['large_pair_shape_change'],
                  unit_crossing_large=d['large_pair_unit_crossing_candidate'],
                  unit_crossing_real=d['real_mode_unit_crossing_candidate'],turn=d['zeta_turn_candidate'],
                  status='SPECTRAL_EVENT_NOT_BIFURCATION'))
        print(f"step {row['step']}: ζ={row['zeta']:.9f} T={row['period']:.7f} "
              f"max|μ|={d['large_pair_max_modulus']:.4g} unstable={row['unstable_count']} "
              f"pair={d['large_pair_shape']}",flush=True)
        x0,x1=x1,sol;export(a.output_dir,report)
        if row['zeta']<=a.zeta_stop:break
        if attempt==1:ds=min(a.ds_max,ds*1.25)
    print(f'P15: {len(orbits)} orbit records; {len(events)} spectral events; {len(report["failed_attempts"])} failed attempts',flush=True)
if __name__=='__main__':main()

