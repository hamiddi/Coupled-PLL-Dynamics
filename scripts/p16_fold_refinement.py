#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p16_fold_refinement.py

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
import argparse,csv,json,os
from pathlib import Path
from dataclasses import replace
import numpy as np
from p00_model_validation import Parameters
from p12_author_cycle_shooting import basis
from p13_pseudo_arclength import packed,correct,entry


def turns(rows):
    return [i for i in range(1,len(rows)-1) if
            (rows[i]['zeta']-rows[i-1]['zeta'])*(rows[i+1]['zeta']-rows[i]['zeta'])<0]


def local_poly(rows,i,scale):
    x=np.stack([packed(r,scale) for r in rows[i-1:i+2]])
    ds0=np.linalg.norm(x[1]-x[0]);ds1=np.linalg.norm(x[2]-x[1])
    if min(ds0,ds1)<1e-8:raise ValueError('Adjacent orbit states nearly identical')
    s=np.array([-ds0,0.,ds1]);coefs=np.polyfit(s,x,2)
    a,b,c=coefs[:,7]/scale
    if abs(a)<1e-14:raise ValueError('Flat quadratic coupling fit')
    vertex=-b/(2*a)
    if not s[0]<vertex<s[2]:raise ValueError('Fitted extremum outside local bracket')
    return s,coefs,float(vertex)


def classify(mu,neutral_tol=.02):
    vals=np.array([complex(*v) for v in mu]); neutral=int(np.argmin(abs(vals-1)))
    non=np.delete(vals,neutral)
    near=non[np.argmin(abs(non-1))]
    return dict(neutral_error=float(abs(vals[neutral]-1)),
        nearest_nontrivial_plus_one=[float(near.real),float(near.imag)],
        nearest_nontrivial_plus_one_distance=float(abs(near-1)),
        unstable_count=int(np.sum(abs(non)>1+1e-3)),
        dominant_modulus=float(max(abs(non))),
        all_multipliers=[[float(v.real),float(v.imag)] for v in vals])


def refine_one(rows,i,scale,base,hopf,ye,v,args):
    s,poly,sv=local_poly(rows,i,scale)
    span=min(abs(sv-s[0]),abs(s[2]-sv))
    h=min(args.probe_ds,0.4*span)
    if h<1e-5:raise ValueError('Insufficient local arclength for probes')
    probes=[]
    for label,si in [('left',sv-h),('vertex',sv),('right',sv+h)]:
        predictor=np.polyval(poly,si)
        tangent=2*poly[0]*si+poly[1]
        tangent/=np.linalg.norm(tangent)
        w,info=correct(predictor,tangent,ye,v,base,scale,args.rtol,args.atol,1e-11,args.maxfev)
        if not info['success']:raise RuntimeError(f'{label} corrector failed: {info}')
        row=entry(w,scale,base,args.rtol,args.atol,hopf,f'fold_{i}_{label}',info)
        if row['status']=='UNRESOLVED':raise RuntimeError(f'{label} Floquet validation unresolved')
        probes.append(dict(label=label,arclength=float(si),row=row,spectrum=classify(row['multipliers'])))
    z=[p['row']['zeta'] for p in probes]
    slope_left=(z[1]-z[0])/h;slope_right=(z[2]-z[1])/h
    geometric=bool(slope_left*slope_right<0)
    plus=probes[1]['spectrum']['nearest_nontrivial_plus_one_distance']
    # A loose numerical screen, not a mathematical proof of a generic fold.
    spectral=bool(plus<args.plus_one_tol)
    status='FOLD_CANDIDATE_REFINED' if geometric and spectral else ('GEOMETRIC_TURN_ONLY' if geometric else 'UNRESOLVED')
    return dict(source_center_step=rows[i]['step'],source_center_index=i,
        source_bracket_steps=[r['step'] for r in rows[i-1:i+2]],
        original_zeta=[float(r['zeta']) for r in rows[i-1:i+2]],
        quadratic_vertex_arclength=sv,probe_spacing=h,
        corrected_slope_left=float(slope_left),corrected_slope_right=float(slope_right),
        geometric_turn_verified=geometric,additional_plus_one_screen=spectral,
        status=status,probes=probes,
        interpretation='Numerically refined fold candidate; genericity and independent tolerance convergence remain to be established' if geometric and spectral else 'Do not claim a periodic-orbit fold from this result')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p15-report',required=True,type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('results/author_fold_refinement'))
    p.add_argument('--fold-steps',type=int,nargs='*',default=None,help='Optional P15 step labels to refine; default all detected turns')
    p.add_argument('--probe-ds',type=float,default=.04)
    p.add_argument('--plus-one-tol',type=float,default=.05)
    p.add_argument('--rtol',type=float,default=1e-10);p.add_argument('--atol',type=float,default=1e-12)
    p.add_argument('--maxfev',type=int,default=100)
    args=p.parse_args()
    src=json.loads(args.p15_report.read_text());rows=src['orbits']
    if len(rows)<3:raise ValueError('At least three P15 orbits required')
    if any('initial_state' not in r or 'multipliers' not in r for r in rows):
        raise ValueError('P15 full JSON required: branch CSV or console log cannot substitute orbit states and spectra')
    prm=src['parameters'];ass=src['assumptions'];scale=float(src.get('settings',{}).get('zeta_scale',1000.))
    # P15 settings omit scale; infer from parent P14/P13 convention only when absent.
    base=Parameters(**prm,omega=ass['omega'],W=ass['W'],zeta=rows[-1]['zeta'])
    ye,_,v=basis(src['hopf'],base)
    candidates=turns(rows)
    if args.fold_steps is not None:candidates=[i for i in candidates if rows[i]['step'] in args.fold_steps]
    result=dict(scope=src['scope'],source_p15=str(args.p15_report),
        parameters=prm,assumptions=ass,settings=dict(probe_ds=args.probe_ds,plus_one_tol=args.plus_one_tol,
        rtol=args.rtol,atol=args.atol,zeta_scale=scale),
        source_orbit_count=len(rows),detected_turn_steps=[rows[i]['step'] for i in turns(rows)],
        source_spectral_events=src.get('spectral_events',[]),refinements=[],errors=[],
        limitations=['A second +1 multiplier and corrected coupling turn are numerical screens, not a proof of genericity',
          'The first fold may have near-unit mode ambiguity: independently check tolerance and probe-size convergence',
          'No physical K0 calibration; omega and W are test assumptions'])
    for i in candidates:
        print(f'Refining candidate turn centered on step {rows[i]["step"]}',flush=True)
        try:
            rec=refine_one(rows,i,scale,base,src['hopf']['zeta_hopf'],ye,v,args)
            result['refinements'].append(rec)
            print(f'  {rec["status"]}: ζvertex={rec["probes"][1]["row"]["zeta"]:.10f} '
                  f'plus_one_distance={rec["probes"][1]["spectrum"]["nearest_nontrivial_plus_one_distance"]:.5g}',flush=True)
        except Exception as exc:
            result['errors'].append(dict(step=rows[i]['step'],reason=str(exc)))
            print(f'  UNRESOLVED: {exc}',flush=True)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    dest=out/'p16_report.json';tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,dest)
    with (out/'p16_folds.csv').open('w',newline='') as f:
        fields=['source_center_step','status','vertex_zeta','vertex_period','slope_left','slope_right',
                'plus_one_distance','vertex_unstable_count','vertex_periodicity_error']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in result['refinements']:
            mid=r['probes'][1];w.writerow(dict(source_center_step=r['source_center_step'],status=r['status'],
                vertex_zeta=mid['row']['zeta'],vertex_period=mid['row']['period'],
                slope_left=r['corrected_slope_left'],slope_right=r['corrected_slope_right'],
                plus_one_distance=mid['spectrum']['nearest_nontrivial_plus_one_distance'],
                vertex_unstable_count=mid['row']['unstable_count'],
                vertex_periodicity_error=mid['row']['periodicity_error']))
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5.5));ax.plot([r['zeta'] for r in rows],[r['period'] for r in rows],'.-',ms=3,lw=1,label='P15 branch')
        for r in result['refinements']:
            mid=r['probes'][1]['row'];ax.scatter(mid['zeta'],mid['period'],s=90,marker='x',label=f'Candidate step {r["source_center_step"]}')
        ax.set(xlabel='Coupling ζ',ylabel='Period',title='P16: candidate periodic-orbit folds')
        ax.legend();fig.tight_layout();fig.savefig(out/'p16_fold_diagram.png',dpi=300);plt.close(fig)
    except Exception as exc:print('Plot warning:',exc)
    print(f'P16: {len(result["refinements"])} refined; {len(result["errors"])} unresolved; output {out}',flush=True)
if __name__=='__main__':main()

