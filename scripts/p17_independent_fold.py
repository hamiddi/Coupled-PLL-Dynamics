#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p17_independent_fold.py

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
from scipy.optimize import minimize_scalar,brentq
from p00_model_validation import Parameters
from p12_author_cycle_shooting import basis
from p13_pseudo_arclength import packed,correct,entry
from p16_fold_refinement import classify

def make_local(rec,scale):
    probes=rec['probes']
    s=np.array([float(p['arclength']) for p in probes]);x=np.stack([packed(p['row'],scale) for p in probes]);
    center=float(s[1]);h=float(s[2]-s[1]);q=(s-center)/h
    coef=np.polyfit(q,x,2)
    return center,h,coef

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p16-report',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path('results/author_fold_verification'))
    ap.add_argument('--fold-steps',type=int,nargs='*',default=[50,86])
    ap.add_argument('--probe-fractions',type=float,nargs='+',default=[.5,.25,.125])
    ap.add_argument('--rtols',type=float,nargs='+',default=[1e-10,3e-11,1e-11])
    ap.add_argument('--atols',type=float,nargs='+',default=[1e-12,3e-13,1e-13])
    ap.add_argument('--maxfev',type=int,default=90)
    ap.add_argument('--scan-points',type=int,default=13)
    args=ap.parse_args()
    if not(len(args.rtols)==len(args.atols)==len(args.probe_fractions)) or any(v<=0 for v in args.probe_fractions):ap.error('Each probe fraction needs one positive rtol/atol')
    src=json.loads(args.p16_report.read_text());scale=float(src['settings']['zeta_scale']);base=Parameters(**src['parameters'],omega=src['assumptions']['omega'],W=src['assumptions']['W'],zeta=.9)
    # Phase anchor uses P16's corrected vertex; local orbit flow is transverse to phase section.
    from p01_rotating_frame_validation import rotating_rhs
    result=dict(scope=src['scope'],source_p16=str(args.p16_report),parameters=src['parameters'],assumptions=src['assumptions'],settings=vars(args)|{'p16_report':str(args.p16_report),'output_dir':str(args.output_dir)},folds=[],errors=[],limitations=['Finite precision numerical verification, not generic-fold proof','Fixed P16 local quadratic chart and phase anchor; chart breakdown must be checked','No physical K0 calibration'])
    for rec in src['refinements']:
        step=int(rec['source_center_step'])
        if step not in args.fold_steps:continue
        print('P17 candidate',step,flush=True)
        try:
            center,h,coef=make_local(rec,scale)
            anchor=np.array(rec['probes'][1]['row']['initial_state'],float);anchor_p=replace(base,zeta=rec['probes'][1]['row']['zeta'])
            phase=rotating_rhs(0,anchor,anchor_p);phase/=np.linalg.norm(phase)
            kind='min' if rec['corrected_slope_left']<0 else 'max'
            stages=[]
            for fraction,rtol,atol in zip(args.probe_fractions,args.rtols,args.atols):
                cache={}
                def evaluate(q):
                    q=float(q);key=round(q,12)
                    if key in cache:return cache[key]
                    pred=np.polyval(coef,q);tang=2*coef[0]*q+coef[1];tang/=np.linalg.norm(tang)
                    w,info=correct(pred,tang,anchor,phase,base,scale,rtol,atol,1e-11,args.maxfev)
                    if w is None:raise RuntimeError(f'correction produced no state at q={q}: {info}')
                    # MINPACK can return status=5 after reaching a numerically excellent
                    # solution. Accept only after independent high-accuracy orbit/Floquet
                    # validation below; never infer convergence from residual alone.
                    stagnated=not info['success']
                    if stagnated and (info['residual'] is None or info['residual']>1e-9):
                        raise RuntimeError(f'correction failed at q={q}: {info}')
                    row=entry(w,scale,base,rtol,atol,0.,f'p17_{step}_{q:.9f}',info)
                    if (row['status']=='UNRESOLVED' or row['periodicity_error']>1e-8
                            or row['neutral_error']>1e-3):
                        raise RuntimeError(f'independent orbit/Floquet validation failed at q={q}: '
                                           f'periodicity={row["periodicity_error"]}, neutral={row["neutral_error"]}')
                    if stagnated:
                        info=dict(info,accepted_after_validation=True,
                                  validation_periodicity_error=row['periodicity_error'],
                                  validation_neutral_error=row['neutral_error'])
                    spec=classify(row['multipliers']);obj=dict(q=q,zeta=row['zeta'],period=row['period'],initial_state=row['initial_state'],periodicity_error=row['periodicity_error'],neutral_error=row['neutral_error'],unstable_count=row['unstable_count'],multipliers=row['multipliers'],nontrivial_plus_one=spec['nearest_nontrivial_plus_one'],plus_one_distance=spec['nearest_nontrivial_plus_one_distance'],max_modulus=row['max_nontrivial_modulus'],correction=info)
                    cache[key]=obj;return obj
                # Locate extremum using corrected orbits, not the original quadratic vertex.
                radius=max(.20,min(1.,fraction*2.))
                opt=minimize_scalar(lambda q:(1 if kind=='min' else -1)*evaluate(q)['zeta'],bounds=(-radius,radius),method='bounded',options={'xatol':min(1e-5,fraction*.001),'maxiter':32})
                if not opt.success:raise RuntimeError('coupling extremum optimizer did not converge')
                qc=float(opt.x);vertex=evaluate(qc);dq=float(fraction)
                left=evaluate(qc-dq);right=evaluate(qc+dq)
                slope_l=(vertex['zeta']-left['zeta'])/dq;slope_r=(right['zeta']-vertex['zeta'])/dq
                # Follow a continuous real eigenvalue near +1 when it is spectrally isolated.
                # Report all roots from sampled signed distances; a complex mode cannot be used.
                scan=[]
                for q in np.linspace(qc-dq,qc+dq,args.scan_points):
                    e=evaluate(q);v=complex(*e['nontrivial_plus_one']);scan.append((float(q),float(v.real-1) if e['neutral_error']<1e-5 and abs(v.imag)<1e-6 and abs(v-1)<.6 else None))
                crossings=[]
                for (qa,fa),(qb,fb) in zip(scan,scan[1:]):
                    if fa is None or fb is None or fa*fb>0:continue
                    try:
                        root=brentq(lambda q:complex(*evaluate(q)['nontrivial_plus_one']).real-1,qa,qb,xtol=1e-8)
                        candidate=evaluate(root)
                        if (candidate['neutral_error']>1e-5 or abs(complex(*candidate['nontrivial_plus_one']).imag)>1e-6):
                            crossings.append({'bracket':[qa,qb],'status':'AMBIGUOUS_COMPLEX_NEAR_PLUS_ONE','candidate':candidate})
                        else:crossings.append({'bracket':[qa,qb],'status':'REAL_PLUS_ONE_CANDIDATE','candidate':candidate})
                    except Exception as exc:crossings.append({'bracket':[qa,qb],'error':str(exc)})
                stage=dict(probe_fraction=fraction,rtol=rtol,atol=atol,vertex=vertex,left=left,right=right,slope_left=slope_l,slope_right=slope_r,geometric_turn=bool(slope_l*slope_r<0),plus_one_crossings=crossings,scan=scan)
                stages.append(stage)
                print(f'  fraction={fraction:g} zeta={vertex["zeta"]:.11f} T={vertex["period"]:.8f} near+1={vertex["nontrivial_plus_one"]} turn={stage["geometric_turn"]} roots={len(crossings)}',flush=True)
            z=[s['vertex']['zeta'] for s in stages];dist=[s['vertex']['plus_one_distance'] for s in stages]
            result['folds'].append(dict(source_step=step,kind=kind,stages=stages,vertex_zeta_spread=float(max(z)-min(z)),last_plus_one_distance=dist[-1],classification='NUMERIC_FOLD_SUPPORTED' if all(s['geometric_turn'] for s in stages) and dist[-1]<.02 and any(any(c.get('status')=='REAL_PLUS_ONE_CANDIDATE' for c in s['plus_one_crossings']) for s in stages) else 'UNRESOLVED_REQUIRES_FURTHER_TESTS'))
        except Exception as exc:
            result['errors'].append(dict(step=step,error=str(exc)));print('  ERROR',exc,flush=True)
        out=args.output_dir;out.mkdir(parents=True,exist_ok=True);tmp=out/'p17_report.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,out/'p17_report.json')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    with (out/'p17_folds.csv').open('w',newline='') as fh:
        fields=['step','stage','probe_fraction','rtol','atol','vertex_zeta','vertex_period','near_plus_one_real','near_plus_one_imag','plus_one_distance','unstable_count','slope_left','slope_right','geometric_turn','crossing_count','periodicity_error','neutral_error'];writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
        for f in result['folds']:
            for i,s in enumerate(f['stages']):
                v=s['vertex'];writer.writerow(dict(step=f['source_step'],stage=i,probe_fraction=s['probe_fraction'],rtol=s['rtol'],atol=s['atol'],vertex_zeta=v['zeta'],vertex_period=v['period'],near_plus_one_real=v['nontrivial_plus_one'][0],near_plus_one_imag=v['nontrivial_plus_one'][1],plus_one_distance=v['plus_one_distance'],unstable_count=v['unstable_count'],slope_left=s['slope_left'],slope_right=s['slope_right'],geometric_turn=s['geometric_turn'],crossing_count=len(s['plus_one_crossings']),periodicity_error=v['periodicity_error'],neutral_error=v['neutral_error']))
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        if not result['folds']:
            print('plot skipped: no completed fold stages',flush=True)
        else:
            fig,axes=plt.subplots(1,len(result['folds']),figsize=(6*max(1,len(result['folds'])),5),squeeze=False)
            for ax,f in zip(axes[0],result['folds']):
                for s in f['stages']:
                    vals=[s['left'],s['vertex'],s['right']];ax.plot([v['period'] for v in vals],[v['zeta'] for v in vals],'.-',label=f'probe {s["probe_fraction"]:g}')
                ax.set(xlabel='Orbit period',ylabel='Coupling ζ',title=f'P17 local turn near step {f["source_step"]}');ax.legend()
            fig.tight_layout();fig.savefig(out/'p17_folds.png',dpi=300);plt.close(fig)
    except Exception as exc:print('plot warning:',exc)
    print('P17',len(result['folds']),'completed',len(result['errors']),'errors; outputs',out,flush=True)
if __name__=='__main__':main()

