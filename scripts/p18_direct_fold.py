#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p18_direct_fold.py

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
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p04_periodic_orbit_floquet import integrate_orbit


def shooting(y,T,z,base,phase_anchor,phase,rtol,atol):
    p=replace(base,zeta=float(z))
    end,M,_=integrate_orbit(y,T,p,rtol,atol,True)
    A=np.zeros((7,7));A[:6,:6]=M-np.eye(6)
    A[:6,6]=rotating_rhs(T,end,p);A[6,:6]=phase
    F=np.r_[end-y,np.dot(y-phase_anchor,phase)]
    return F,A,M


def scaled_matrix(A):
    # Row and column equilibration makes the bordered test less sensitive to
    # the strongly unstable Floquet directions and units of period.
    rs=np.maximum(np.linalg.norm(A,axis=1),1e-12)
    B=A/rs[:,None]
    cs=np.maximum(np.linalg.norm(B,axis=0),1e-12)
    return B/cs[None,:],rs,cs


def bordered_value(B,u,v):
    C=np.zeros((8,8));C[:7,:7]=B;C[:7,7]=u;C[7,:7]=v
    if np.linalg.cond(C)>1e12:raise ArithmeticError('Bordered matrix ill-conditioned; reseed required')
    x=np.linalg.solve(C,np.r_[np.zeros(7),1.])
    return float(x[7]),float(np.linalg.cond(C))


def run_one(rec,base,args):
    step=int(rec['source_center_step']);probes=rec['probes']
    anchor=np.asarray(probes[1]['row']['initial_state'],float)
    p_anchor=replace(base,zeta=float(probes[1]['row']['zeta']))
    phase=rotating_rhs(0.,anchor,p_anchor);phase/=np.linalg.norm(phase)
    seeds=[probes[1],probes[0],probes[2]]
    reports=[]
    for stage,(rtol,atol) in enumerate(zip(args.rtols,args.atols)):
        # Restart from the previous stage's independently validated solution.
        if reports and reports[-1]['validated']:
            prior=reports[-1];starts=[np.r_[prior['initial_state'],prior['period'],prior['zeta']]]
        else:
            starts=[np.r_[p['row']['initial_state'],p['row']['period'],p['row']['zeta']] for p in seeds]
        stage_result=None
        for si,start in enumerate(starts):
            try:
                F0,A0,_=shooting(start[:6],start[6],start[7],base,anchor,phase,rtol,atol)
                B0,_,_=scaled_matrix(A0)
                U,S,Vh=np.linalg.svd(B0);u=U[:,-1];v=Vh[-1]
                def residual(w):
                    y=w[:6];T=w[6];z=w[7]
                    if not (args.period_min<T<args.period_max and args.zeta_min<z<args.zeta_max):
                        return np.ones(8)*100.
                    F,A,_=shooting(y,T,z,base,anchor,phase,rtol,atol)
                    B,rs,cs=scaled_matrix(A)
                    g,_=bordered_value(B,u,v)
                    return np.r_[F[:6],F[6],args.fold_weight*g]
                lower=np.r_[np.full(6,-np.inf),args.period_min,args.zeta_min]
                upper=np.r_[np.full(6,np.inf),args.period_max,args.zeta_max]
                sol=least_squares(residual,start,bounds=(lower,upper),max_nfev=args.max_nfev,
                    xtol=args.solver_tol,ftol=args.solver_tol,gtol=args.solver_tol,x_scale='jac')
                y=sol.x[:6];T=float(sol.x[6]);z=float(sol.x[7])
                # Recompute at independently tighter tolerances, not merely
                # trusting the values used in the nonlinear corrector.
                Fr,Ar,Mr=shooting(y,T,z,base,anchor,phase,rtol*.3,atol*.3)
                Br,rs,cs=scaled_matrix(Ar)
                U2,S2,Vh2=np.linalg.svd(Br)
                g,condition=bordered_value(Br,u,v)
                eig=np.linalg.eigvals(Mr);neutral=int(np.argmin(abs(eig-1)))
                non=np.delete(eig,neutral);near=non[np.argmin(abs(non-1))]
                defect=float(np.max(np.abs(Fr)))
                fold_defect=float(abs(g));near_error=float(abs(near-1))
                validated=bool(defect<args.defect_tol and fold_defect<args.fold_tol and
                    abs(eig[neutral]-1)<args.neutral_tol and near_error<args.multiplier_tol)
                stage_result=dict(stage=stage,rtol=rtol,atol=atol,seed_index=si,
                    solver_success=bool(sol.success),nfev=int(sol.nfev),message=sol.message,
                    zeta=z,period=T,initial_state=y.tolist(),shooting_defect=defect,
                    bordered_fold_defect=fold_defect,bordered_condition=condition,
                    singular_values=S2.tolist(),neutral_error=float(abs(eig[neutral]-1)),
                    nearest_nontrivial_plus_one=[float(near.real),float(near.imag)],
                    nearest_nontrivial_plus_one_distance=near_error,
                    unstable_count=int(np.sum(abs(non)>1+1e-3)),
                    max_nontrivial_modulus=float(max(abs(non))),
                    multipliers=[[float(x.real),float(x.imag)] for x in eig],validated=validated)
                if validated:break
            except Exception as exc:
                stage_result=dict(stage=stage,seed_index=si,validated=False,error=str(exc))
        reports.append(stage_result)
        if stage_result['validated']:
            print(f'  step {step}, stage {stage}: zeta={stage_result["zeta"]:.12f} T={stage_result["period"]:.9f} '
                  f'fold_defect={stage_result["bordered_fold_defect"]:.2e} near+1={stage_result["nearest_nontrivial_plus_one"]}',flush=True)
        else:
            print(f'  step {step}, stage {stage}: UNRESOLVED: {stage_result}',flush=True)
            if args.stop_on_failure:break
    good=[x for x in reports if x['validated']]
    convergence=(abs(good[-1]['zeta']-good[-2]['zeta']) if len(good)>=2 else None)
    return dict(source_step=step,stages=reports,zeta_convergence=convergence,
        classification=('NUMERIC_DIRECT_FOLD_SUPPORTED' if len(good)==len(args.rtols) and
            convergence is not None and convergence<args.zeta_convergence_tol else 'UNRESOLVED'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p16-report',type=Path,required=True)
    ap.add_argument('--fold-steps',nargs='+',type=int,default=[50,86])
    ap.add_argument('--rtols',nargs='+',type=float,default=[3e-10,3e-11,1e-11])
    ap.add_argument('--atols',nargs='+',type=float,default=[3e-12,3e-13,1e-13])
    ap.add_argument('--max-nfev',type=int,default=75)
    ap.add_argument('--solver-tol',type=float,default=1e-11)
    ap.add_argument('--fold-weight',type=float,default=1.)
    ap.add_argument('--defect-tol',type=float,default=3e-8)
    ap.add_argument('--fold-tol',type=float,default=1e-5)
    ap.add_argument('--neutral-tol',type=float,default=2e-4)
    ap.add_argument('--multiplier-tol',type=float,default=.02)
    ap.add_argument('--zeta-convergence-tol',type=float,default=2e-6)
    ap.add_argument('--period-min',type=float,default=20.)
    ap.add_argument('--period-max',type=float,default=45.)
    ap.add_argument('--zeta-min',type=float,default=.88)
    ap.add_argument('--zeta-max',type=float,default=.92)
    ap.add_argument('--stop-on-failure',action='store_true')
    ap.add_argument('--output-dir',type=Path,default=Path('results/author_direct_folds'))
    args=ap.parse_args()
    if len(args.rtols)!=len(args.atols):ap.error('rtols and atols must have same length')
    src=json.loads(args.p16_report.read_text())
    base=Parameters(**src['parameters'],omega=src['assumptions']['omega'],W=src['assumptions']['W'],zeta=.9)
    result=dict(scope=src['scope'],source_p16=str(args.p16_report),parameters=src['parameters'],
        assumptions=src['assumptions'],settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        folds=[],errors=[],limitations=['Direct bordered singularity and Floquet checks are numerical evidence, not genericity proof',
        'The two tiny Floquet modes are numerically unresolved and must not be individually tracked',
        'No physical K0 calibration; omega and W are test assumptions'])
    for step in args.fold_steps:
        rec=next((r for r in src['refinements'] if int(r['source_center_step'])==step),None)
        if rec is None:result['errors'].append(dict(step=step,error='P16 seed missing'));continue
        print(f'P18 direct fold candidate {step}',flush=True)
        try:result['folds'].append(run_one(rec,base,args))
        except Exception as exc:result['errors'].append(dict(step=step,error=str(exc)));print(' ERROR:',exc,flush=True)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        temp=args.output_dir/'p18_report.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');os.replace(temp,args.output_dir/'p18_report.json')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'p18_folds.csv').open('w',newline='') as fh:
        fields=['source_step','stage','rtol','atol','zeta','period','shooting_defect','bordered_fold_defect',
            'neutral_error','nearest_nontrivial_plus_one_distance','unstable_count','validated','classification','error']
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
        for fold in result['folds']:
            for stage in fold['stages']:
                writer.writerow({k:(fold['source_step'] if k=='source_step' else fold['classification'] if k=='classification' else stage.get(k)) for k in fields})
    print(f'P18 finished: {len(result["folds"])} candidates, {len(result["errors"])} errors; {args.output_dir}',flush=True)
if __name__=='__main__':main()

