#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p02_equilibrium_hopf_analysis.py

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
import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from scipy.optimize import root

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from p00_model_validation import ILLUSTRATIVE, Parameters
from p01_rotating_frame_validation import rotating_rhs


def effective_detunings(p: Parameters) -> tuple[float, float]:
    return p.delta1 + p.c*p.omega/p.W, p.delta2 + p.c*p.omega/p.W


def equilibrium_branches(p: Parameters, tol: float = 1e-10) -> list[tuple[str, np.ndarray]]:
    """Analytic principal phase branches; other 2*pi winding copies exist.

    At equilibrium y2=y3=y5=y6=0, sin(theta1)=-D1,
    sin(theta2)=-D2. For zeta != 1 invert theta1=y4-y1,
    theta2=zeta*y1-y4. At zeta=1, equilibrium requires D2=-D1;
    common phase is neutral and must be gauge-fixed.
    """
    D1,D2 = effective_detunings(p)
    if max(abs(D1),abs(D2)) > 1 + tol:
        return []
    if abs(p.zeta-1) < tol and abs(D1+D2)>tol:
        return []
    t1 = np.arcsin(np.clip(-D1,-1,1))
    t2 = np.arcsin(np.clip(-D2,-1,1))
    branches=[]
    for i,theta1 in enumerate((t1, np.pi-t1)):
        for j,theta2 in enumerate((t2,np.pi-t2)):
            if abs(p.zeta-1)<tol:
                # theta2 = -theta1 modulo 2pi, so filter incompatible branches.
                if abs(np.arctan2(np.sin(theta1+theta2),np.cos(theta1+theta2)))>1e-8:
                    continue
                y1=0.0; y4=float(theta1)
            else:
                y1=float((theta1+theta2)/(p.zeta-1))
                y4=y1+float(theta1)
            y=np.array([y1,0,0,y4,0,0],float)
            if np.linalg.norm(rotating_rhs(0,y,p),ord=np.inf)<1e-8:
                branches.append((f'{i}{j}',y))
    return branches


def equilibrium_jacobian(y: np.ndarray,p: Parameters) -> np.ndarray:
    """Exact Jacobian at equilibrium (not the off-equilibrium Jacobian)."""
    th1=y[3]-y[0]; th2=p.zeta*y[0]-y[3]
    c1=np.cos(th1); c2=np.cos(th2)
    J=np.zeros((6,6))
    J[0,1]=J[1,2]=J[3,4]=J[4,5]=1.0
    J[2,:]=[-c1,-p.c-p.d*c1,-p.a-p.b*c1,c1,p.d*c1,p.b*c1]
    J[5,:]=[p.zeta*c2,p.zeta*p.d*c2,p.zeta*p.b*c2,-c2,-p.c-p.d*c2,-p.a-p.b*c2]
    return J


def finite_difference_jacobian(y,p,h=1e-6):
    J=np.zeros((6,6))
    for k in range(6):
        yp=y.copy();ym=y.copy();yp[k]+=h;ym[k]-=h
        J[:,k]=(rotating_rhs(0,yp,p)-rotating_rhs(0,ym,p))/(2*h)
    return J


def evaluate(zeta,p0):
    p=replace(p0,zeta=float(zeta))
    out=[]
    for name,y in equilibrium_branches(p):
        J=equilibrium_jacobian(y,p)
        eig=np.linalg.eigvals(J)
        residual=float(np.linalg.norm(rotating_rhs(0,y,p),ord=np.inf))
        maxreal=float(max(eig.real))
        # zeta=1 has a symmetry-induced neutral eigenvalue; never label
        # asymptotically stable without quotienting this neutral direction.
        if abs(zeta-1)<1e-10:
            label='neutral symmetry; quotient stability requires separate analysis'
        else:
            label='linearly stable' if maxreal < -1e-8 else ('nonhyperbolic' if abs(maxreal)<=1e-8 else 'unstable')
        out.append(dict(zeta=float(zeta),branch=name,y1=float(y[0]),y4=float(y[3]),
                        theta1=float(y[3]-y[0]),theta2=float(p.zeta*y[0]-y[3]),
                        residual=residual,max_real_eigenvalue=maxreal,
                        eigenvalues=[[float(v.real),float(v.imag)] for v in eig],classification=label))
    return out


def candidate_crossings(rows):
    """Screen complex eigenvalue crossings, NOT a full Hopf certification."""
    hits=[]
    branches=sorted(set(r['branch'] for r in rows))
    for branch in branches:
        subset=sorted((r for r in rows if r['branch']==branch),key=lambda r:r['zeta'])
        for left,right in zip(subset,subset[1:]):
            # Use maximum real part among complex eigenvalues; conservative
            # screening can miss non-leading Hopf crossings, so no completeness claim.
            def leading_complex(row):
                v=[a for a,b in row['eigenvalues'] if abs(b)>1e-5]
                return max(v) if v else None
            a,b=leading_complex(left),leading_complex(right)
            if a is not None and b is not None and a*b<=0:
                hits.append(dict(branch=branch,zeta_interval=[left['zeta'],right['zeta']],
                                 real_parts=[a,b],status='CANDIDATE ONLY; requires continuation, transversality and nondegeneracy'))
    return hits


def self_test(p):
    rng=np.random.default_rng(20260929)
    errors=[]; residuals=[]; count=0
    for z in np.r_[rng.uniform(.05,.95,20),[1.0]]:
        pp=replace(p,zeta=float(z))
        for _,y in equilibrium_branches(pp):
            errors.append(float(np.max(np.abs(equilibrium_jacobian(y,pp)-finite_difference_jacobian(y,pp)))))
            residuals.append(float(np.linalg.norm(rotating_rhs(0,y,pp),ord=np.inf)))
            count+=1
    # Explicit symmetry-compatible zeta=1 case.
    D1,_=effective_detunings(p)
    ps=replace(p,zeta=1.,delta2=-D1-p.c*p.omega/p.W)
    sym=equilibrium_branches(ps)
    assert sym, 'Expected gauge-fixed equilibrium at zeta=1'
    assert all(min(abs(np.linalg.eigvals(equilibrium_jacobian(y,ps))))<1e-7 for _,y in sym)
    result=dict(status='PASS' if count>0 and max(errors)<1e-6 and max(residuals)<1e-8 else 'FAIL',
                tested_equilibria=count,max_jacobian_error=max(errors),max_equilibrium_residual=max(residuals),
                zeta_one_symmetry_neutral_mode=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zeta-min',type=float,default=.1)
    ap.add_argument('--zeta-max',type=float,default=.95)
    ap.add_argument('--points',type=int,default=101)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'equilibrium_hopf')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.points<2 or args.zeta_min>=args.zeta_max:
        ap.error('Require --points >= 2 and --zeta-min < --zeta-max')
    p=ILLUSTRATIVE
    test=self_test(p)
    if test['status']!='PASS':
        print(json.dumps(test,indent=2));sys.exit(1)
    if args.self_test:
        print(json.dumps(test,indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for z in np.linspace(args.zeta_min,args.zeta_max,args.points):
        rows.extend(evaluate(z,p))
    candidates=candidate_crossings(rows)
    with (args.output_dir/'p02_equilibria.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['zeta','branch','y1','y4','theta1','theta2','residual','max_real_eigenvalue','classification','eigenvalues'])
        w.writeheader();w.writerows(rows)
    report={'status':'PASS','scope':'Illustrative normalized model; NOT published K0 reproduction',
            'parameters':asdict(p),'effective_detunings':effective_detunings(p),
            'self_test':test,'grid_points':args.points,'equilibria':len(rows),
            'candidate_complex_crossings':candidates,
            'limitations':['K0-to-normalized-coefficient mapping is unavailable; no K0 sweep performed.',
              'Only principal phase branches are enumerated; other 2*pi winding copies exist.',
              'Candidate complex eigenvalue crossings are NOT verified Hopf bifurcations.',
              'At zeta=1 a common-phase neutral mode may exist; gauge reduction is required.',
              'No conclusions about published Hopf points or multistability follow from this scan.']}
    (args.output_dir/'p02_report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for branch in sorted(set(r['branch'] for r in rows)):
        rr=[r for r in rows if r['branch']==branch]
        ax.plot([r['zeta'] for r in rr],[r['max_real_eigenvalue'] for r in rr],label=f'Branch {branch}')
    ax.axhline(0,color='black',lw=1,ls='--')
    ax.set(xlabel='Coupling factor ζ',ylabel='Maximum real Jacobian eigenvalue',
           title='Illustrative equilibrium stability (not reference reproduction)')
    ax.legend(ncol=2);fig.tight_layout()
    fig.savefig(args.output_dir/'p02_eigenvalue_scan.png',dpi=600)
    fig.savefig(args.output_dir/'p02_eigenvalue_scan.pdf')
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print('Outputs:',args.output_dir)

if __name__=='__main__':
    main()

