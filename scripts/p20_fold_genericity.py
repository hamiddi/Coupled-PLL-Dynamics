#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p20_fold_genericity.py

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
from scipy.optimize import root
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p04_periodic_orbit_floquet import integrate_orbit
from p19_multiple_shooting import system,initial_segments

def unpack_report(path):
    src=json.loads(path.read_text());base=Parameters(**src['parameters'],omega=src['assumptions']['omega'],W=src['assumptions']['W'],zeta=.9)
    return src,base

def evaluate(w,m,base,anchor,phase,rtol,atol):
    return system(w,m,base,anchor,phase,rtol,atol)

def match_spectra(a,b):
    # Hungarian matching; report only well-separated modes and near-unit cluster.
    from scipy.optimize import linear_sum_assignment
    ii,jj=linear_sum_assignment(abs(a[:,None]-b[None,:]))
    return float(np.max(abs(a[ii]-b[jj]))), [[float(a[i].real),float(a[i].imag),float(b[j].real),float(b[j].imag),float(abs(a[i]-b[j]))] for i,j in zip(ii,jj)]

def analyze(stage,base,args,step):
    m=args.segments;z=float(stage['zeta']);T=float(stage['period']);p=replace(base,zeta=z)
    y=np.asarray(stage['initial_state'],float)
    Y=np.asarray(stage.get('segment_states',[]),float)
    if Y.shape!=(m,6):
        Y=initial_segments(y,T,p,m,args.rtol,args.atol)
    anchor=y.copy();phase=rotating_rhs(0,y,p);phase/=np.linalg.norm(phase)
    w=np.r_[Y.ravel(),T,z]; n=6*m+1
    F,J,Jz,Ms=evaluate(w,m,base,anchor,phase,args.rtol,args.atol)
    # Re-polish the P19 orbit at fixed zeta to avoid residual contamination.
    def fixed(q):
        ww=np.r_[q[:6*m+1],z]
        return evaluate(ww,m,base,anchor,phase,args.rtol,args.atol)[0]
    pol=root(fixed,w[:6*m+1],method='hybr',options={'xtol':1e-10,'maxfev':args.maxfev})
    if np.max(abs(fixed(pol.x)))<args.polish_tol:
        w=np.r_[pol.x,z]
    F,J,Jz,Ms=evaluate(w,m,base,anchor,phase,args.rtol,args.atol)
    U,s,Vh=np.linalg.svd(J,full_matrices=True)
    v=Vh[-1];left=U[:,-1]
    # Ensure v points to increasing period (when possible) to fix orientation.
    if v[-1]<0:v=-v
    trans=float(left@Jz)
    h=args.curvature_h
    Fplus=evaluate(np.r_[w[:n]+h*v,z],m,base,anchor,phase,args.rtol,args.atol)[0]
    Fminus=evaluate(np.r_[w[:n]-h*v,z],m,base,anchor,phase,args.rtol,args.atol)[0]
    quad=float(left@(Fplus-2*F+Fminus)/(2*h*h))
    # Compare two finite-difference increments for curvature stability.
    h2=h/2
    Fp2=evaluate(np.r_[w[:n]+h2*v,z],m,base,anchor,phase,args.rtol,args.atol)[0]
    Fm2=evaluate(np.r_[w[:n]-h2*v,z],m,base,anchor,phase,args.rtol,args.atol)[0]
    quad2=float(left@(Fp2-2*F+Fm2)/(2*h2*h2))
    # Use v and z as the pseudo-arclength gauge; solve full periodicity at prescribed
    # displacement from the fold. The normal form predicts dz/ds²=-quad/trans.
    branches=[]
    for ds in [-args.branch_step,args.branch_step]:
        guess=np.r_[w[:n]+ds*v,z-(quad/trans)*ds*ds if abs(trans)>1e-15 else z]
        def branch_eq(x):
            ff=evaluate(x,m,base,anchor,phase,args.rtol,args.atol)[0]
            return np.r_[ff,np.dot(x[:n]-w[:n],v)-ds]
        try:
            sol=root(branch_eq,guess,method='hybr',options={'xtol':1e-10,'maxfev':args.maxfev})
            err=float(np.max(abs(branch_eq(sol.x))))
            branches.append(dict(ds=ds,zeta=float(sol.x[-1]),period=float(sol.x[6*m]),residual=err,accepted=bool(err<args.branch_tol),solver_success=bool(sol.success)))
        except Exception as e:branches.append(dict(ds=ds,error=str(e),accepted=False))
    # Product M_{m-1}...M_0 and independently integrated full-period monodromy.
    product=np.eye(6)
    for A in Ms:product=A@product
    y0=w[:6];end,Msingle,_=integrate_orbit(y0,float(w[6*m]),replace(base,zeta=float(w[-1])),args.rtol*.3,args.atol*.3,True)
    mu_prod=np.linalg.eigvals(product);mu_single=np.linalg.eigvals(Msingle)
    diff,matched=match_spectra(mu_prod,mu_single)
    near_prod=sorted(mu_prod,key=lambda x:abs(x-1))[:2]
    near_single=sorted(mu_single,key=lambda x:abs(x-1))[:2]
    zetas=[r['zeta'] for r in branches if r.get('accepted')]
    geometry=(len(zetas)==2 and np.sign(zetas[0]-z)==np.sign(zetas[1]-z) and abs(zetas[0]-z)>args.zeta_min_separation and abs(zetas[1]-z)>args.zeta_min_separation)
    cond=(abs(trans)>args.trans_tol and abs(quad)>args.quad_tol and abs(quad-quad2)/max(abs(quad),abs(quad2),1e-30)<args.curvature_rel_tol and s[-1]<args.null_singular_tol and s[-2]>args.second_singular_min and np.max(abs(F))<args.polish_tol and geometry)
    return dict(source_step=step,zeta=float(w[-1]),period=float(w[6*m]),polish_success=bool(pol.success),polish_residual=float(np.max(abs(F))),singular_values=s.tolist(),null_defect=float(np.max(abs(J@v))),left_null_defect=float(np.max(abs(left@J))),transversality=trans,quadratic_coefficient=quad,quadratic_half_step=quad2,curvature_relative_change=float(abs(quad-quad2)/max(abs(quad),abs(quad2),1e-30)),normal_form_zeta_curvature=float(-quad/trans) if abs(trans)>1e-15 else None,branches=branches,two_branch_geometry=bool(geometry),periodicity_error=float(np.max(abs(end-y0))),monodromy_matrix_max_diff=float(np.max(abs(product-Msingle))),spectral_matching_max_diff=diff,spectral_matching=matched,near_unit_product=[[float(q.real),float(q.imag)] for q in near_prod],near_unit_single=[[float(q.real),float(q.imag)] for q in near_single],genericity_screen_pass=bool(cond),classification='GENERIC_FOLD_NUMERICALLY_SUPPORTED' if cond else 'INCONCLUSIVE',limitations=['Tiny Floquet eigenvalues and near-unit eigenvalue splitting may be ill-conditioned','Genericity screen is numerical, not rigorous bifurcation proof'])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p19-report',type=Path,nargs='+',required=True,help='P19 report(s) from both candidates')
    ap.add_argument('--segments',type=int,default=4);ap.add_argument('--rtol',type=float,default=3e-11);ap.add_argument('--atol',type=float,default=3e-13)
    ap.add_argument('--curvature-h',type=float,default=.02);ap.add_argument('--branch-step',type=float,default=.10)
    ap.add_argument('--maxfev',type=int,default=120);ap.add_argument('--polish-tol',type=float,default=1e-8);ap.add_argument('--branch-tol',type=float,default=1e-8)
    ap.add_argument('--trans-tol',type=float,default=1e-7);ap.add_argument('--quad-tol',type=float,default=1e-7);ap.add_argument('--curvature-rel-tol',type=float,default=.2)
    ap.add_argument('--null-singular-tol',type=float,default=1e-5);ap.add_argument('--second-singular-min',type=float,default=1e-8);ap.add_argument('--zeta-min-separation',type=float,default=1e-10)
    ap.add_argument('--output-dir',type=Path,default=Path('results/author_fold_genericity'))
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    result={'scope':'Harb normalized representative benchmark; no physical K0 calibration','settings':{k:([str(x) for x in v] if isinstance(v,list) and k=='p19_report' else str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},'folds':[],'errors':[]}
    for path in args.p19_report:
        src,base=unpack_report(path)
        for fold in src['folds']:
            good=[s for s in fold['stages'] if s.get('validated')]
            if not good:result['errors'].append({'file':str(path),'step':fold['source_step'],'error':'No validated P19 stage'});continue
            step=fold['source_step'];print(f'P20 candidate {step}',flush=True)
            try:
                row=analyze(good[-1],base,args,step);result['folds'].append(row)
                print(f"  zeta={row['zeta']:.12f} trans={row['transversality']:.4e} quad={row['quadratic_coefficient']:.4e} geometry={row['two_branch_geometry']} status={row['classification']}",flush=True)
            except Exception as e:result['errors'].append({'file':str(path),'step':step,'error':str(e)});print(f'  ERROR: {e}',flush=True)
            tmp=args.output_dir/'p20_report.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,args.output_dir/'p20_report.json')
    with (args.output_dir/'p20_folds.csv').open('w',newline='') as f:
        cols=['source_step','zeta','period','polish_residual','transversality','quadratic_coefficient','curvature_relative_change','two_branch_geometry','periodicity_error','monodromy_matrix_max_diff','spectral_matching_max_diff','genericity_screen_pass','classification']
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows([{k:r.get(k) for k in cols} for r in result['folds']])
    print(f"P20 finished: {len(result['folds'])} folds, {len(result['errors'])} errors",flush=True)
if __name__=='__main__':main()

