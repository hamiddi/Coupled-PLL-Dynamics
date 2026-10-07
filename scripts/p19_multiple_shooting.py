#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p19_multiple_shooting.py

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
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p04_periodic_orbit_floquet import full_jacobian,integrate_orbit

def fz(y,p):
    z=p.zeta; th=z*y[0]-y[3];v=z*y[1]-y[4];a=z*y[2]-y[5]
    ans=np.zeros(6)
    ans[5]=p.b*(-np.sin(th)*y[0]*a+np.cos(th)*y[2])-p.b*(np.cos(th)*y[0]*v*v+2*np.sin(th)*v*y[1])+p.d*(-np.sin(th)*y[0]*v+np.cos(th)*y[1])+np.cos(th)*y[0]
    return ans

def segment(y,h,p,rtol,atol):
    # Integrate state, 6x6 variational matrix and analytic zeta sensitivity.
    x=np.r_[y,np.eye(6).ravel(),np.zeros(6)]
    def rhs(t,x):
        q=x[:6];J=full_jacobian(q,p)
        return np.r_[rotating_rhs(t,q,p),(J@x[6:42].reshape(6,6)).ravel(),J@x[42:48]+fz(q,p)]
    sol=solve_ivp(rhs,(0,h),x,method='DOP853',rtol=rtol,atol=atol,max_step=min(.25,h/12))
    if not sol.success:raise RuntimeError(sol.message)
    x=sol.y[:,-1];return x[:6],x[6:42].reshape(6,6),x[42:48]

def initial_segments(y,T,p,m,rtol,atol):
    sol=solve_ivp(lambda t,q:rotating_rhs(t,q,p),(0,T),y,method='DOP853',rtol=rtol,atol=atol,dense_output=True,max_step=.25)
    if not sol.success:raise RuntimeError(sol.message)
    return np.array([sol.sol(k*T/m) for k in range(m)])

def system(w,m,base,anchor,phase,rtol,atol,with_zeta=True):
    Y=w[:6*m].reshape(m,6);T=float(w[6*m]);z=float(w[6*m+1]);p=replace(base,zeta=z);h=T/m
    E=np.empty((m,6));Ms=[];Z=[]
    for k in range(m):
        e,M,s=segment(Y[k],h,p,rtol,atol);E[k]=e;Ms.append(M);Z.append(s)
    F=np.r_[(E-np.roll(Y,-1,axis=0)).ravel(),np.dot(Y[0]-anchor,phase)]
    J=np.zeros((6*m+1,6*m+1));Jz=np.zeros(6*m+1)
    for k in range(m):
        i=6*k;j=6*((k+1)%m)
        J[i:i+6,i:i+6]+=Ms[k];J[i:i+6,j:j+6]-=np.eye(6)
        J[i:i+6,6*m]=rotating_rhs(0,E[k],p)/m
        Jz[i:i+6]=Z[k]
    J[6*m,:6]=phase
    return F,J,Jz,Ms

def seed(rec,base,m,rtol,atol):
    row=rec['probes'][1]['row'];y=np.asarray(row['initial_state'],float);T=float(row['period']);z=float(row['zeta']);p=replace(base,zeta=z)
    Y=initial_segments(y,T,p,m,rtol,atol)
    anchor=y.copy();phase=rotating_rhs(0,y,p);phase/=np.linalg.norm(phase)
    w=np.r_[Y.ravel(),T,z];F,J,_,_=system(w,m,base,anchor,phase,rtol,atol)
    # Equilibrate fixed-zeta Jacobian only for a better initial null direction.
    cs=np.maximum(np.linalg.norm(J,axis=0),1e-9);U,s,Vh=np.linalg.svd(J/cs[None,:]);v=Vh[-1]/cs;v/=np.linalg.norm(v)
    return np.r_[w,v],anchor,phase

def run(rec,base,args):
    m=args.segments;step=int(rec['source_center_step']);out=[];prior=None
    for stage,(rtol,atol) in enumerate(zip(args.rtols,args.atols)):
        if prior is None: x0,anchor,phase=seed(rec,base,m,rtol,atol)
        else:x0,anchor,phase=prior
        n=6*m+2; nv=6*m+1
        # Scale the continuity equations to prevent large phase coordinates dominating.
        def residual(x):
            w=x[:n];v=x[n:];T=w[6*m];z=w[6*m+1]
            if not args.period_min<T<args.period_max or not args.zeta_min<z<args.zeta_max:
                return np.full(2*nv+1,1e3)
            F,J,_,_=system(w,m,base,anchor,phase,rtol,atol)
            return np.r_[F[:6*m],F[-1],args.null_weight*(J@v),args.norm_weight*(v@v-1)]
        lo=np.r_[np.full(6*m,-np.inf),args.period_min,args.zeta_min,np.full(nv,-np.inf)]
        hi=np.r_[np.full(6*m,np.inf),args.period_max,args.zeta_max,np.full(nv,np.inf)]
        try:
            sol=least_squares(residual,x0,bounds=(lo,hi),method='trf',jac='2-point',x_scale='jac',max_nfev=args.max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            x=sol.x;w=x[:n];v=x[n:];T=float(w[6*m]);z=float(w[6*m+1]);p=replace(base,zeta=z)
            F,J,_,Ms=system(w,m,base,anchor,phase,rtol*.3,atol*.3)
            shooting=float(np.max(abs(F)));null=float(np.max(abs(J@v)));norm=float(abs(v@v-1))
            # Independently validate the complete periodic orbit with single shooting.
            y=w[:6];end,M,_=integrate_orbit(y,T,p,rtol*.1,atol*.1,True)
            single=float(np.max(abs(end-y)));mu=np.linalg.eigvals(M)
            neutral=int(np.argmin(abs(mu-1)));non=np.delete(mu,neutral)
            near=non[np.argmin(abs(non-1))];near_dist=float(abs(near-1))
            # An additional multiplier must be nearly REAL and close to +1.
            valid=bool(shooting<args.defect_tol and single<args.defect_tol and null<args.null_tol and norm<1e-6 and abs(mu[neutral]-1)<args.neutral_tol and near_dist<args.multiplier_tol and abs(near.imag)<args.imag_tol)
            row=dict(stage=stage,rtol=rtol,atol=atol,solver_success=bool(sol.success),nfev=int(sol.nfev),message=sol.message,zeta=z,period=T,initial_state=y.tolist(),segment_states=w[:6*m].reshape(m,6).tolist(),shooting_defect=shooting,independent_periodicity_error=single,null_defect=null,null_norm_error=norm,neutral_error=float(abs(mu[neutral]-1)),near_plus_one=[float(near.real),float(near.imag)],near_plus_one_distance=near_dist,unstable_count=int(np.sum(abs(non)>1+1e-3)),multipliers=[[float(a.real),float(a.imag)] for a in mu],validated=valid)
            if valid:prior=(x,anchor,phase)
            else:prior=None
            print(f'  stage {stage}: zeta={z:.12f} T={T:.8f} shooting={shooting:.2e} single={single:.2e} null={null:.2e} near+1={near} validated={valid}',flush=True)
        except Exception as e:
            row=dict(stage=stage,validated=False,error=str(e));print(f'  stage {stage}: ERROR {e}',flush=True);prior=None
        out.append(row)
        if args.stop_on_failure and not row['validated']:break
    good=[r for r in out if r['validated']]
    converged=len(good)==len(args.rtols) and len(good)>=2 and abs(good[-1]['zeta']-good[-2]['zeta'])<args.zeta_convergence_tol
    return dict(source_step=step,stages=out,classification='NUMERICAL_FOLD_SUPPORTED' if converged else 'UNRESOLVED',zeta_convergence=abs(good[-1]['zeta']-good[-2]['zeta']) if len(good)>=2 else None)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p16-report',type=Path,required=True);ap.add_argument('--fold-steps',type=int,nargs='+',default=[86,50]);ap.add_argument('--segments',type=int,default=4)
    ap.add_argument('--rtols',type=float,nargs='+',default=[3e-10,3e-11]);ap.add_argument('--atols',type=float,nargs='+',default=[3e-12,3e-13]);ap.add_argument('--max-nfev',type=int,default=40)
    ap.add_argument('--null-weight',type=float,default=1.);ap.add_argument('--norm-weight',type=float,default=1.);ap.add_argument('--defect-tol',type=float,default=3e-8);ap.add_argument('--null-tol',type=float,default=3e-7)
    ap.add_argument('--neutral-tol',type=float,default=3e-4);ap.add_argument('--multiplier-tol',type=float,default=.02);ap.add_argument('--imag-tol',type=float,default=2e-4);ap.add_argument('--zeta-convergence-tol',type=float,default=2e-6)
    ap.add_argument('--period-min',type=float,default=20);ap.add_argument('--period-max',type=float,default=45);ap.add_argument('--zeta-min',type=float,default=.88);ap.add_argument('--zeta-max',type=float,default=.92)
    ap.add_argument('--stop-on-failure',action='store_true');ap.add_argument('--output-dir',type=Path,default=Path('results/author_multiple_shooting'))
    args=ap.parse_args()
    if args.segments<2:ap.error('segments must be >=2')
    if len(args.rtols)!=len(args.atols):ap.error('rtols and atols must match')
    src=json.loads(args.p16_report.read_text());base=Parameters(**src['parameters'],omega=src['assumptions']['omega'],W=src['assumptions']['W'],zeta=.9)
    result=dict(scope=src['scope'],source_p16=str(args.p16_report),parameters=src['parameters'],assumptions=src['assumptions'],settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},folds=[],errors=[],limitations=['Numerical fold evidence, not a genericity proof','No physical K0 mapping','Tiny Floquet modes are not individually resolved'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for step in args.fold_steps:
        rec=next((r for r in src['refinements'] if int(r['source_center_step'])==step),None)
        if rec is None:result['errors'].append(dict(step=step,error='P16 seed missing'));continue
        print(f'P19 multiple shooting candidate {step}',flush=True)
        try:result['folds'].append(run(rec,base,args))
        except Exception as e:result['errors'].append(dict(step=step,error=str(e)))
        tmp=args.output_dir/'p19_report.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,args.output_dir/'p19_report.json')
    with (args.output_dir/'p19_folds.csv').open('w',newline='') as fh:
        keys=['source_step','stage','zeta','period','shooting_defect','independent_periodicity_error','null_defect','neutral_error','near_plus_one_distance','unstable_count','validated','classification','error']
        wr=csv.DictWriter(fh,fieldnames=keys);wr.writeheader()
        for fold in result['folds']:
            for st in fold['stages']:wr.writerow({k:fold['source_step'] if k=='source_step' else fold['classification'] if k=='classification' else st.get(k) for k in keys})
    print(f'P19 finished: {len(result["folds"])} candidates, {len(result["errors"])} errors',flush=True)
if __name__=='__main__':main()

