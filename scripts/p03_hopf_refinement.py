#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p03_hopf_refinement.py

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
import argparse, csv, json, sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from p00_model_validation import ILLUSTRATIVE
from p01_rotating_frame_validation import rotating_rhs
from p02_equilibrium_hopf_analysis import equilibrium_branches,equilibrium_jacobian,finite_difference_jacobian

DEFAULT_INTERVALS={'01':(.508,.5165),'10':(.5845,.593)}

def branch_state(z,branch,p0):
    p=replace(p0,zeta=float(z))
    matches=dict(equilibrium_branches(p))
    if branch not in matches: raise ValueError(f'Branch {branch} absent at zeta={z}')
    return matches[branch],p

def tracked_pair(z,branch,p0,anchor=None):
    y,p=branch_state(z,branch,p0)
    A=equilibrium_jacobian(y,p)
    values,vectors=np.linalg.eig(A)
    inds=np.flatnonzero(values.imag>1e-7)
    if len(inds)==0: raise ValueError('No positive-imaginary eigenvalue')
    if anchor is None:
        k=inds[np.argmax(values[inds].real)]
    else:
        k=inds[np.argmin(np.abs(values[inds]-anchor))]
    return values[k],vectors[:,k],A,y,p,values

def multilinear(y,p):
    """Exact symmetric Hessian/trilinear forms at an equilibrium.

    Derivatives of sin(theta)+b*cos(theta)*acc+d*cos(theta)*vel
    -b*sin(theta)*vel**2. theta, vel and acc are linear in y.
    """
    z=p.zeta
    configs=[(2,np.array([-1,0,0,1,0,0.]),np.array([0,-1,0,0,1,0.]),
              np.array([0,0,-1,0,0,1.])),
             (5,np.array([z,0,0,-1,0,0.]),np.array([0,z,0,0,-1,0.]),
              np.array([0,0,z,0,0,-1.]))]
    def B(u,v):
        out=np.zeros(6,dtype=np.result_type(u,v,complex))
        for row,T,V,H in configs:
            s=np.sin(T@y)
            tu,tv=T@u,T@v; vu,vv=V@u,V@v; hu,hv=H@u,H@v
            out[row]=-s*tu*tv-p.b*s*(tu*hv+tv*hu)-p.d*s*(tu*vv+tv*vu)-2*p.b*s*vu*vv
        return out
    def C(u,v,w):
        out=np.zeros(6,dtype=np.result_type(u,v,w,complex))
        for row,T,V,H in configs:
            co=np.cos(T@y)
            tu,tv,tw=T@u,T@v,T@w
            vu,vv,vw=V@u,V@v,V@w
            hu,hv,hw=H@u,H@v,H@w
            out[row]=(-co*tu*tv*tw
                -p.b*co*(tu*tv*hw+tu*tw*hv+tv*tw*hu)
                -p.d*co*(tu*tv*vw+tu*tw*vv+tv*tw*vu)
                -2*p.b*co*(tu*vv*vw+tv*vu*vw+tw*vu*vv))
        return out
    return B,C

def first_lyapunov(A,q,omega,B,C):
    """Kuznetsov l1 with <p,q>=1 and ||q||_2=1."""
    ev,left=np.linalg.eig(A.conj().T)
    k=np.argmin(np.abs(ev+1j*omega))
    p=left[:,k]
    # np.vdot(p,q)=1: vdot(p/ conj(vdot(p,q)),q)=1
    p=p/np.conj(np.vdot(p,q))
    term=(C(q,q,np.conj(q))
          -2*B(q,np.linalg.solve(A,B(q,np.conj(q))))
          +B(np.conj(q),np.linalg.solve(2j*omega*np.eye(6)-A,B(q,q))))
    return float(np.real(np.vdot(p,term))/(2*omega)),float(abs(np.vdot(p,q)-1))

def refine(branch,lo,hi,p0,xtol=1e-12):
    mid=.5*(lo+hi)
    anchor=tracked_pair(mid,branch,p0)[0]
    def realpart(z): return tracked_pair(z,branch,p0,anchor)[0].real
    f_lo,f_hi=realpart(lo),realpart(hi)
    if f_lo*f_hi>=0: raise ValueError(f'No tracked sign change: {f_lo}, {f_hi}')
    zstar=brentq(realpart,lo,hi,xtol=xtol,rtol=1e-13)
    eigen,q,A,y,p,all_eig=tracked_pair(zstar,branch,p0,anchor)
    h=min(1e-4,(hi-lo)/10)
    slope=(realpart(zstar+h)-realpart(zstar-h))/(2*h)
    slope2=(realpart(zstar+h/2)-realpart(zstar-h/2))/h
    # Local isolation: exactly one simple conjugate pair on imaginary axis.
    remaining=np.array(sorted(all_eig,key=lambda x:abs(x-eigen)))[1:]
    spectral_gap=float(min(abs(v.real) for v in remaining if abs(v-np.conj(eigen))>1e-5))
    B,C=multilinear(y,p)
    l1,norm_err=first_lyapunov(A,q,float(eigen.imag),B,C)
    residual=float(np.linalg.norm(rotating_rhs(0,y,p),ord=np.inf))
    jerror=float(np.max(abs(A-finite_difference_jacobian(y,p))))
    tests={'equilibrium_residual':residual,'jacobian_error':jerror,'crossing_residual':abs(eigen.real),
           'transversality':bool(abs(slope)>1e-6),'slope_relative_change':abs(slope-slope2)/max(abs(slope),1e-15),
           'other_eigenvalue_real_gap':spectral_gap,'nonzero_l1':bool(abs(l1)>1e-7),
           'adjoint_normalization_error':norm_err}
    verified=(residual<1e-8 and jerror<1e-6 and abs(eigen.real)<1e-8
              and tests['transversality'] and tests['slope_relative_change']<.01
              and spectral_gap>1e-5 and tests['nonzero_l1'] and norm_err<1e-8)
    return {'branch':branch,'zeta':float(zstar),'initial_interval':[lo,hi],
            'eigenvalue':[float(eigen.real),float(eigen.imag)],
            'transversality_dreal_dzeta':float(slope),'first_lyapunov_coefficient':l1,
            'l1_convention':'Kuznetsov: Re(<p,C(q,q,qbar)-2B(q,A^-1 B(q,qbar))+B(qbar,(2iωI-A)^-1 B(q,q))>)/(2ω), ||q||=1',
            'other_eigenvalues':[[float(v.real),float(v.imag)] for v in all_eig if abs(v-eigen)>1e-6 and abs(v-np.conj(eigen))>1e-6],
            'checks':tests,'status':'NONDEGENERATE_HOPF_VERIFIED' if verified else 'UNRESOLVED'}

def derivative_test(p0):
    """Independent directional Taylor check of B and C using RHS finite differences."""
    rng=np.random.default_rng(90210)
    y,p=branch_state(.55,'01',p0)
    B,C=multilinear(y,p)
    u=rng.normal(size=6);u/=np.linalg.norm(u)
    v=rng.normal(size=6);v/=np.linalg.norm(v)
    h=1e-3
    f=lambda x:rotating_rhs(0,x,p)
    # Hessian directional central mixed difference
    bfd=(f(y+h*u+h*v)-f(y+h*u-h*v)-f(y-h*u+h*v)+f(y-h*u-h*v))/(4*h*h)
    # Third derivative diagonal via five-point central stencil
    cfd=(f(y+2*h*u)-2*f(y+h*u)+2*f(y-h*u)-f(y-2*h*u))/(2*h**3)
    # stencil above yields +f''' (check x^3: (8-2+2-(-8))*h³/(2h³)=8? actually 8-2+(-2)+8=12 ->6 yes)
    return {'B_directional_error':float(max(abs(B(u,v)-bfd))),
            'C_directional_error':float(max(abs(C(u,u,u)-cfd)))}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'hopf_refinement')
    args=ap.parse_args()
    checks=derivative_test(ILLUSTRATIVE)
    if args.self_test:
        print(json.dumps(checks,indent=2));return
    results=[]
    for branch,(lo,hi) in DEFAULT_INTERVALS.items():
        try: results.append(refine(branch,lo,hi,ILLUSTRATIVE))
        except Exception as exc: results.append({'branch':branch,'status':'UNRESOLVED','error':str(exc)})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'scope':'Illustrative normalized model; not published K0 reproduction',
            'parameters':asdict(ILLUSTRATIVE),'derivative_checks':checks,'results':results,
            'limitations':['Local nondegenerate Hopf does not imply an attracting cycle.',
              'l1 magnitude depends on eigenvector and time normalization.',
              'zeta=1 requires a symmetry quotient and is excluded.',
              'Other winding branches and global attractors are not enumerated.']}
    (args.output_dir/'p03_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output_dir/'p03_hopf_points.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['branch','zeta','omega','transversality','l1','status'])
        w.writeheader()
        for r in results:
            eig=r.get('eigenvalue',[None,None]);w.writerow({'branch':r['branch'],'zeta':r.get('zeta'),
              'omega':eig[1],'transversality':r.get('transversality_dreal_dzeta'),
              'l1':r.get('first_lyapunov_coefficient'),'status':r['status']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for r in results:
        if 'zeta' not in r: continue
        z=r['zeta']; lo,hi=r['initial_interval']; anchor=complex(*r['eigenvalue'])
        xx=np.linspace(lo,hi,101)
        yy=[tracked_pair(t,r['branch'],ILLUSTRATIVE,anchor)[0].real for t in xx]
        ax.plot(xx,yy,label=f"Branch {r['branch']}")
        ax.scatter([z],[0],s=40)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(xlabel='Coupling factor ζ',ylabel='Tracked eigenvalue real part',
           title='Illustrative Hopf candidate refinement (not reference reproduction)')
    ax.legend();fig.tight_layout()
    fig.savefig(args.output_dir/'p03_crossing_refinement.png',dpi=600)
    fig.savefig(args.output_dir/'p03_crossing_refinement.pdf')
    plt.close(fig)
    print(json.dumps(report,indent=2))
    print('Outputs:',args.output_dir)
    if any(r['status']=='UNRESOLVED' for r in results): sys.exit(2)

if __name__=='__main__':main()

