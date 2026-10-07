#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p04_periodic_orbit_floquet.py

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
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from p00_model_validation import ILLUSTRATIVE
from p01_rotating_frame_validation import rotating_rhs
from p02_equilibrium_hopf_analysis import equilibrium_branches, equilibrium_jacobian
from p03_hopf_refinement import refine, DEFAULT_INTERVALS, tracked_pair


def full_jacobian(y,p):
    """Analytic Jacobian at arbitrary rotating-frame state."""
    z=p.zeta
    th1=y[3]-y[0]; th2=z*y[0]-y[3]
    v1=y[4]-y[1]; a1=y[5]-y[2]
    v2=z*y[1]-y[4]; a2=z*y[2]-y[5]
    J=np.zeros((6,6))
    J[0,1]=J[1,2]=J[3,4]=J[4,5]=1
    for row,th,v,acc,dt,dv,da in (
        (2,th1,v1,a1,np.array([-1,0,0,1,0,0]),
         np.array([0,-1,0,0,1,0]),np.array([0,0,-1,0,0,1])),
        (5,th2,v2,a2,np.array([z,0,0,-1,0,0]),
         np.array([0,z,0,0,-1,0]),np.array([0,0,z,0,0,-1]))):
        ct,st=np.cos(th),np.sin(th)
        J[row] += (-p.b*st*acc-p.b*ct*v*v-p.d*st*v+ct)*dt
        J[row] += (-2*p.b*st*v+p.d*ct)*dv+p.b*ct*da
    J[2,1]-=p.c; J[2,2]-=p.a
    J[5,4]-=p.c; J[5,5]-=p.a
    return J


def integrate_orbit(y0,T,p,rtol,atol,variational=True):
    """Integrate state and, optionally, 6x6 state-transition matrix."""
    if variational:
        initial=np.r_[y0,np.eye(6).ravel()]
        def fun(t,v):
            y=v[:6]; Phi=v[6:].reshape(6,6)
            return np.r_[rotating_rhs(t,y,p),(full_jacobian(y,p)@Phi).ravel()]
    else:
        initial=y0
        fun=lambda t,y:rotating_rhs(t,y,p)
    sol=solve_ivp(fun,(0,float(T)),initial,method='DOP853',rtol=rtol,atol=atol,
                  dense_output=False,max_step=min(.25,float(T)/40))
    if not sol.success: raise RuntimeError(sol.message)
    return sol.y[:6,-1],sol.y[6:,-1].reshape(6,6) if variational else None,sol


def hopf_basis(branch,p0):
    r=refine(branch,*DEFAULT_INTERVALS[branch],p0)
    if r['status']!='NONDEGENERATE_HOPF_VERIFIED':
        raise RuntimeError(f'P03 did not verify branch {branch}')
    zh=r['zeta']; omega=r['eigenvalue'][1]
    lam,q,A,ye,p,vals=tracked_pair(zh,branch,p0,complex(0,omega))
    # Rotate eigenvector so its real part has maximum possible norm.
    q=q*np.exp(-.5j*np.angle(np.dot(q,q)))
    u=np.real(q); v=-np.imag(q)
    u/=np.linalg.norm(u)
    v-=u*np.dot(u,v); v/=np.linalg.norm(v)
    return r,ye,u,v


def shoot(branch,amplitude,seed,p0,rtol,atol,max_nfev):
    """Solve 6 periodicity + phase + amplitude equations for (y0[6],T,zeta)."""
    r,ye,u,v=seed
    zh=r['zeta']; omega=r['eigenvalue'][1]
    # l1<0 and dRe(lambda)/dzeta<0: local cycles on zeta<zh side.
    guess=np.r_[ye+amplitude*u,2*np.pi/omega,zh-.5*amplitude**2]
    def residual(w):
        y0=w[:6];T=w[6];z=w[7]
        if T<=0 or not 0<z<.999:
            return np.full(8,1e3+abs(T)+abs(z))
        p=replace(p0,zeta=float(z))
        try: yT,_,_=integrate_orbit(y0,T,p,rtol,atol,False)
        except Exception: return np.full(8,1e3)
        ye_z=dict(equilibrium_branches(p))[branch]
        # Phase gauge and nonzero amplitude suppress the trivial equilibrium.
        return np.r_[yT-y0, np.dot(y0-ye_z,v), np.dot(y0-ye_z,u)-amplitude]
    bounds_lo=np.r_[np.full(6,-np.inf),.5*(2*np.pi/omega),max(.001,zh-.12)]
    bounds_hi=np.r_[np.full(6,np.inf),1.5*(2*np.pi/omega),min(.999,zh+.03)]
    result=least_squares(residual,guess,bounds=(bounds_lo,bounds_hi),
                         xtol=2e-11,ftol=2e-11,gtol=2e-11,max_nfev=max_nfev)
    y0=result.x[:6];T=float(result.x[6]);z=float(result.x[7])
    defect=float(np.linalg.norm(residual(result.x),ord=np.inf))
    item={'branch':branch,'amplitude_constraint':amplitude,'zeta':z,'period':T,
          'shooting_residual':defect,'solver_success':bool(result.success),
          'solver_evaluations':int(result.nfev),'initial_state':y0.tolist(),
          'status':'UNRESOLVED'}
    if not result.success or defect>2e-7 or abs(z-zh)<1e-9:
        item['reason']='Shooting failed, residual too large, or equilibrium collapse'
        return item
    p=replace(p0,zeta=z)
    yT,M,_=integrate_orbit(y0,T,p,rtol*.1,atol*.1,True)
    mu=np.linalg.eigvals(M)
    trivial=int(np.argmin(abs(mu-1)))
    nontrivial=np.delete(mu,trivial)
    trivial_error=float(abs(mu[trivial]-1))
    eigmax=float(np.max(abs(nontrivial)))
    # An unstable multiplier may be very large; do not mistake a stable
    # center-manifold cycle for a full-system attractor.
    item.update({'periodicity_error':float(np.linalg.norm(yT-y0,ord=np.inf)),
                 'trivial_multiplier_error':trivial_error,
                 'multipliers':[[float(x.real),float(x.imag)] for x in mu],
                 'largest_nontrivial_multiplier_modulus':eigmax,
                 'unstable_multiplier_count':int(np.sum(abs(nontrivial)>1+1e-3)),
                 'status':('UNSTABLE_PERIODIC_ORBIT' if eigmax>1+1e-3 else
                           'STABLE_PERIODIC_ORBIT' if eigmax<1-1e-3 else 'MARGINAL_OR_UNRESOLVED')
                 if trivial_error<2e-3 and defect<2e-7 else 'UNRESOLVED'})
    if item['status']=='UNRESOLVED':item['reason']='Floquet neutral mode or shooting validation failed'
    return item


def self_test(p0):
    rng=np.random.default_rng(20260929)
    errors=[]
    for _ in range(12):
        p=replace(p0,zeta=float(rng.uniform(.25,.85)))
        y=rng.normal(size=6)*.4
        h=1e-6;J=np.column_stack([(rotating_rhs(0,y+h*np.eye(6)[j],p)-
                                   rotating_rhs(0,y-h*np.eye(6)[j],p))/(2*h)
                                  for j in range(6)])
        errors.append(float(np.max(abs(full_jacobian(y,p)-J))))
    p=replace(p0,zeta=.55)
    y0=np.array([.2,.1,-.05,-.3,.04,.06]);T=.75
    _,M,_=integrate_orbit(y0,T,p,2e-10,1e-12,True)
    h=1e-5
    fd=np.column_stack([(integrate_orbit(y0+h*np.eye(6)[j],T,p,2e-10,1e-12,False)[0]-
                          integrate_orbit(y0-h*np.eye(6)[j],T,p,2e-10,1e-12,False)[0])/(2*h)
                         for j in range(6)])
    result={'off_equilibrium_jacobian_error':max(errors),
            'variational_monodromy_error':float(np.max(abs(M-fd)))}
    result['status']='PASS' if result['off_equilibrium_jacobian_error']<1e-6 and result['variational_monodromy_error']<1e-6 else 'FAIL'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--amplitudes',nargs='+',type=float,default=[.02,.035,.05,.075])
    parser.add_argument('--rtol',type=float,default=2e-9)
    parser.add_argument('--atol',type=float,default=2e-11)
    parser.add_argument('--max-nfev',type=int,default=160)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results'/'periodic_orbits')
    args=parser.parse_args()
    tests=self_test(ILLUSTRATIVE)
    if args.self_test:
        print(json.dumps(tests,indent=2));sys.exit(0 if tests['status']=='PASS' else 2)
    if tests['status']!='PASS':raise RuntimeError('P04 self-tests failed')
    rows=[];hopf={}
    for branch in ('01','10'):
        seed=hopf_basis(branch,ILLUSTRATIVE)
        hopf[branch]={'zeta':seed[0]['zeta'],'omega':seed[0]['eigenvalue'][1]}
        for amplitude in args.amplitudes:
            try:
                if amplitude<=0:raise ValueError('Amplitude must be positive')
                row=shoot(branch,amplitude,seed,ILLUSTRATIVE,args.rtol,args.atol,args.max_nfev)
            except Exception as exc:
                row={'branch':branch,'amplitude_constraint':amplitude,'status':'UNRESOLVED','reason':str(exc)}
            rows.append(row)
            print(branch,amplitude,row['status'],row.get('shooting_residual'),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'scope':'Illustrative normalized model, NOT published K0 reproduction',
            'parameters':asdict(ILLUSTRATIVE),'self_test':tests,'hopf_points':hopf,
            'results':rows,'limitations':['Amplitude-constrained local shooting is not global pseudo-arclength continuation.',
              'Only selected small-amplitude orbits are attempted; failure is unresolved.',
              'Floquet stability is for the full six-state system; one neutral time-shift multiplier expected.',
              'No stable attractor or multistability is inferred from an unstable periodic orbit.']}
    (args.output_dir/'p04_report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output_dir/'p04_orbits.csv').open('w',newline='') as f:
        fields=['branch','amplitude_constraint','zeta','period','shooting_residual',
                'periodicity_error','trivial_multiplier_error',
                'largest_nontrivial_multiplier_modulus','unstable_multiplier_count','status','reason']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for branch in ('01','10'):
        valid=[r for r in rows if r['branch']==branch and r['status']!='UNRESOLVED']
        if valid:
            axes[0].plot([r['zeta'] for r in valid],[r['amplitude_constraint'] for r in valid],
                         'o-',label=f'Branch {branch}')
            axes[1].semilogy([r['zeta'] for r in valid],
                             [r['largest_nontrivial_multiplier_modulus'] for r in valid],
                             'o-',label=f'Branch {branch}')
        axes[0].axvline(hopf[branch]['zeta'],ls=':',lw=1,color='gray')
    axes[0].set(xlabel='Coupling ζ',ylabel='Shooting amplitude constraint',title='Near-Hopf periodic solutions')
    axes[1].axhline(1,color='black',ls='--',lw=1)
    axes[1].set(xlabel='Coupling ζ',ylabel='Largest nontrivial |Floquet multiplier|',title='Full-state orbital stability')
    for ax in axes:ax.grid(alpha=.2);ax.legend(loc='best') if ax.lines else None
    fig.suptitle('Illustrative model; not published K0 reproduction')
    fig.tight_layout();fig.savefig(args.output_dir/'p04_periodic_orbits.png',dpi=600)
    fig.savefig(args.output_dir/'p04_periodic_orbits.pdf');plt.close(fig)
    print(json.dumps({'self_test':tests,'status_counts':{s:sum(r['status']==s for r in rows)
          for s in sorted(set(r['status'] for r in rows))},'output':str(args.output_dir)},indent=2))

if __name__=='__main__':main()

