#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p12_author_cycle_shooting.py

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
import argparse,csv,json,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p02_equilibrium_hopf_analysis import equilibrium_branches
from p11_author_hopf_verification import calculate,pair,BASE
from p04_periodic_orbit_floquet import full_jacobian,integrate_orbit

def basis(report,base):
    zh=report['zeta_hopf'];om=report['eigenvalue'][1]
    _,q,_,ye,_,_=pair(zh,base,report['branch'],1j*om)
    q*=np.exp(-.5j*np.angle(np.dot(q,q)))
    u=q.real.copy();v=-q.imag.copy()
    u/=np.linalg.norm(u)
    v-=u*np.dot(u,v);v/=np.linalg.norm(v)
    return ye,u,v

def shoot(amplitude,seed,report,base,rtol,atol,max_nfev):
    ye,u,v=seed; zh=report['zeta_hopf'];om=report['eigenvalue'][1];branch=report['branch']
    # l1 is very small; the zeta shift can be tiny even for visible amplitude.
    guess=np.r_[ye+amplitude*u,2*np.pi/om,zh-min(.02,.001*amplitude**2)]
    n_calls=[0]
    def residual(w):
        n_calls[0]+=1
        y0=w[:6];T=float(w[6]);z=float(w[7])
        if T<=0 or not 0<z<1:return np.full(8,1e5)
        p=replace(base,zeta=z)
        try:
            yT,_,_=integrate_orbit(y0,T,p,rtol,atol,False)
            yez=dict(equilibrium_branches(p))[branch]
        except Exception:return np.full(8,1e5)
        return np.r_[yT-y0,np.dot(y0-yez,v),np.dot(y0-yez,u)-amplitude]
    lower=np.r_[np.full(6,-np.inf),.75*(2*np.pi/om),max(.5,zh-.12)]
    upper=np.r_[np.full(6,np.inf),1.25*(2*np.pi/om),min(.9999,zh+.01)]
    fit=least_squares(residual,guess,bounds=(lower,upper),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=max_nfev,x_scale='jac')
    y0=fit.x[:6];T=float(fit.x[6]);z=float(fit.x[7]);err=float(np.linalg.norm(residual(fit.x),ord=np.inf))
    item=dict(amplitude_constraint=amplitude,zeta=z,delta_zeta_from_hopf=z-zh,period=T,shooting_residual=err,
              solver_success=bool(fit.success),solver_evaluations=int(fit.nfev),integration_calls=n_calls[0],initial_state=y0.tolist(),status='UNRESOLVED')
    if err>1e-7 or not fit.success or abs(z-zh)<1e-10:
        item['reason']='shooting failure, equilibrium collapse, or zeta indistinguishable from onset';return item
    p=replace(base,zeta=z)
    yT,M,_=integrate_orbit(y0,T,p,rtol*.1,atol*.1,True)
    mu=np.linalg.eigvals(M);neutral=int(np.argmin(abs(mu-1)));non=np.delete(mu,neutral)
    neutral_error=float(abs(mu[neutral]-1));unstable=int(np.sum(abs(non)>1+1e-3))
    item.update(periodicity_error=float(max(abs(yT-y0))),neutral_multiplier_error=neutral_error,
                multipliers=[[float(m.real),float(m.imag)] for m in mu],unstable_multiplier_count=unstable,
                max_nontrivial_modulus=float(max(abs(non))),
                status=('UNSTABLE_PERIODIC_ORBIT' if unstable else 'ATTRACTING_CANDIDATE') if neutral_error<1e-3 and max(abs(yT-y0))<1e-7 else 'UNRESOLVED')
    return item

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--omega',type=float,default=1);ap.add_argument('--W',type=float,default=100)
    ap.add_argument('--branch',default='00');ap.add_argument('--bracket',nargs=2,type=float,default=[.915,.920]);ap.add_argument('--guess',type=float,default=.91750958);ap.add_argument('--frequency',type=float,default=.22826782)
    ap.add_argument('--amplitudes',nargs='+',type=float,default=[.05,.1,.2,.35]);ap.add_argument('--rtol',type=float,default=2e-9);ap.add_argument('--atol',type=float,default=2e-11);ap.add_argument('--max-nfev',type=int,default=100)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'author_cycles');args=ap.parse_args()
    if args.W<=0 or any(a<=0 for a in args.amplitudes):ap.error('W and amplitudes must be positive')
    report,base=calculate(args)
    if report['status']!='NONDEGENERATE_HOPF_VERIFIED':raise RuntimeError('P11 verification failed')
    seed=basis(report,base);rows=[]
    for amp in args.amplitudes:
        print(f'P12 amplitude={amp:.5g}',flush=True)
        try:row=shoot(amp,seed,report,base,args.rtol,args.atol,args.max_nfev)
        except Exception as exc:row=dict(amplitude_constraint=amp,status='UNRESOLVED',reason=str(exc))
        rows.append(row);print('  ',row['status'],'zeta=',row.get('zeta'),'residual=',row.get('shooting_residual'),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=dict(scope='Harb representative normalized benchmark, not published K0 reproduction',assumptions={'omega':args.omega,'W':args.W},hopf={'zeta':report['zeta_hopf'],'omega':report['eigenvalue'][1],'l1':report['first_lyapunov_coefficient'],'other_unstable_eigenvalues':report['other_unstable_eigenvalues']},orbits=rows,
        limitations=['Amplitude-parameterized local shooting is not fixed-zeta continuation','No physical K0 calibration','Finite-amplitude solutions need branch-connectivity checks','Unresolved rows are not evidence that cycles do not exist'])
    (args.output_dir/'p12_report.json').write_text(json.dumps(output,indent=2)+'\n')
    cols=['amplitude_constraint','zeta','delta_zeta_from_hopf','period','shooting_residual','periodicity_error','neutral_multiplier_error','unstable_multiplier_count','max_nontrivial_modulus','status','reason']
    with (args.output_dir/'p12_orbits.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        good=[r for r in rows if r['status']!='UNRESOLVED']
        if good:
            fig,ax=plt.subplots(figsize=(8,5));ax.plot([r['zeta'] for r in good],[r['amplitude_constraint'] for r in good],'o-');ax.axvline(report['zeta_hopf'],ls='--',color='black',label='Hopf');ax.set(xlabel='Coupling zeta',ylabel='Shooting amplitude constraint',title='P12: local periodic solutions (not K0 calibration)');ax.legend();fig.tight_layout();fig.savefig(args.output_dir/'p12_branch.png',dpi=300);plt.close(fig)
    except Exception as exc:print('Plot unavailable:',exc)
    print('Output:',args.output_dir)
    if not any(r['status']!='UNRESOLVED' for r in rows):sys.exit(2)
if __name__=='__main__':main()

