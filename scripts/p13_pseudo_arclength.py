#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p13_pseudo_arclength.py

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
from scipy.optimize import root
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p04_periodic_orbit_floquet import integrate_orbit
from p11_author_hopf_verification import calculate
from p12_author_cycle_shooting import basis,shoot


def packed(row,scale):
    return np.r_[np.asarray(row['initial_state'],float),float(row['period']),float(row['zeta'])*scale]


def monodromy(y,T,z,base,rtol,atol):
    return integrate_orbit(y,T,replace(base,zeta=float(z)),rtol,atol,True)


def floquet(y,T,z,base,rtol,atol):
    yT,M,_=monodromy(y,T,z,base,rtol,atol)
    mu=np.linalg.eigvals(M)
    neutral=int(np.argmin(abs(mu-1)))
    non=np.delete(mu,neutral)
    return dict(periodicity_error=float(max(abs(yT-y))),neutral_error=float(abs(mu[neutral]-1)),
                unstable_count=int(sum(abs(non)>1+1e-3)),max_nontrivial_modulus=float(max(abs(non))),
                multipliers=[[float(v.real),float(v.imag)] for v in mu],
                nontrivial_multipliers=[[float(v.real),float(v.imag)] for v in non])


def correct(predictor,tangent,reference,phase_vector,base,scale,rtol,atol,tol,maxfev):
    """Solve shooting+phase+arclength; analytic flow derivatives, FD z derivative."""
    cache={}
    def evaluate(w):
        if 'w' in cache and np.array_equal(w,cache['w']):return cache['r'],cache['J']
        y=w[:6];T=float(w[6]);z=float(w[7]/scale)
        if not (0<z<.99999 and 1<T<500):
            raise ValueError(f'out-of-domain corrector iterate: T={T}, zeta={z}')
        p=replace(base,zeta=z)
        yT,M,_=integrate_orbit(y,T,p,rtol,atol,True)
        dz=min(1e-5,.1*z,.1*(1-z))
        yp,_,_=integrate_orbit(y,T,replace(base,zeta=z+dz),rtol,atol,False)
        ym,_,_=integrate_orbit(y,T,replace(base,zeta=z-dz),rtol,atol,False)
        J=np.zeros((8,8));J[:6,:6]=M-np.eye(6)
        J[:6,6]=rotating_rhs(T,yT,p)
        J[:6,7]=(yp-ym)/(2*dz*scale)
        J[6,:6]=phase_vector
        J[7,:]=tangent
        r=np.r_[yT-y,np.dot(y-reference,phase_vector),np.dot(w-predictor,tangent)]
        cache.update(w=w.copy(),r=r,J=J)
        return r,J
    def fun(w):return evaluate(w)[0]
    def jac(w):return evaluate(w)[1]
    try:
        sol=root(fun,predictor,jac=jac,method='hybr',options={'xtol':tol,'maxfev':maxfev})
        residual=float(max(abs(fun(sol.x))))
        return sol.x,dict(success=bool(sol.success and residual<2e-7),residual=residual,nfev=int(sol.nfev),message=str(sol.message))
    except Exception as e:
        return None,dict(success=False,residual=None,nfev=0,message=str(e))


def entry(w,scale,base,rtol,atol,hopf,step,correction=None):
    y=w[:6];T=float(w[6]);z=float(w[7]/scale)
    f=floquet(y,T,z,base,rtol*.3,atol*.3)
    return dict(step=step,zeta=z,delta_from_hopf=z-hopf,period=T,initial_state=y.tolist(),
                correction=correction,**f,status=('UNSTABLE_PERIODIC_ORBIT' if f['unstable_count'] else 'ATTRACTING_CANDIDATE')
                if f['neutral_error']<2e-3 and f['periodicity_error']<2e-7 else 'UNRESOLVED')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--omega',type=float,default=1.);ap.add_argument('--W',type=float,default=100.)
    ap.add_argument('--branch',default='00');ap.add_argument('--bracket',type=float,nargs=2,default=[.915,.920])
    ap.add_argument('--guess',type=float,default=.91750958);ap.add_argument('--frequency',type=float,default=.22826782)
    ap.add_argument('--seed-amplitudes',type=float,nargs=2,default=[.2,.35])
    ap.add_argument('--p12-report',type=Path,default=None,help='Optional P12 report containing seed solutions')
    ap.add_argument('--zeta-scale',type=float,default=1000.)
    ap.add_argument('--ds',type=float,default=.12);ap.add_argument('--ds-min',type=float,default=.002)
    ap.add_argument('--ds-max',type=float,default=.35);ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--zeta-stop',type=float,default=.90,help='Stop once zeta <= threshold (no extrapolation)')
    ap.add_argument('--rtol',type=float,default=3e-9);ap.add_argument('--atol',type=float,default=3e-11)
    ap.add_argument('--maxfev',type=int,default=35);ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'author_continuation')
    args=ap.parse_args()
    if args.W<=0 or args.zeta_scale<=0 or args.ds_min<=0 or args.steps<1:ap.error('Invalid numeric options')
    report,base=calculate(args)
    if report['status']!='NONDEGENERATE_HOPF_VERIFIED':raise RuntimeError('P11 Hopf verification failed')
    ye,u,v=basis(report,base)
    seeds=[]
    if args.p12_report:
        data=json.loads(args.p12_report.read_text())
        if abs(data['hopf']['zeta']-report['zeta_hopf'])>1e-7:raise ValueError('P12 seed Hopf mismatch')
        if abs(data['assumptions']['omega']-args.omega)>1e-12 or abs(data['assumptions']['W']-args.W)>1e-12:raise ValueError('P12 assumptions mismatch')
        good=[x for x in data['orbits'] if x['status']!='UNRESOLVED' and 'initial_state' in x]
        for amp in args.seed_amplitudes:
            if not good:raise ValueError('No usable P12 seeds')
            x=min(good,key=lambda r:abs(r['amplitude_constraint']-amp))
            if abs(x['amplitude_constraint']-amp)>.025:raise ValueError('P12 seed amplitude mismatch')
            seeds.append(x)
    else:
        for amp in args.seed_amplitudes:
            print(f'Generating P12 seed amplitude {amp}',flush=True)
            x=shoot(amp,(ye,u,v),report,base,args.rtol,args.atol,100)
            if x['status']=='UNRESOLVED':raise RuntimeError('P12 seed failed: '+str(x))
            seeds.append(x)
    x0,x1=[packed(r,args.zeta_scale) for r in seeds]
    if np.linalg.norm(x1-x0)<1e-8:raise ValueError('Seeds too close')
    rows=[entry(x0,args.zeta_scale,base,args.rtol,args.atol,report['zeta_hopf'],0),
          entry(x1,args.zeta_scale,base,args.rtol,args.atol,report['zeta_hopf'],1)]
    if any(x['status']=='UNRESOLVED' for x in rows):raise RuntimeError('Initial Floquet validation failed')
    ds=args.ds;failures=[]
    for k in range(args.steps):
        tangent=(x1-x0)/np.linalg.norm(x1-x0)
        attempt=0;sol=None
        while ds>=args.ds_min*(1-1e-10):
            attempt+=1;predictor=x1+ds*tangent
            if not (0<predictor[7]/args.zeta_scale<.99999):
                failures.append(dict(step=k+2,ds=ds,reason='predictor outside coupling domain'));ds*=.5;continue
            sol,info=correct(predictor,tangent,ye,v,base,args.zeta_scale,args.rtol,args.atol,1e-10,args.maxfev)
            if info['success'] and np.linalg.norm(sol-x1)>.05*ds:break
            failures.append(dict(step=k+2,ds=ds,reason=info['message'],residual=info['residual']))
            sol=None;ds*=.5
        if sol is None:
            print('Stopping: corrector could not converge at minimum step',flush=True);break
        row=entry(sol,args.zeta_scale,base,args.rtol,args.atol,report['zeta_hopf'],k+2,
                  dict(**info,ds=ds,attempts=attempt))
        if row['status']=='UNRESOLVED':
            failures.append(dict(step=k+2,ds=ds,reason='Floquet or periodicity validation failed'))
            print('Stopping: Floquet validation failed',flush=True);break
        rows.append(row)
        print(f"step {k+2}: zeta={row['zeta']:.10f} T={row['period']:.7f} "
              f"max|mu|={row['max_nontrivial_modulus']:.5g} unstable={row['unstable_count']} ds={ds:.4g}",flush=True)
        x0,x1=x1,sol
        if row['zeta']<=args.zeta_stop:break
        if attempt==1:ds=min(args.ds_max,ds*1.25)
    # Diagnostic only: do not label a secondary bifurcation without multiplier tracking.
    for i in range(1,len(rows)):
        rows[i]['zeta_turn_candidate']=bool((rows[i]['zeta']-rows[i-1]['zeta'])*
            (rows[i-1]['zeta']-rows[i-2]['zeta'])<0) if i>=2 else False
        rows[i]['unstable_count_change_candidate']=rows[i]['unstable_count']!=rows[i-1]['unstable_count']
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    result=dict(scope='Representative normalized Harb benchmark, not published K0 reproduction',
      parameters=dict(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02),
      assumptions=dict(omega=args.omega,W=args.W),hopf=report,settings=dict(zeta_scale=args.zeta_scale,
      ds=args.ds,steps=args.steps,zeta_stop=args.zeta_stop),orbits=rows,failed_attempts=failures,
      limitations=['Continuation is finite and may terminate before requested coupling',
       'Near-unit multipliers or unstable-count changes are candidates requiring independent refinement',
       'No physical K0 mapping','No global claim about other periodic or chaotic attractors'])
    (out/'p13_report.json').write_text(json.dumps(result,indent=2)+'\n')
    cols=['step','zeta','delta_from_hopf','period','periodicity_error','neutral_error','unstable_count',
          'max_nontrivial_modulus','zeta_turn_candidate','unstable_count_change_candidate','status']
    with (out/'p13_branch.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot([r['zeta'] for r in rows],[r['period'] for r in rows],'.-')
        ax.axvline(report['zeta_hopf'],color='black',ls='--',label='P11 Hopf')
        ax.set(xlabel='Coupling ζ',ylabel='Orbit period',title='P13 pseudo-arclength continuation (normalized benchmark)')
        ax.legend();fig.tight_layout();fig.savefig(out/'p13_branch.png',dpi=300);plt.close(fig)
    except Exception as e:print('Plot unavailable:',e)
    print(f'P13: {len(rows)} verified orbit records, {len(failures)} failed attempts. Output: {out}',flush=True)
    if len(rows)<3:sys.exit(2)
if __name__=='__main__':main()

