#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p21_fixed_coupling.py

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
from scipy.integrate import solve_ivp
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p04_periodic_orbit_floquet import integrate_orbit
from p19_multiple_shooting import initial_segments,system

def load(p):return json.loads(p.read_text())
def correct(seed,target,base,a):
    y=np.asarray(seed['initial_state'],float);T=float(seed['period']);m=a.segments
    p=replace(base,zeta=float(seed['zeta']))
    Y=initial_segments(y,T,p,m,a.rtol,a.atol)
    anchor=y.copy();phase=rotating_rhs(0,y,p);phase/=np.linalg.norm(phase)
    x0=np.r_[Y.ravel(),T]
    def eval_fixed(x,rtol=None,atol=None):
        w=np.r_[x,target]
        F,J,_,Ms=system(w,m,base,anchor,phase,rtol or a.rtol,atol or a.atol)
        return F,J,Ms
    def fun(x):return eval_fixed(x)[0]
    def jac(x):return eval_fixed(x)[1]
    sol=root(fun,x0,jac=jac,method='hybr',options={'xtol':a.xtol,'maxfev':a.maxfev})
    x=sol.x;F,J,Ms=eval_fixed(x,a.rtol*.3,a.atol*.3)
    y0=x[:6];period=float(x[6*m]);p=replace(base,zeta=target)
    yend,M,_=integrate_orbit(y0,period,p,a.rtol*.1,a.atol*.1,True)
    mu=np.linalg.eigvals(M);neutral=int(np.argmin(abs(mu-1)));non=np.delete(mu,neutral)
    residual=float(np.max(abs(F)));single=float(np.max(abs(yend-y0)))
    valid=bool(residual<a.defect_tol and single<a.single_tol and abs(mu[neutral]-1)<a.neutral_tol and a.period_min<period<a.period_max)
    # Phase-invariant orbit fingerprint, based on state components invariant to a common phase shift.
    sol_ivp=solve_ivp(lambda t,q:rotating_rhs(t,q,p),(0,period),y0,method='DOP853',rtol=a.rtol,atol=a.atol,dense_output=True,max_step=.25)
    if not sol_ivp.success:raise RuntimeError(sol_ivp.message)
    states=sol_ivp.sol(np.linspace(0,period,64,endpoint=False)).T
    # Relative phase and both velocity/acceleration pairs; compare with cyclic alignment.
    sig=np.column_stack([np.cos(states[:,0]-states[:,3]),np.sin(states[:,0]-states[:,3]),states[:,1],states[:,2],states[:,4],states[:,5]])
    return dict(seed_step=seed['step'],seed_zeta=float(seed['zeta']),zeta=target,period=period,initial_state=y0.tolist(),segment_states=x[:6*m].reshape(m,6).tolist(),solver_success=bool(sol.success),solver_message=str(sol.message),nfev=int(sol.nfev),shooting_defect=residual,independent_periodicity_error=single,neutral_error=float(abs(mu[neutral]-1)),unstable_count=int(np.sum(abs(non)>1+1e-3)),max_nontrivial_modulus=float(np.max(abs(non))),multipliers=[[float(q.real),float(q.imag)] for q in mu],validated=valid,signature=sig.tolist())

def distance(a,b):
    # Avoid conflating close periods with same orbit; cyclic phase matching.
    A=np.asarray(a['signature']);B=np.asarray(b['signature']);scale=np.array([1,1,.2,.2,.2,.2]);
    d=min(np.sqrt(np.mean(((A-np.roll(B,k,axis=0))*scale)**2)) for k in range(64))
    return float(d)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p15-report',type=Path,required=True)
    ap.add_argument('--targets',type=float,nargs='+',default=[.899,.901,.903])
    ap.add_argument('--arc-ranges',type=int,nargs='+',default=[0,50,51,86,87,108],help='Pairs of inclusive P15 step bounds')
    ap.add_argument('--seeds-per-arc',type=int,default=2)
    ap.add_argument('--segments',type=int,default=4)
    ap.add_argument('--rtol',type=float,default=3e-10);ap.add_argument('--atol',type=float,default=3e-12)
    ap.add_argument('--xtol',type=float,default=1e-10);ap.add_argument('--maxfev',type=int,default=150)
    ap.add_argument('--defect-tol',type=float,default=1e-8);ap.add_argument('--single-tol',type=float,default=3e-8)
    ap.add_argument('--neutral-tol',type=float,default=2e-4);ap.add_argument('--period-min',type=float,default=20);ap.add_argument('--period-max',type=float,default=45)
    ap.add_argument('--orbit-distance-tol',type=float,default=.01);ap.add_argument('--period-distance-tol',type=float,default=.02)
    ap.add_argument('--output-dir',type=Path,default=Path('results/author_fixed_coupling'))
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if len(a.arc_ranges)%2:ap.error('--arc-ranges must be pairs')
    src=load(a.p15_report);base=Parameters(**src['parameters'],omega=src['assumptions']['omega'],W=src['assumptions']['W'],zeta=.9)
    orbits=src['orbits'];arcs=[]
    for low,high in zip(a.arc_ranges[::2],a.arc_ranges[1::2]):
        rr=[r for r in orbits if low<=int(r['step'])<=high];arcs.append((f'{low}-{high}',rr))
    out=dict(scope=src.get('scope'),source_p15=str(a.p15_report),parameters=src['parameters'],assumptions=src['assumptions'],settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},targets=[],errors=[],limitations=['Multiple periodic orbits are not necessarily attracting','No physical K0 calibration','Only sampled P15 branch arcs are surveyed; additional branches may exist'])
    for target in a.targets:
        print(f'P21 target zeta={target:.9f}',flush=True);rec=dict(zeta=target,arcs=[],distinct_orbits=[],pairwise_distances=[],coexisting_periodic_orbits=0,attracting_orbits=0)
        for name,rows in arcs:
            arc=dict(arc=name,seed_count=len(rows),attempts=[],selected=None)
            seeds=sorted(rows,key=lambda r:abs(r['zeta']-target))[:a.seeds_per_arc]
            for seed in seeds:
                try:
                    r=correct(seed,target,base,a);arc['attempts'].append(r)
                    print(f'  arc {name} seed={seed["step"]} T={r["period"]:.7f} residual={r["shooting_defect"]:.2e} independent={r["independent_periodicity_error"]:.2e} unstable={r["unstable_count"]} valid={r["validated"]}',flush=True)
                    if r['validated']:
                        arc['selected']=r;break
                except Exception as exc:
                    arc['attempts'].append(dict(seed_step=seed['step'],error=str(exc),validated=False));out['errors'].append(dict(zeta=target,arc=name,seed_step=seed['step'],error=str(exc)))
            rec['arcs'].append(arc)
            if arc['selected'] is not None:
                candidate=arc['selected'];same=None
                for j,prev in enumerate(rec['distinct_orbits']):
                    dd=distance(candidate,prev)
                    if abs(candidate['period']-prev['period'])<a.period_distance_tol and dd<a.orbit_distance_tol:same=j;break
                if same is None:rec['distinct_orbits'].append(candidate)
                else:arc['duplicate_of']=same
        rec['coexisting_periodic_orbits']=len(rec['distinct_orbits']);rec['attracting_orbits']=sum(x['unstable_count']==0 for x in rec['distinct_orbits'])
        rec['pairwise_distances']=[dict(i=i,j=j,distance=distance(x,y),period_difference=abs(x['period']-y['period'])) for i,x in enumerate(rec['distinct_orbits']) for j,y in enumerate(rec['distinct_orbits']) if j>i]
        out['targets'].append(rec)
        temp=a.output_dir/'p21_report.tmp';temp.write_text(json.dumps(out,indent=2)+'\n');os.replace(temp,a.output_dir/'p21_report.json')
    with (a.output_dir/'p21_orbits.csv').open('w',newline='') as f:
        cols=['zeta','arc','seed_step','period','shooting_defect','independent_periodicity_error','neutral_error','unstable_count','max_nontrivial_modulus','validated','duplicate_of'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for target in out['targets']:
            for arc in target['arcs']:
                r=arc['selected']
                if r:w.writerow({k:target['zeta'] if k=='zeta' else arc['arc'] if k=='arc' else arc.get('duplicate_of','') if k=='duplicate_of' else r.get(k) for k in cols})
    print(f'P21 complete: {len(out["targets"])} target couplings; {len(out["errors"])} errors',flush=True)
if __name__=='__main__':main()

