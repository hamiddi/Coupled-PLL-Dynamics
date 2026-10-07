#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p22_attractor_survey.py

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
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p04_periodic_orbit_floquet import full_jacobian


def equilibria(p):
    # At an equilibrium all velocities and accelerations vanish.  Both phase
    # equations reduce to sin(theta_i)=-(delta_i+c*omega/W).
    rhs1=-(p.delta1+p.c*p.omega/p.W)
    rhs2=-(p.delta2+p.c*p.omega/p.W)
    if max(abs(rhs1),abs(rhs2))>1: return []
    t1=np.arcsin(rhs1);t2=np.arcsin(rhs2)
    result=[]
    for i,a in enumerate((t1,np.pi-t1)):
        for j,b in enumerate((t2,np.pi-t2)):
            # theta1=y4-y1, theta2=zeta*y1-y4 -> y1=(t1+t2)/(zeta-1).
            if abs(p.zeta-1)<1e-10: continue
            u=(a+b)/(p.zeta-1); w=u+a
            y=np.array([u,0,0,w,0,0],float)
            residual=float(np.linalg.norm(rotating_rhs(0,y,p),ord=np.inf))
            ev=np.linalg.eigvals(full_jacobian(y,p))
            result.append(dict(id=f'E{i}{j}',state=y.tolist(),residual=residual,
                max_real_eigenvalue=float(max(ev.real)),
                eigenvalues=[[float(z.real),float(z.imag)] for z in ev],
                locally_asymptotically_stable=bool(max(ev.real)<-1e-7)))
    return result


def phases(y,p):
    return np.array([y[3]-y[0],p.zeta*y[0]-y[3]])


def integrate(y0,p,duration,dt,rtol,atol,max_abs):
    if duration<=0: return np.array([0.]),np.array([y0])
    t=np.linspace(0,duration,max(2,int(np.ceil(duration/dt))+1))
    def escape(_t,y):return max_abs-np.max(np.abs(y))
    escape.terminal=True;escape.direction=-1
    sol=solve_ivp(lambda t,y:rotating_rhs(t,y,p),(0,duration),y0,method='DOP853',
        t_eval=t,rtol=rtol,atol=atol,max_step=min(.5,dt),events=escape)
    if not sol.success or sol.t.size!=t.size:
        raise RuntimeError(f'integration failed/escaped: {sol.message}; last_t={sol.t[-1] if len(sol.t) else None}')
    return sol.t,sol.y.T


def signature(Y,p):
    theta=np.array([phases(y,p) for y in Y]); vel=Y[:,[1,2,4,5]]
    Z=np.column_stack([np.sin(theta),np.cos(theta),vel])
    return np.r_[Z.mean(axis=0),Z.std(axis=0)]


def largest_ftle(y,p,duration,chunk,rtol,atol,max_abs):
    if duration<=0:return None
    vec=np.ones(6)/np.sqrt(6); total=0.; t=0.
    while t<duration-1e-10:
        h=min(chunk,duration-t)
        def rhs_aug(tt,u):
            x=u[:6];return np.r_[rotating_rhs(tt,x,p),full_jacobian(x,p)@u[6:]]
        sol=solve_ivp(rhs_aug,(0,h),np.r_[y,vec],method='DOP853',rtol=rtol,
            atol=atol,max_step=min(.5,chunk/4))
        if not sol.success or np.max(np.abs(sol.y[:6,-1]))>max_abs:
            return None
        y=sol.y[:6,-1]; v=sol.y[6:,-1]; norm=np.linalg.norm(v)
        if not np.isfinite(norm) or norm<=0:return None
        total+=np.log(norm);vec=v/norm;t+=h
    return float(total/t)


def make_seeds(p,eqs,arcs,rng,n_random,n_perturb,scale):
    seeds=[]
    for e in eqs:
        x=np.array(e['state']);seeds.append((f"{e['id']}_exact",x))
        for k in range(n_perturb):seeds.append((f"{e['id']}_pert{k}",x+rng.normal(0,scale,6)))
    for arc in arcs:
        selected=arc.get('selected') or {}
        if isinstance(selected,list):selected=selected[0] if selected else {}
        if not isinstance(selected,dict):continue
        x=selected.get('initial_state')
        if x is None:
            attempts=[a for a in arc.get('attempts',[]) if a.get('validated')]
            x=attempts[0].get('initial_state') if attempts else None
        if x is None:continue
        x=np.array(x,float)
        for k in range(n_perturb):
            seeds.append((f"orbit_{arc['arc']}_pert{k}",x+rng.normal(0,scale,6)))
    for k in range(n_random):
        # Solve for phase coordinates so both physical detector phases cover [-pi,pi].
        th1,th2=rng.uniform(-np.pi,np.pi,2)
        y0=(th1+th2)/(p.zeta-1);y3=y0+th1
        seeds.append((f'random_{k:03d}',np.array([y0,*rng.normal(0,.5,2),y3,*rng.normal(0,.5,2)])))
    return seeds


def run(args):
    report=json.loads(Path(args.p21_report).read_text())
    base=report['parameters'];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    targets=args.targets if args.targets else [float(t['zeta']) for t in report['targets']]
    by_z={round(float(t['zeta']),9):t for t in report['targets']}
    rows=[];eqrows=[];summary=[];rng=np.random.default_rng(args.seed)
    for z in targets:
        if round(z,9) not in by_z:raise ValueError(f'ζ={z} missing in P21 report')
        p=Parameters(**base,zeta=z,omega=args.omega,W=args.W)
        eqs=equilibria(p)
        for e in eqs:eqrows.append(dict(zeta=z,**e))
        seeds=make_seeds(p,eqs,by_z[round(z,9)]['arcs'],rng,args.n_random,args.n_perturb,args.perturb_scale)
        print(f'zeta={z:.6f} equilibria={len(eqs)} seeds={len(seeds)}',flush=True)
        for idx,(name,x) in enumerate(seeds):
            row=dict(zeta=z,seed_id=name,seed_index=idx,status='unresolved',
                     initial_state=x.tolist(),lyapunov_ftle=None)
            try:
                _,trans=integrate(x,p,args.transient,args.dt,args.rtol,args.atol,args.max_abs)
                y=trans[-1]
                _,obs=integrate(y,p,args.observe,args.dt,args.rtol,args.atol,args.max_abs)
                split=max(2,len(obs)//2)
                s1=signature(obs[:split],p);s2=signature(obs[split:],p)
                stationarity=float(np.linalg.norm(s1-s2)/(1+np.linalg.norm(s2)))
                residuals=np.array([np.linalg.norm(rotating_rhs(0,v,p),ord=np.inf) for v in obs])
                residual=float(np.max(residuals[len(obs)//2:]))
                distances=[np.linalg.norm(obs[-1]-np.array(e['state'])) for e in eqs]
                closest=int(np.argmin(distances)) if eqs else None
                eqmatch=bool(closest is not None and distances[closest]<args.eq_tol and residual<args.eq_tol)
                if eqmatch:
                    if name.endswith('_exact') and not eqs[closest]['locally_asymptotically_stable']:
                        status='exact_unstable_equilibrium_not_attractor'
                    else:
                        status='converged_equilibrium'
                elif stationarity<=args.stationarity_tol:
                    status='stationary_non_equilibrium_candidate'
                else:status='nonstationary_or_long_transient'
                ftle=None
                if status!='converged_equilibrium' and args.lyapunov_time>0:
                    ftle=largest_ftle(obs[-1],p,args.lyapunov_time,args.lyapunov_chunk,
                                      args.rtol,args.atol,args.max_abs)
                row.update(status=status,final_state=obs[-1].tolist(),
                    signature=s2.tolist(),stationarity_score=stationarity,
                    max_late_rhs_residual=residual,closest_equilibrium=eqs[closest]['id'] if eqmatch else None,
                    lyapunov_ftle=ftle,observation_amplitude=float(np.max(np.ptp(obs,axis=0))),
                    final_phase_detectors=phases(obs[-1],p).tolist())
            except Exception as exc:
                row.update(status='integration_failed_or_escaped',error=str(exc))
            rows.append(row)
            if (idx+1)%10==0:print(f'  completed {idx+1}/{len(seeds)}',flush=True)
            # Atomic checkpoint for long HPC jobs.
            temp=out/'p22_checkpoint.tmp';temp.write_text(json.dumps(rows,indent=2,allow_nan=False));temp.replace(out/'p22_checkpoint.json')
        local=[r for r in rows if r['zeta']==z]
        summary.append(dict(zeta=z,n_seeds=len(seeds),stable_equilibria=[e['id'] for e in eqs if e['locally_asymptotically_stable']],
            status_counts={s:sum(r['status']==s for r in local) for s in sorted({r['status'] for r in local})}))
    # Candidate clustering uses stationary, non-equilibrium signatures only;
    # nearby signatures alone do not establish identical invariant attractors.
    clusters=[]
    for z in targets:
        group=[r for r in rows if r['zeta']==z and r['status']=='stationary_non_equilibrium_candidate']
        for row in group:
            sig=np.array(row['signature']);found=None
            for c in clusters:
                if c['zeta']==z and np.linalg.norm(sig-np.array(c['representative_signature']))<args.cluster_tol:
                    found=c;break
            if found is None:
                found=dict(zeta=z,cluster_id=f'z{z:.3f}_C{sum(c["zeta"]==z for c in clusters):02d}',
                           representative_signature=sig.tolist(),members=[])
                clusters.append(found)
            found['members'].append(row['seed_id'])
            row['candidate_cluster']=found['cluster_id']
    payload=dict(scope='Harb representative normalized benchmark; NOT physical K0 calibration',
        source_p21=str(args.p21_report),settings=vars(args),summary=summary,
        equilibria=eqrows,trajectories=rows,candidate_clusters=clusters,
        limitations=['Stable equilibrium classification is local linearization only.',
          'Non-equilibrium clusters are provisional trajectory signatures, not proven attractors.',
          'Finite-time Lyapunov exponents are diagnostic, not a chaos proof.',
          'Finite ensemble and observation times cannot establish exhaustive multistability.',
          'Large absolute rotating-frame phases may represent drift; max_abs escape is a numerical safeguard.'])
    (out/'p22_report.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    fields=['zeta','seed_id','status','stationarity_score','max_late_rhs_residual','closest_equilibrium',
            'lyapunov_ftle','observation_amplitude','candidate_cluster','error']
    with (out/'p22_trajectories.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:r.get(k) for k in fields})
    with (out/'p22_equilibria.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['zeta','id','residual','max_real_eigenvalue','locally_asymptotically_stable']);writer.writeheader()
        for r in eqrows:writer.writerow({k:r[k] for k in writer.fieldnames})
    (out/'p22_candidate_clusters.json').write_text(json.dumps(clusters,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('Wrote',out/'p22_report.json')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--p21-report',default='results/author_fixed_coupling/p21_report.json')
    ap.add_argument('--output-dir',default='results/author_attractor_survey')
    ap.add_argument('--targets',type=float,nargs='*',default=None)
    ap.add_argument('--omega',type=float,default=1.);ap.add_argument('--W',type=float,default=100.)
    ap.add_argument('--n-random',type=int,default=24)
    ap.add_argument('--n-perturb',type=int,default=3)
    ap.add_argument('--perturb-scale',type=float,default=.02)
    ap.add_argument('--seed',type=int,default=20260930)
    ap.add_argument('--transient',type=float,default=600.)
    ap.add_argument('--observe',type=float,default=300.)
    ap.add_argument('--lyapunov-time',type=float,default=150.)
    ap.add_argument('--lyapunov-chunk',type=float,default=2.)
    ap.add_argument('--dt',type=float,default=.5)
    ap.add_argument('--rtol',type=float,default=1e-9)
    ap.add_argument('--atol',type=float,default=1e-11)
    ap.add_argument('--max-abs',type=float,default=1e6)
    ap.add_argument('--eq-tol',type=float,default=1e-5)
    ap.add_argument('--stationarity-tol',type=float,default=.05)
    ap.add_argument('--cluster-tol',type=float,default=.15)
    args=ap.parse_args()
    if min(args.n_random,args.n_perturb,args.transient,args.observe,args.dt)<=0:
        ap.error('counts, durations and dt must be positive')
    run(args)
if __name__=='__main__':main()

