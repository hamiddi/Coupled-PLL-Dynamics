#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p09_author_parameter_validation.py

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
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from p00_model_validation import Parameters,rhs,jacobian,finite_difference_jacobian
from p01_rotating_frame_validation import rotating_rhs,to_rotating,to_original

BENCHMARK = dict(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02)
PUBLISHED_HOPF_K0 = {'0.0':100.0,'0.5':1500.0,'0.8':2310.0,'1.0':3665.4}
# The PDF provides no unique omega, W, or original initial state. These are
# intentionally explicit, user-adjustable NUMERICAL TEST SETTINGS.
DEFAULT_X0 = [.1,0.,0.,-.15,0.,0.,0.]

def integrate(p,x0,t_end,n_points,rtol,atol):
    times=np.linspace(0,t_end,n_points)
    full=solve_ivp(lambda t,x:rhs(t,x,p),(0,t_end),x0,method='DOP853',
                   t_eval=times,rtol=rtol,atol=atol)
    red0=to_rotating(0.,np.asarray(x0),p)
    red=solve_ivp(lambda t,y:rotating_rhs(t,y,p),(0,t_end),red0,
                  method='DOP853',t_eval=times,rtol=rtol,atol=atol)
    if not full.success or not red.success:
        raise RuntimeError(f'Integration failed: {full.message}; {red.message}')
    # Absolute phases can drift linearly, but both descriptions use the same
    # clock and are compared at the same physical times.
    mapped=np.stack([to_rotating(float(t),full.y[:,j],p)
                     for j,t in enumerate(times)],axis=1)
    differences=mapped-red.y
    scale=1+np.max(np.abs(mapped),axis=1)[:,None]
    return full,red,float(np.max(np.abs(differences))),float(np.max(np.abs(differences)/scale))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'author_parameter_validation')
    ap.add_argument('--omega',type=float,default=1.,help='TEST ASSUMPTION, not confirmed publication value')
    ap.add_argument('--W',type=float,default=100.,help='TEST ASSUMPTION; PDF states range 100-1000')
    ap.add_argument('--initial-state',nargs=7,type=float,default=DEFAULT_X0,metavar='X',
                    help='x1 x2 x3 x4 x5 x6 x7; TEST ASSUMPTION')
    ap.add_argument('--zetas',nargs='+',type=float,default=[0.,.5,.8,1.])
    ap.add_argument('--t-end',type=float,default=30.)
    ap.add_argument('--points',type=int,default=301)
    ap.add_argument('--rtol',type=float,default=1e-10)
    ap.add_argument('--atol',type=float,default=1e-12)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.W<100 or args.W>1000:
        print('WARNING: W is outside the PDF stated representative range [100,1000].')
    if args.W==0 or args.points<2 or args.t_end<=0: ap.error('W nonzero, points >=2, t-end >0 required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    x0=np.asarray(args.initial_state,dtype=float)
    rows=[]; trajectories={}
    for z in args.zetas:
        p=Parameters(**BENCHMARK,zeta=z,omega=args.omega,W=args.W)
        jacerr=float(np.max(np.abs(jacobian(0.,x0,p)-finite_difference_jacobian(x0,p))))
        full,red,abs_err,rel_err=integrate(p,x0,args.t_end,args.points,args.rtol,args.atol)
        row=dict(zeta=z,omega=args.omega,W=args.W,omega_over_W=args.omega/args.W,
                 jacobian_max_abs_error=jacerr,frame_max_abs_error=abs_err,
                 frame_max_scaled_error=rel_err,
                 seven_state_clock_error=float(np.max(np.abs(full.y[6]-(full.t+x0[6])))),
                 published_Hopf_K0_rad_s=PUBLISHED_HOPF_K0.get(f'{z:.1f}'),
                 K0_mapping_available=False,
                 jacobian_pass=jacerr<1e-7,frame_pass=rel_err<2e-7,
                 status='PASS' if jacerr<1e-7 and rel_err<2e-7 else 'CHECK')
        rows.append(row)
        trajectories[f'zeta_{z:.4f}']={'t':full.t.tolist(),'original_x':full.y.T.tolist(),
                                     'rotating_y':red.y.T.tolist()}
        print(f'zeta={z:g}: {row["status"]}; Jacobian={jacerr:.3e}; frame={rel_err:.3e}; '
              f'published Hopf K0={row["published_Hopf_K0_rad_s"]}')
    meta={'source':'Harb supplied PDF pp. 3-5; representative benchmark, not figure-specific settings',
          'benchmark_parameters':BENCHMARK,'omega':args.omega,'W':args.W,
          'initial_state':x0.tolist(),'test_assumptions':['omega','W','initial_state'],
          'publication_calibration':{'K0_to_normalized_coefficients':'NOT PROVIDED',
             'figure_specific_initial_conditions':'NOT PROVIDED',
             'original_MATLAB_files':'NOT PROVIDED'},
          'published_Hopf_K0_rad_s':PUBLISHED_HOPF_K0,
          'interpretation':'Mathematical and numerical consistency only; no published K0 bifurcation reproduction',
          'results':rows}
    (args.output_dir/'p09_report.json').write_text(json.dumps(meta,indent=2))
    with (args.output_dir/'p09_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if not args.smoke:
        (args.output_dir/'p09_trajectories.json').write_text(json.dumps(trajectories))
    print('Output:',args.output_dir)
    if any(r['status']!='PASS' for r in rows):sys.exit(1)
if __name__=='__main__':main()

