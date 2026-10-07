#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p11_author_hopf_verification.py

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
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from p00_model_validation import Parameters
from p01_rotating_frame_validation import rotating_rhs
from p02_equilibrium_hopf_analysis import equilibrium_branches,equilibrium_jacobian,finite_difference_jacobian
from p03_hopf_refinement import multilinear,first_lyapunov
BASE=dict(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02)

def state(z,base,branch):
 p=replace(base,zeta=float(z)); ys=dict(equilibrium_branches(p))
 if branch not in ys:raise ValueError(f'branch {branch} missing at zeta {z}')
 y=ys[branch]; A=equilibrium_jacobian(y,p)
 return y,p,A

def pair(z,base,branch,anchor):
 y,p,A=state(z,base,branch)
 eig,vec=np.linalg.eig(A)
 inds=np.where(eig.imag>1e-6)[0]
 if not len(inds):raise ValueError('no positive imaginary eigenvalue')
 k=inds[np.argmin(abs(eig[inds]-anchor))]
 return eig[k],vec[:,k],eig,y,p,A

def derivative_checks(y,p,B,C):
 rng=np.random.default_rng(11017)
 u=rng.normal(size=6);u/=np.linalg.norm(u)
 v=rng.normal(size=6);v/=np.linalg.norm(v)
 h=1e-3;f=lambda x:rotating_rhs(0,x,p)
 bfd=(f(y+h*u+h*v)-f(y+h*u-h*v)-f(y-h*u+h*v)+f(y-h*u-h*v))/(4*h*h)
 cfd=(f(y+2*h*u)-2*f(y+h*u)+2*f(y-h*u)-f(y-2*h*u))/(2*h**3)
 return dict(B_directional_error=float(max(abs(B(u,v)-bfd))),C_directional_error=float(max(abs(C(u,u,u)-cfd))))

def calculate(args):
 base=Parameters(**BASE,zeta=args.guess,omega=args.omega,W=args.W)
 anchor=complex(0,args.frequency)
 def f(z):return pair(z,base,args.branch,anchor)[0].real
 lo,hi=args.bracket
 if f(lo)*f(hi)>=0:raise ValueError(f'bracket does not straddle crossing: {f(lo)}, {f(hi)}')
 root=float(brentq(f,lo,hi,xtol=1e-13,rtol=1e-14))
 lam,q,eig,y,p,A=pair(root,base,args.branch,anchor)
 h=min(1e-4,(hi-lo)/8)
 slope=(f(root+h)-f(root-h))/(2*h)
 slope_half=(f(root+h/2)-f(root-h/2))/h
 other=[v for v in eig if abs(v-lam)>1e-6 and abs(v-lam.conjugate())>1e-6]
 gap=float(min(abs(v.real) for v in other))
 B,C=multilinear(y,p)
 l1,norm_err=first_lyapunov(A,q,float(lam.imag),B,C)
 derivatives=derivative_checks(y,p,B,C)
 jerr=float(max(abs(A-finite_difference_jacobian(y,p)).ravel()))
 residual=float(max(abs(rotating_rhs(0,y,p))))
 other_unstable=sum(v.real>1e-7 for v in other)
 tests=dict(equilibrium_residual=residual,jacobian_error=jerr,adjoint_normalization_error=norm_err,
  B_directional_error=derivatives['B_directional_error'],C_directional_error=derivatives['C_directional_error'],
  crossing_real_part=abs(lam.real),other_eigenvalue_real_gap=gap,
  transversality_slope_difference=abs(slope-slope_half),nonzero_l1=abs(l1)>1e-7)
 verified=(residual<1e-8 and jerr<1e-6 and norm_err<1e-8 and derivatives['B_directional_error']<1e-5 and derivatives['C_directional_error']<1e-4 and abs(lam.real)<1e-9 and gap>1e-5 and abs(slope)>1e-5 and abs(slope-slope_half)/abs(slope)<.01 and abs(l1)>1e-7)
 result=dict(scope='Harb representative benchmark; not published K0 reproduction',parameters=BASE,
  assumptions=dict(omega=args.omega,W=args.W),branch=args.branch,initial_bracket=list(args.bracket),
  zeta_hopf=root,eigenvalue=[float(lam.real),float(lam.imag)],period_at_onset=float(2*np.pi/lam.imag),
  transversality_dreal_dzeta=float(slope),first_lyapunov_coefficient=float(l1),
  l1_convention='Kuznetsov l1=Re(G21)/(2*omega), Euclidean unit right eigenvector',
  other_eigenvalues=[[float(v.real),float(v.imag)] for v in other],other_unstable_eigenvalues=int(other_unstable),
  tests=tests,status='NONDEGENERATE_HOPF_VERIFIED' if verified else 'UNRESOLVED',
  local_branch_direction=('zeta below Hopf' if l1*slope>0 else 'zeta above Hopf'),
  critical_pair_radial_stability=('stable in center directions' if l1<0 else 'unstable in center directions'),
  full_system_attracting_cycle_excluded=bool(other_unstable>0),
  limitations=['No physical K0-to-normalized coefficient mapping','Only branch '+args.branch+' near supplied bracket is analyzed',
  'First Lyapunov coefficient is local; no finite-amplitude cycle or Floquet continuation performed',
  'If other eigenvalues are unstable, small Hopf cycles inherit those unstable directions',
  'omega and W are test assumptions, not confirmed figure-specific values'])
 return result,base

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--omega',type=float,default=1.);ap.add_argument('--W',type=float,default=100.)
 ap.add_argument('--branch',default='00');ap.add_argument('--guess',type=float,default=.91750958)
 ap.add_argument('--frequency',type=float,default=.22826782)
 ap.add_argument('--bracket',nargs=2,type=float,default=[.915,.920]);ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'author_hopf_verification')
 ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 if args.W<=0 or not(0<args.bracket[0]<args.bracket[1]<1):ap.error('Require W>0 and 0<lower<upper<1')
 report,base=calculate(args);args.output_dir.mkdir(parents=True,exist_ok=True)
 (args.output_dir/'p11_report.json').write_text(json.dumps(report,indent=2)+'\n')
 with (args.output_dir/'p11_hopf_summary.csv').open('w',newline='') as fh:
  fields=['branch','zeta_hopf','frequency','period_at_onset','transversality_dreal_dzeta','first_lyapunov_coefficient','other_unstable_eigenvalues','status'];w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerow({k:(report['eigenvalue'][1] if k=='frequency' else report.get(k)) for k in fields})
 if not args.self_test:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  zs=np.linspace(args.bracket[0],args.bracket[1],101);vals=[pair(z,base,args.branch,complex(0,args.frequency))[0].real for z in zs]
  fig,ax=plt.subplots(figsize=(8,5));ax.plot(zs,vals,label='Tracked complex eigenvalue');ax.axhline(0,color='black',ls='--');ax.axvline(report['zeta_hopf'],color='gray',ls=':');ax.set(xlabel='Coupling zeta',ylabel='Real eigenvalue',title='P11: author-benchmark Hopf verification (no K0 calibration)');ax.legend();fig.tight_layout();fig.savefig(args.output_dir/'p11_hopf_refinement.png',dpi=300);plt.close(fig)
 print(json.dumps(report,indent=2));print('Output:',args.output_dir)
 if report['status']!='NONDEGENERATE_HOPF_VERIFIED':sys.exit(2)
if __name__=='__main__':main()

