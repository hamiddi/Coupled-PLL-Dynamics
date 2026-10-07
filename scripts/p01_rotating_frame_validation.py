#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p01_rotating_frame_validation.py

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
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from p00_model_validation import Parameters, rhs, jacobian


def rotating_rhs(t: float, y: np.ndarray, p: Parameters) -> np.ndarray:
    """Autonomous six-state system in co-rotating coordinates.

    y1=x1+r*t, y2=x2+r, y3=x3, y4=x4+r*t,
    y5=x5+r, y6=x6; r=omega/W; t is normalized time.
    """
    del t
    p.validate()
    y1, y2, y3, y4, y5, y6 = np.asarray(y, dtype=float)
    r = p.omega / p.W
    th1, th2 = y4-y1, p.zeta*y1-y4
    v1, a1 = y5-y2, y6-y3
    v2, a2 = p.zeta*y2-y5, p.zeta*y3-y6
    return np.array([
        y2, y3,
        p.delta1 + p.c*r - p.a*y3 - p.c*y2
        + p.b*np.cos(th1)*a1 - p.b*np.sin(th1)*v1*v1
        + p.d*np.cos(th1)*v1 + np.sin(th1),
        y5, y6,
        p.delta2 + p.c*r - p.a*y6 - p.c*y5
        + p.b*np.cos(th2)*a2 - p.b*np.sin(th2)*v2*v2
        + p.d*np.cos(th2)*v2 + np.sin(th2)
    ], dtype=float)


def to_rotating(t: float, x: np.ndarray, p: Parameters) -> np.ndarray:
    """Transform original physical states to the six co-rotating states."""
    r = p.omega / p.W
    y = np.array(x[:6], dtype=float, copy=True)
    y[[0, 3]] += r * float(x[6])  # x7 is normalized time plus its initial offset
    y[[1, 4]] += r
    return y


def to_original(y: np.ndarray, clock: float, p: Parameters) -> np.ndarray:
    """Inverse transform for any value of the original clock x7."""
    r = p.omega / p.W
    x = np.array(y, dtype=float, copy=True)
    x[[0, 3]] -= r * clock
    x[[1, 4]] -= r
    return np.r_[x, clock]


def symbolic_proof() -> bool:
    y = sp.symbols('y1:7', real=True)
    a,b,c,d,D1,D2,z,r = sp.symbols('a b c d D1 D2 z r', real=True)
    u,v,w,h,j,k = y
    th1, th2 = h-u, z*u-h
    v1, acc1 = j-v, k-w
    v2, acc2 = z*v-j, z*w-k
    reduced = sp.Matrix([
        v,w,D1+c*r-a*w-c*v+b*sp.cos(th1)*acc1-b*sp.sin(th1)*v1**2+d*sp.cos(th1)*v1+sp.sin(th1),
        j,k,D2+c*r-a*k-c*j+b*sp.cos(th2)*acc2-b*sp.sin(th2)*v2**2+d*sp.cos(th2)*v2+sp.sin(th2)
    ])
    # Independently reconstruct original equations under x1=y1-r*t, x2=y2-r,
    # x4=y4-r*t, x5=y5-r, q=(z-1)r; theta2 loses explicit time.
    t=sp.symbols('t', real=True)
    x1,x2,x3,x4,x5,x6=u-r*t,v-r,w,h-r*t,j-r,k
    theta1=x4-x1
    theta2=z*x1-x4+(z-1)*r*t
    vv2=z*x2-x5+(z-1)*r
    original=sp.Matrix([
        x2,x3,D1-a*x3-c*x2+b*sp.cos(theta1)*(x6-x3)-b*sp.sin(theta1)*(x5-x2)**2+d*sp.cos(theta1)*(x5-x2)+sp.sin(theta1),
        x5,x6,D2-a*x6-c*x5+b*sp.cos(theta2)*(z*x3-x6)-b*sp.sin(theta2)*vv2**2+d*sp.cos(theta2)*vv2+sp.sin(theta2)
    ])
    # d/dt of y = d/dt of x + [r,0,0,r,0,0].
    transformed=original+sp.Matrix([r,0,0,r,0,0])
    return all(sp.simplify(v)==0 for v in transformed-reduced)


def validate(samples=40, seed=20260929, tolerance=1e-7):
    rng=np.random.default_rng(seed)
    errors=[]; jac_errors=[]; roundtrip=[]
    for _ in range(samples):
        p=Parameters(a=float(rng.uniform(.1,2)), b=float(rng.uniform(.05,1)),
            c=float(rng.uniform(.1,2)),d=float(rng.uniform(.05,1)),
            delta1=float(rng.uniform(-1,1)),delta2=float(rng.uniform(-1,1)),
            zeta=float(rng.uniform(0,1.5)),omega=float(rng.uniform(.1,2)),
            W=float(rng.uniform(.5,3)))
        x=rng.normal(size=7); y=to_rotating(0,x,p)
        fx=rhs(0,x,p)
        fy=rotating_rhs(0,y,p)
        errors.append(float(np.max(np.abs(fy-(fx[:6]+np.array([p.omega/p.W,0,0,p.omega/p.W,0,0]))))))
        roundtrip.append(float(np.max(np.abs(to_original(y,x[6],p)-x))))
        # Transformation state derivative wrt x at fixed clock is identity,
        # so reduced Jacobian is the upper-left 6x6 block of original Jacobian.
        h=1e-6; jfd=np.zeros((6,6))
        for i in range(6):
            yp=y.copy();ym=y.copy();yp[i]+=h;ym[i]-=h
            jfd[:,i]=(rotating_rhs(0,yp,p)-rotating_rhs(0,ym,p))/(2*h)
        jac_errors.append(float(np.max(np.abs(jfd-jacobian(0,x,p)[:6,:6]))))
    p=Parameters(a=.8,b=.2,c=.5,d=.3,delta1=.1,delta2=-.1,zeta=.7,omega=1.,W=2.)
    x0=np.array([.1,0,0,-.1,0,0,0.])
    y0=to_rotating(0,x0,p)
    t_eval=np.linspace(0,10,201)
    original=solve_ivp(lambda t,x:rhs(t,x,p),(0,10),x0,t_eval=t_eval,method='DOP853',rtol=1e-11,atol=1e-13)
    reduced=solve_ivp(lambda t,y:rotating_rhs(t,y,p),(0,10),y0,t_eval=t_eval,method='DOP853',rtol=1e-11,atol=1e-13)
    traj_err=max(float(np.max(np.abs(to_rotating(t,original.y[:,i],p)-reduced.y[:,i]))) for i,t in enumerate(t_eval))
    proof=symbolic_proof()
    ok=proof and original.success and reduced.success and max(errors)<tolerance and max(roundtrip)<tolerance and max(jac_errors)<tolerance and traj_err<tolerance
    return {'status':'PASS' if ok else 'FAIL','scope':'Exact rotating-frame reduction of transcribed Eq. (4); NOT reference reproduction',
        'symbolic_identity':proof,'samples':samples,'seed':seed,'tolerance':tolerance,
        'max_rhs_error':max(errors),'max_jacobian_error':max(jac_errors),
        'max_inverse_transform_error':max(roundtrip),'max_trajectory_error':traj_err,
        'reference_bifurcations_reproduced':False,
        'caveats':['Reference dimensional-to-normalized coefficient definitions are not supplied in the PDF.',
        'The transformation changes delta_i to delta_i+c*omega/W.',
        'The original x7 remains an unbounded clock; only the co-rotating physical subsystem is autonomous.',
        'The original coupling is asymmetric; the transformation does not make it symmetric.']}


if __name__=='__main__':
    report=validate()
    dest=ROOT/'results'/'model_validation'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'p01_rotating_frame_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print('Report:',dest/'p01_rotating_frame_validation.json')
    if report['status']!='PASS':sys.exit(1)

