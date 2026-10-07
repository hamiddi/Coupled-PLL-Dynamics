#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p00_model_validation.py

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

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Parameters:
    """Dimensionless Eq. (4) coefficients; omega and W share units."""
    a: float
    b: float
    c: float
    d: float
    delta1: float
    delta2: float
    zeta: float
    omega: float
    W: float

    def validate(self) -> None:
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All parameters must be finite.")
        if self.W == 0:
            raise ValueError("W must be nonzero.")


# EXPLICITLY ILLUSTRATIVE: these are not claimed to match the source paper.
ILLUSTRATIVE = Parameters(a=0.8, b=0.2, c=0.5, d=0.3,
                          delta1=0.1, delta2=-0.1, zeta=0.7,
                          omega=1.0, W=2.0)


def symbolic_model():
    """Return symbols, seven-state vector field, and exact 7x7 Jacobian."""
    x = sp.symbols('x1:8', real=True)
    x1, x2, x3, x4, x5, x6, x7 = x
    a, b, c, d, delta1, delta2, zeta, omega, W = sp.symbols(
        'a b c d delta1 delta2 zeta omega W', real=True)
    q = omega / W * (zeta - 1)
    theta1 = x4 - x1
    theta2 = zeta * x1 - x4 + q * x7
    v1, acc1 = x5 - x2, x6 - x3
    v2, acc2 = zeta * x2 - x5 + q, zeta * x3 - x6
    f = sp.Matrix([
        x2,
        x3,
        delta1 - a*x3 - c*x2 + b*sp.cos(theta1)*acc1
        - b*sp.sin(theta1)*v1**2 + d*sp.cos(theta1)*v1
        + sp.sin(theta1),
        x5,
        x6,
        delta2 - a*x6 - c*x5 + b*sp.cos(theta2)*acc2
        - b*sp.sin(theta2)*v2**2 + d*sp.cos(theta2)*v2
        + sp.sin(theta2),
        sp.Integer(1),
    ])
    p = (a, b, c, d, delta1, delta2, zeta, omega, W)
    return x, p, f, f.jacobian(x)


X_SYMBOLS, P_SYMBOLS, F_SYMBOLIC, J_SYMBOLIC = symbolic_model()
_F = sp.lambdify((*X_SYMBOLS, *P_SYMBOLS), F_SYMBOLIC, 'numpy', cse=True)
_J = sp.lambdify((*X_SYMBOLS, *P_SYMBOLS), J_SYMBOLIC, 'numpy', cse=True)


def _arguments(y: np.ndarray, p: Parameters) -> tuple:
    p.validate()
    y = np.asarray(y, dtype=float)
    if y.shape != (7,) or not np.all(np.isfinite(y)):
        raise ValueError("State must be a finite seven-element vector.")
    return (*y, *asdict(p).values())


def rhs(t: float, y: np.ndarray, p: Parameters) -> np.ndarray:
    """Seven-dimensional autonomous extension; x7 is normalized time."""
    del t  # time enters via the state x7, not the solver's time argument
    return np.asarray(_F(*_arguments(y, p)), dtype=float).reshape(7)


def jacobian(t: float, y: np.ndarray, p: Parameters) -> np.ndarray:
    del t
    return np.asarray(_J(*_arguments(y, p)), dtype=float).reshape(7, 7)


def finite_difference_jacobian(y: np.ndarray, p: Parameters,
                               step: float = 1e-6) -> np.ndarray:
    """Central differences, scaled separately for each state coordinate."""
    result = np.empty((7, 7), dtype=float)
    for j in range(7):
        h = step * max(1.0, abs(float(y[j])))
        yp, ym = y.copy(), y.copy()
        yp[j] += h
        ym[j] -= h
        result[:, j] = (rhs(0, yp, p) - rhs(0, ym, p)) / (2*h)
    return result


def run_validation(samples: int = 12, seed: int = 20260929,
                   tolerance: float = 1e-7) -> dict:
    """Test multiple arbitrary illustrative parameter/state combinations."""
    rng = np.random.default_rng(seed)
    errors = []
    time_dependence_errors = []
    for i in range(samples):
        p = Parameters(a=float(rng.uniform(.2, 2)),
                       b=float(rng.uniform(.05, 1)),
                       c=float(rng.uniform(.1, 2)),
                       d=float(rng.uniform(.05, 1)),
                       delta1=float(rng.uniform(-1, 1)),
                       delta2=float(rng.uniform(-1, 1)),
                       zeta=float(rng.uniform(.1, .9)),
                       omega=float(rng.uniform(.2, 2)),
                       W=float(rng.uniform(.5, 3)))
        y = rng.normal(size=7)
        exact = jacobian(0, y, p)
        approx = finite_difference_jacobian(y, p)
        errors.append(float(np.max(np.abs(exact - approx))))
        # d(f_1,...,f_6)/d(x7) equals the last Jacobian column.
        h = 1e-6
        yp, ym = y.copy(), y.copy()
        yp[6] += h
        ym[6] -= h
        observed = (rhs(0, yp, p)[:6] - rhs(0, ym, p)[:6])/(2*h)
        time_dependence_errors.append(float(np.max(np.abs(observed-exact[:6, 6]))))
    # At zeta=1 the explicit x7 dependence vanishes, independent of state.
    p_one = Parameters(**{**asdict(ILLUSTRATIVE), 'zeta': 1.0})
    one_coupling_error = float(np.max(np.abs(jacobian(
        0, np.array([.1, .2, .3, .4, .5, .6, .7]), p_one)[:6, 6])))
    # x7'=1 means no equilibria of the seven-dimensional extended system.
    seventh_derivative_error = float(abs(rhs(0, np.zeros(7), ILLUSTRATIVE)[6]-1))
    max_error = max(errors)
    passed = (max_error < tolerance and
              max(time_dependence_errors) < tolerance and
              one_coupling_error < tolerance and
              seventh_derivative_error == 0)
    return {
        'status': 'PASS' if passed else 'FAIL',
        'scope': 'Equation transcription internal consistency; not reference reproduction',
        'samples': samples, 'seed': seed, 'tolerance': tolerance,
        'max_jacobian_absolute_error': max_error,
        'max_time_dependence_column_error': max(time_dependence_errors),
        'zeta_one_time_dependence_error': one_coupling_error,
        'seventh_derivative_error': seventh_derivative_error,
        'seven_state_equilibria_exist': False,
        'six_state_autonomous_when': '(zeta-1)*omega/W = 0',
        'reference_bifurcations_reproduced': False,
        'limitations': [
            'Original dimensional equations and full reference parameter set must be independently verified.',
            'This checks a transcription of the supplied paper equation, not circuit-level derivation.',
            'No Hopf or chaotic-attractor classification is attempted.',
            'For general coupling, x7 is an unbounded clock state; standard attractor and equilibrium '
            'calculations require a justified reduction or nonautonomous framework.'
        ]
    }


def illustrative_trajectory(output_dir: Path) -> dict:
    """Optional solver smoke test, explicitly NOT a reference reproduction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y0 = np.array([.1, 0, 0, -.1, 0, 0, 0.0])
    t_eval = np.linspace(0, 40, 2001)
    sol = solve_ivp(lambda t, y: rhs(t, y, ILLUSTRATIVE),
                    (0, 40), y0, t_eval=t_eval, method='DOP853',
                    rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir/'illustrative_trajectory.csv',
               np.column_stack([sol.t, sol.y.T]), delimiter=',',
               header='t,x1,x2,x3,x4,x5,x6,x7', comments='')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sol.t, sol.y[0], label='x1')
    ax.plot(sol.t, sol.y[3], label='x4')
    ax.set(xlabel='Normalized time', ylabel='State',
           title='Illustrative parameters — NOT a reference reproduction')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir/'illustrative_trajectory.png', dpi=600)
    plt.close(fig)
    return {'solver_success': sol.success,
            'max_clock_error': float(np.max(np.abs(sol.y[6]-sol.t))),
            'parameters': asdict(ILLUSTRATIVE)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/model_validation'))
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--illustrative-trajectory', action='store_true')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be at least 1')
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT/args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_validation(args.samples, args.seed)
    (output_dir/'symbolic_equations.txt').write_text(
        'Source: supplied Harb et al. paper, p. 2, Eq. (4)\n'
        'x7 is normalized time; x7_prime = 1\n\n'
        + '\n'.join(f'd{x}/dt = {sp.sstr(expr)}' for x, expr in zip(X_SYMBOLS, F_SYMBOLIC))
        + '\n', encoding='utf-8')
    (output_dir/'symbolic_jacobian.txt').write_text(
        'Exact 7x7 Jacobian of Eq. (4):\n'+str(J_SYMBOLIC)+'\n', encoding='utf-8')
    if args.illustrative_trajectory:
        report['illustrative_solver_test'] = illustrative_trajectory(output_dir)
    (output_dir/'validation_report.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Files written to: {output_dir}')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())

