#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    make_all_figures.py

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
from pathlib import Path
import os, subprocess, sys

root = Path(__file__).resolve().parents[2]
scripts = Path(__file__).resolve().parent
p64 = root / "results/author_reproducibility_manuscript_freeze/manuscript_freeze"
p70 = root / "results/author_reproducibility_manuscript_freeze_v2/manuscript_freeze"

jobs = [
    ("fig01_model_validation_workflow.py", "figure_01", p64),
    ("fig02_local_invariant_structure.py", "figure_02", p64),
    ("fig03_pm_phase_slipping.py", "figure_03", p64),
    ("fig04_regime_selection_geometry.py", "figure_04", p64),
    ("fig05_transition_dynamics.py", "figure_05", p64),
    ("fig06_global_coupling_landscape.py", "figure_06", p70),
    ("fig07_temporal_switching_lyapunov.py", "figure_07", p70),
]
for freeze in (p64, p70):
    if not freeze.is_dir():
        raise FileNotFoundError(f"Required freeze directory not found: {freeze}")

env = os.environ.copy()
for script, out_name, freeze in jobs:
    out = root / "figures/output" / out_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[figures] {script}  freeze={freeze}", flush=True)
    subprocess.run([sys.executable, str(scripts/script), "--freeze-dir", str(freeze),
                    "--output-dir", str(out)], check=True, env=env)
print("[figures] all seven manuscript figures completed.", flush=True)
