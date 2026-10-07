#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    fig06_global_coupling_landscape.py

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
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def H(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--freeze-dir",
        default="results/author_reproducibility_manuscript_freeze_v2/manuscript_freeze",
    )
    ap.add_argument("--output-dir", default="figures/figure_06")
    a = ap.parse_args()

    p = Path(a.freeze_dir) / "figure_source_data/fig_global_coupling_regimes.csv"
    d = pd.read_csv(p)

    o = Path(a.output_dir)
    o.mkdir(parents=True, exist_ok=True)

    z = d["zeta"].to_numpy()
    n = d["n_ic"].to_numpy(dtype=float)
    P = d["P"].to_numpy(dtype=float) / n
    M = d["M"].to_numpy(dtype=float) / n
    A = d["ambiguous"].to_numpy(dtype=float) / n

    # P/M coexistence among the sampled initial conditions.
    # This is the fraction represented by the less frequent physical regime.
    coexistence_fraction = np.minimum(P, M)

    # Retain the temporal-variability screen as contextual information.
    screen_fraction = d["intermittency_screen_candidates"].to_numpy(dtype=float) / n

    fig, axs = plt.subplots(2, 1, figsize=(3.5, 5.55), sharex=True)

    # ------------------------------------------------------------------
    # A. Global P/M composition across coupling
    # Step plots are appropriate because each value is a fraction of a
    # finite set of eight deterministic initial conditions.
    # ------------------------------------------------------------------
    axs[0].step(z, P, where="mid", label="P")
    axs[0].step(z, M, where="mid", label="M")
    axs[0].step(z, A, where="mid", label="ambiguous")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].set_ylabel("Fraction of sampled ICs")
    axs[0].legend(frameon=False, ncol=3, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.015), borderaxespad=0.0, columnspacing=1.1, handlelength=2.0)
    axs[0].set_title(
        "Global P/M composition across coupling",
        fontsize=9,
        fontweight="bold",
        pad=24,
    )

    # ------------------------------------------------------------------
    # B. Coexistence of P and M across coupling
    # The minority-regime fraction is >0 exactly where both physical
    # orientations occur among the sampled ICs.
    # Temporal-variability screen candidates are shown only as a light contextual rug.
    # ------------------------------------------------------------------
    axs[1].step(
        z,
        coexistence_fraction,
        where="mid",
        lw=1.1,
        label="P/M coexistence",
    )

    screen = screen_fraction > 0
    if np.any(screen):
        axs[1].scatter(
            z[screen],
            np.full(np.count_nonzero(screen), 0.51),
            marker="|",
            s=24,
            linewidths=0.8,
            alpha=0.45,
            label="Temporal screen",
        )

    axs[1].set_ylim(-0.02, 0.55)
    axs[1].set_ylabel("Minority-regime fraction")
    axs[1].set_xlabel(r"Coupling strength $\zeta$")
    axs[1].legend(frameon=False, ncol=2, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.005), borderaxespad=0.0, columnspacing=1.4, handlelength=2.2)

    for i, ax in enumerate(axs):
        ax.text(
            -0.18,
            1.03,
            chr(65 + i),
            transform=ax.transAxes,
            fontweight="bold",
        )
        ax.grid(alpha=0.15)

    fig.tight_layout(h_pad=2.4)

    stem = o / "Figure_6_IEEE_single_column"
    for e in ["png", "pdf", "svg"]:
        fig.savefig(
            stem.with_suffix("." + e),
            dpi=600 if e == "png" else None,
            bbox_inches="tight",
        )

    plt.close(fig)

    manifest = {
        "input": str(p),
        "sha256": H(p),
        "figure": 6,
        "panel_A": "Fractions of sampled ICs classified as P, M, or ambiguous.",
        "panel_B": (
            "Minority-regime fraction min(P/n_ic, M/n_ic); values > 0 indicate "
            "sampled P/M coexistence. P65 temporal-variability screen candidates "
            "are shown only as contextual markers."
        ),
        "guardrail": (
            "P65 is a trajectory-sampled global coupling survey, not the complete "
            "global bifurcation structure. P/M coexistence is finite-time coexistence "
            "among the sampled initial conditions. P65 screen candidates are screening "
            "labels and are not by themselves evidence of intermittency."
        ),
    }
    (o / "figure_06_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

