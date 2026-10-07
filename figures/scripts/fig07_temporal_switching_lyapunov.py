#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    fig07_temporal_switching_lyapunov.py

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
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 3.5
DPI = 600

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", required=True)
    ap.add_argument("--output-dir", default="figures/figure_07")
    a = ap.parse_args()

    F = Path(a.freeze_dir).resolve()
    O = Path(a.output_dir).resolve()
    O.mkdir(parents=True, exist_ok=True)

    src = F / "figure_source_data" / "fig_temporal_switching_lyapunov.csv"
    if not src.is_file():
        raise FileNotFoundError(src)

    D = pd.read_csv(src)
    required = {
        "site_id", "zeta", "role", "verified_temporal_switching",
        "min_switches_across_windows", "lle_mean", "lle_std"
    }
    missing = required - set(D.columns)
    if missing:
        raise ValueError(f"Frozen source table missing columns: {sorted(missing)}")

    sw = "min_switches_across_windows"
    lle = "lle_mean"
    verified = as_bool(D["verified_temporal_switching"])

    # Frozen-result integrity checks from P70.
    assert len(D) == 14, f"Expected 14 tested sites, found {len(D)}"
    assert int(verified.sum()) == 11, f"Expected 11 verified switching sites, found {int(verified.sum())}"
    assert np.all(D[lle].to_numpy(float) > 0), "Expected all site-mean finite-time LLEs to be positive"

    D = D.sort_values("zeta").reset_index(drop=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": .8,
        "lines.linewidth": 1,
    })

    fig = plt.figure(figsize=(W, 7.15))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=.68)

    # A — one visual class for all tested sites; no candidate/control distinction.
    ax = fig.add_subplot(gs[0])
    ax.scatter(D["zeta"], D[sw], s=24, marker="o")
    ax.set_xlabel(r"Coupling strength $\zeta$")
    ax.set_ylabel("Minimum P/M switch count")
    ax.set_title("Robust fixed-coupling temporal switching", fontweight="bold", pad=5)
    ax.grid(alpha=.2)
    ax.text(-.16, 1.06, "A", transform=ax.transAxes, fontweight="bold",
            fontsize=9.5, ha="right", va="bottom", clip_on=False)

    # B — retain uncertainty from the frozen P69 summary.
    ax = fig.add_subplot(gs[1])
    ax.errorbar(D["zeta"], D[lle], yerr=D["lle_std"],
                fmt="o", ms=4, capsize=2, linestyle="none")
    ax.axhline(0, ls="--", lw=.7)
    ax.set_xlabel(r"Coupling strength $\zeta$")
    ax.set_ylabel("Finite-time LLE")
    ax.set_title("Positive finite-time Lyapunov exponents", fontweight="bold", pad=5)
    ax.grid(alpha=.2)
    ax.text(-.16, 1.06, "B", transform=ax.transAxes, fontweight="bold",
            fontsize=9.5, ha="right", va="bottom", clip_on=False)

    # C — descriptive association only; no fitted trend is imposed.
    ax = fig.add_subplot(gs[2])
    ax.scatter(D[sw], D[lle], s=24, marker="o")
    ax.axhline(0, ls="--", lw=.7)
    ax.set_xlabel("Minimum P/M switch count")
    ax.set_ylabel("Finite-time LLE")
    ax.set_title("LLE magnitude does not determine switch count",
                 fontweight="bold", pad=7)
    ax.grid(alpha=.2)
    ax.text(-.16, 1.06, "C", transform=ax.transAxes, fontweight="bold",
            fontsize=9.5, ha="right", va="bottom", clip_on=False)

    fig.subplots_adjust(left=.22, right=.97, top=.975, bottom=.065)

    stem = O / "Figure_7_IEEE_single_column"
    outputs = []
    for ext in (".png", ".pdf", ".svg"):
        q = stem.with_suffix(ext)
        fig.savefig(q, dpi=DPI if ext == ".png" else None,
                    bbox_inches="tight", pad_inches=.03)
        outputs.append(str(q))
    plt.close(fig)

    manifest = {
        "figure": "Figure 7",
        "format": "IEEE single-column",
        "freeze_version": "P70-v1",
        "input": {"path": str(src), "sha256": sha(src)},
        "checks": {
            "tested_sites": int(len(D)),
            "verified_temporal_switching_sites": int(verified.sum()),
            "positive_site_mean_finite_time_LLE": int((D[lle] > 0).sum()),
        },
        "visible_grouping": (
            "All tested sites use one marker style. The frozen candidate/control "
            "screening role is retained in the source table but is not displayed "
            "because it is not a physical dynamical classification."
        ),
        "outputs": outputs,
        "guardrail": (
            "Robust finite-time P/M switching occurs in a positive finite-time-LLE "
            "region. Positive finite-time LLE supports sensitive dependence consistent "
            "with chaos, but does not by itself establish chaotic itinerancy, attractor "
            "hopping, crisis-induced switching, a classical intermittency type, or a "
            "mathematically proven asymptotic chaotic invariant set."
        ),
    }
    (O / "figure_07_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()

