#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    fig04_regime_selection_geometry.py

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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path
import argparse, hashlib, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 3.5
DPI = 600
ZETA0 = 0.899

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="results/author_reproducibility_manuscript_freeze/manuscript_freeze")
    ap.add_argument("--output-dir", default="figures/figure_04")
    a = ap.parse_args()
    F, O = Path(a.freeze_dir), Path(a.output_dir)
    O.mkdir(parents=True, exist_ok=True)

    p50 = F / "tables/p50_transition_brackets.csv"
    p57 = F / "tables/p57_refined_thresholds.csv"
    p60 = F / "tables/p60_transition_brackets.csv"
    for p in (p50, p57, p60):
        if not p.is_file():
            raise FileNotFoundError(p)

    A, B, C = map(pd.read_csv, (p50, p57, p60))

    req50 = {"edge_index", "row_index", "angle_a", "angle_b", "resolved"}
    req57 = {"ic_id", "zeta_midpoint", "both_endpoints_persistent"}
    req60 = {"ic_id", "zeta_left", "zeta_right", "orientation_left", "orientation_right", "persistent"}
    for name, df, req in [("P50", A, req50), ("P57", B, req57), ("P60", C, req60)]:
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    A = A[as_bool(A["resolved"])].copy()
    B = B[as_bool(B["both_endpoints_persistent"])].copy()
    C = C[as_bool(C["persistent"])].copy()
    assert len(A) == 42, f"Expected 42 resolved P50 transitions, found {len(A)}"
    assert len(B) == 23, f"Expected 23 persistent P57 thresholds, found {len(B)}"
    assert len(C) == 78, f"Expected 78 persistent P60 transitions, found {len(C)}"

    A["angle_mid"] = 0.5 * (A["angle_a"] + A["angle_b"])
    C["zeta_midpoint"] = 0.5 * (C["zeta_left"] + C["zeta_right"])
    B["zeta_scaled"] = (B["zeta_midpoint"] - ZETA0) * 1e6
    C["zeta_scaled"] = (C["zeta_midpoint"] - ZETA0) * 1e6

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 7.8,
        "xtick.labelsize": 7, "ytick.labelsize": 6.6,
        "legend.fontsize": 6.6, "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
    })

    fig = plt.figure(figsize=(W, 7.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.15, 1.45], hspace=0.62)

    # A — retain all eight sampled local IC cuts rather than collapsing by edge.
    ax = fig.add_subplot(gs[0])
    cuts = sorted(A[["edge_index", "row_index"]].drop_duplicates().itertuples(index=False, name=None))
    ypos = {cut: i for i, cut in enumerate(cuts)}
    for edge, marker in [(30, "o"), (34, "s")]:
        d = A[A["edge_index"] == edge]
        ax.scatter(d["angle_mid"], [ypos[(int(e), int(r))] for e, r in zip(d.edge_index, d.row_index)],
                   s=18, marker=marker, label=f"Edge {edge}")
    labels = [f"E{int(e)}–IC{int(r)+1}" for e, r in cuts]
    ax.set_yticks(range(len(cuts)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Local angular coordinate")
    ax.set_ylabel("Local IC cut")
    ax.set_title("Multibanded local regime selection", fontweight="bold", pad=4)
    ax.grid(axis="x", alpha=0.20)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(-0.17, 1.05, "A", transform=ax.transAxes, fontweight="bold", fontsize=9.5)

    # B — sort only for visual display; scaled coordinate avoids Matplotlib offset notation.
    ax = fig.add_subplot(gs[1])
    q = B.sort_values("zeta_midpoint").reset_index(drop=True)
    y = range(1, len(q) + 1)
    ax.scatter(q["zeta_scaled"], y, s=18)
    ax.set_xlabel(r"$10^6(\zeta-0.899)$")
    ax.set_ylabel("Initial condition (sorted)")
    ax.set_yticks([])
    ax.set_title("IC-dependent switching thresholds", fontweight="bold", pad=4)
    ax.grid(axis="x", alpha=0.20)
    ax.text(-0.17, 1.05, "B", transform=ax.transAxes, fontweight="bold", fontsize=9.5)

    # C — same scaled coupling coordinate as B, preserving true cross-IC positions.
    ax = fig.add_subplot(gs[2])
    ic_ids = sorted(C["ic_id"].unique())
    ic_y = {ic: i + 1 for i, ic in enumerate(ic_ids)}
    for (left, right), marker in [(('P', 'M'), '>'), (('M', 'P'), '<')]:
        d = C[(C.orientation_left == left) & (C.orientation_right == right)]
        ax.scatter(d["zeta_scaled"], [ic_y[x] for x in d["ic_id"]],
                   s=22, marker=marker, label=f"{left}→{right}")
    ax.set_yticks(range(1, len(ic_ids) + 1), [f"IC{i}" for i in range(1, len(ic_ids) + 1)])
    ax.invert_yaxis()
    ax.set_xlabel(r"$10^6(\zeta-0.899)$")
    ax.set_ylabel("Initial condition")
    ax.set_title("Re-entrant coupling transitions", fontweight="bold", pad=4)
    ax.grid(axis="x", alpha=0.20)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(-0.17, 1.05, "C", transform=ax.transAxes, fontweight="bold", fontsize=9.5)

    fig.subplots_adjust(left=0.23, right=0.97, top=0.98, bottom=0.065)
    stem = O / "Figure_4_IEEE_single_column"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=DPI if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    prov = {
        "figure": "Figure 4", "freeze_version": "P64-v2",
        "inputs": {p.name: {"path": str(p), "sha256": sha(p)} for p in (p50, p57, p60)},
        "rows_used": {"P50_resolved": len(A), "P57_persistent": len(B), "P60_persistent": len(C)},
        "display": {
            "P50_coordinate": "native local angular coordinate",
            "P57_P60_coordinate": "1e6*(zeta-0.899)",
            "P50_local_cuts": len(cuts), "P60_initial_conditions": len(ic_ids)
        },
        "guardrail": "Finite-time transition bands/thresholds are not automatically bifurcations and do not establish a fractal basin boundary."
    }
    (O / "Figure_4_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov, indent=2))

if __name__ == "__main__":
    main()

