#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    fig02_local_invariant_structure.py

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
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

from pathlib import Path
import argparse, hashlib, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IEEE_WIDTH=3.5
DPI=600

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--freeze-dir",default="results/author_reproducibility_manuscript_freeze/manuscript_freeze")
    ap.add_argument("--output-dir",default="figures/figure_02")
    a=ap.parse_args()
    freeze=Path(a.freeze_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    paths={k:freeze/"tables"/v for k,v in {
        "folds":"p20_folds.csv","orbits":"p21_orbits.csv","equilibria":"p22_equilibria.csv"}.items()}
    for p in paths.values():
        if not p.exists(): raise SystemExit(f"Missing frozen input: {p}")

    print("[Figure 2] Loading frozen P20/P21/P22 tables...", flush=True)
    folds=pd.read_csv(paths["folds"]); orbits=pd.read_csv(paths["orbits"]); eq=pd.read_csv(paths["equilibria"])
    assert len(folds)==2 and folds["genericity_screen_pass"].all()
    assert len(orbits)==9 and orbits["validated"].all()
    assert len(eq)==12 and (~eq["locally_asymptotically_stable"]).all()

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.0,"axes.titlesize":8.4,
        "axes.labelsize":7.8,"xtick.labelsize":7.0,"ytick.labelsize":7.0,
        "legend.fontsize":6.8,"axes.linewidth":0.8,"lines.linewidth":1.0})
    print("[Figure 2] Rendering...", flush=True)
    fig=plt.figure(figsize=(IEEE_WIDTH,5.65))
    gs=fig.add_gridspec(3,1,height_ratios=[0.70,1.05,1.12],hspace=.48)

    # A: numerically supported periodic-orbit folds
    ax=fig.add_subplot(gs[0])
    ax.text(-0.10,1.02,"A",ha="right",va="bottom",fontweight="bold",fontsize=9.2,transform=ax.transAxes,clip_on=False,zorder=20)
    ax.set_title("Periodic-orbit folds",fontweight="bold",pad=3)
    ax.set_xlim(float(folds.zeta.min())-.003,float(folds.zeta.max())+.003); ax.set_ylim(0,1)
    ax.set_yticks([])
    ax.hlines(.5,*ax.get_xlim(),lw=.8)
    for _,r in folds.iterrows():
        ax.axvline(r.zeta,ls="--",lw=.9)
        ax.plot(r.zeta,.5,"o",ms=5)
        ax.text(r.zeta,.72,fr"$\zeta={r.zeta:.6f}$",ha="center",fontsize=6.8,rotation=25)
    ax.set_xlabel(r"Coupling parameter $\zeta$")
    # Four sparse context ticks; fold values are already annotated above the markers.
    xa0, xa1 = ax.get_xlim()
    ax.set_xticks([0.890, 0.895, 0.900, 0.905])
    ax.set_xticklabels(["0.890", "0.895", "0.900", "0.905"])
    ax.tick_params(axis="x", labelsize=6.4)

    # B: periodic orbits
    ax=fig.add_subplot(gs[1])
    ax.text(-0.10,1.02,"B",ha="right",va="bottom",fontweight="bold",fontsize=9.2,transform=ax.transAxes,clip_on=False,zorder=20)
    ax.set_title("Unstable periodic-orbit families",fontweight="bold",pad=3)
    arcs=list(dict.fromkeys(orbits["arc"].astype(str)))
    markers=["o","s","^","D"]
    for i,arc in enumerate(arcs):
        q=orbits[orbits["arc"].astype(str)==arc].sort_values("zeta")
        ax.plot(q.zeta,q.period,marker=markers[i%len(markers)],label=f"arc {arc}")
    ax.set_xlabel(r"$\zeta$"); ax.set_ylabel("Period")
    ax.set_xticks([0.899, 0.901, 0.903])
    ax.set_xticklabels(["0.899", "0.901", "0.903"])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(frameon=False,ncol=1,loc="best")
    ax.grid(alpha=.2)

    # C: equilibria
    ax=fig.add_subplot(gs[2])
    ax.text(-0.10,1.02,"C",ha="right",va="bottom",fontweight="bold",fontsize=9.2,transform=ax.transAxes,clip_on=False,zorder=20)
    ax.set_title("Unstable sampled equilibria",fontweight="bold",pad=3)
    ids=list(dict.fromkeys(eq["id"].astype(str)))
    markers=["o","s","^","D"]
    for i,eid in enumerate(ids):
        q=eq[eq["id"].astype(str)==eid].sort_values("zeta")
        ax.plot(q.zeta,q.max_real_eigenvalue,marker=markers[i%len(markers)],label=eid)
    ax.axhline(0,ls="--",lw=.8)
    ax.set_xlabel(r"$\zeta$"); ax.set_ylabel(r"max Re$(\lambda)$")
    ax.set_xticks([0.899, 0.901, 0.903])
    ax.set_xticklabels(["0.899", "0.901", "0.903"])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(frameon=False,ncol=2,loc="best")
    ax.grid(alpha=.2)

    fig.subplots_adjust(left=.24,right=.97,top=.965,bottom=.075)
    stem=out/"Figure_2_IEEE_single_column"
    print("[Figure 2] Saving PNG/PDF/SVG...", flush=True)
    fig.savefig(stem.with_suffix(".png"),dpi=DPI,bbox_inches="tight",pad_inches=.03)
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",pad_inches=.03)
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

    manifest={"figure":"Figure 2","freeze_version":"P64-v2",
      "inputs":{k:{"path":str(p),"sha256":sha256(p)} for k,p in paths.items()},
      "checks":{"folds":len(folds),"validated_orbits":len(orbits),"unstable_equilibria":len(eq)},
      "outputs":[str(stem.with_suffix(x)) for x in [".png",".pdf",".svg"]],
      "presentation_version":"publication-clean-v1",
      "scientific_note":"All plotted scientific values remain loaded from the frozen P64-v2 tables; this revision changes presentation only.",
      "guardrail":"The periodic-orbit folds are numerically supported rather than a rigorous characterization of the complete global bifurcation structure; the validated periodic orbits and sampled equilibria shown are unstable."}
    (out/"figure_02_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2), flush=True)

if __name__=="__main__": main()

