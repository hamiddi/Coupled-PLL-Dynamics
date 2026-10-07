#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    fig01_model_validation_workflow.py

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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figure_style import IEEE_SINGLE_COLUMN_IN, RASTER_DPI, apply_ieee_style

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--freeze-dir", default="results/author_reproducibility_manuscript_freeze/manuscript_freeze")
    ap.add_argument("--output-dir", default="figures/figure_01")
    args=ap.parse_args()
    freeze=Path(args.freeze_dir)
    src=freeze/"figure_source_data"/"fig_model_validation.csv"
    if not src.exists():
        raise SystemExit(f"Missing frozen input: {src}")
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(src)

    def get(exp, validation):
        q=df[(df["experiment"]==exp)&(df["validation"]==validation)]
        if len(q)!=1:
            raise SystemExit(f"Expected exactly one row for {exp}: {validation}; found {len(q)}")
        return float(q.iloc[0]["value"])

    p00=get("P00","Analytic vs numerical Jacobian")
    p01=get("P01","Rotating-frame RHS equivalence")

    apply_ieee_style(plt)
    fig=plt.figure(figsize=(IEEE_SINGLE_COLUMN_IN,8.0))
    gs=fig.add_gridspec(4,1,height_ratios=[1.25,.85,1.25,1.25],hspace=.55)

    ax=fig.add_subplot(gs[0]); ax.set_axis_off()
    ax.text(0,1.03,"A",fontweight="bold",fontsize=9.5,transform=ax.transAxes)
    def bx(x,y,w,h,title,sub):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fill=False,lw=1))
        ax.text(x+w/2,y+h*.64,title,ha="center",fontweight="bold",fontsize=8.5)
        ax.text(x+w/2,y+h*.30,sub,ha="center",fontsize=7.3)
    bx(.06,.34,.32,.35,"PLL 1","third-order loop")
    bx(.62,.34,.32,.35,"PLL 2","third-order loop")
    ax.add_patch(FancyArrowPatch((.38,.58),(.62,.58),arrowstyle="-|>",mutation_scale=10,lw=1))
    ax.add_patch(FancyArrowPatch((.62,.43),(.38,.43),arrowstyle="-|>",mutation_scale=10,lw=1))
    ax.text(.5,.76,r"mutual coupling $\zeta$",ha="center",fontsize=8)
    ax.text(.5,.13,r"$\phi_1=\theta_2-\theta_1\qquad \phi_2=\zeta\theta_1-\theta_2$",ha="center",fontsize=9)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    ax=fig.add_subplot(gs[1]); ax.set_axis_off()
    ax.text(0,1.05,"B",fontweight="bold",fontsize=9.5,transform=ax.transAxes)
    ax.text(.5,.78,"Six-state co-rotating formulation",ha="center",fontweight="bold",fontsize=8.4)
    ax.text(.5,.49,r"$\mathbf{y}=(\theta_1,v_1,a_1,\theta_2,v_2,a_2)^{T}$",ha="center",fontsize=9)
    ax.text(.5,.23,"Explicit clock removed from the physical subsystem;\noriginal coupling asymmetry is retained.",
            ha="center",va="center",fontsize=7.3)

    ax=fig.add_subplot(gs[2])
    ax.text(-.14,1.06,"C",fontweight="bold",fontsize=9.5,transform=ax.transAxes)
    vals=[p00,p01]; labs=["P00\nJacobian","P01\nrotating-frame RHS"]
    ax.barh([0,1],vals,height=.48)
    ax.set_yticks([0,1],labs); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(1e-16,3e-7)
    ax.axvline(1e-7,ls="--",lw=1)
    ax.text(8e-8,1.46,r"$10^{-7}$ tolerance",ha="right",va="bottom",fontsize=7)
    for i,v in enumerate(vals):
        ax.text(v*1.6,i,f"{v:.2e}",va="center",fontsize=7.1)
    ax.set_xlabel("Maximum absolute error")
    ax.grid(axis="x",alpha=.22)
    ax.set_title("Frozen model-validation checks",pad=4,fontweight="bold")

    ax=fig.add_subplot(gs[3]); ax.set_axis_off()
    ax.text(0,1.04,"D",fontweight="bold",fontsize=9.5,transform=ax.transAxes)
    ax.text(.5,.98,"Computational study design",ha="center",va="top",fontweight="bold",fontsize=8.4)
    steps=["Validation","Local dynamics","Multistability","Regime selection","Transition dynamics","Resilience"]
    ys=[.82,.68,.54,.40,.26,.12]
    for i,(s,y) in enumerate(zip(steps,ys)):
        ax.add_patch(FancyBboxPatch((.25,y-.045),.5,.085,boxstyle="round,pad=0.008,rounding_size=.015",fill=False,lw=.9))
        ax.text(.5,y,s,ha="center",va="center",fontsize=7.6)
        if i<len(steps)-1:
            ax.add_patch(FancyArrowPatch((.5,y-.047),(.5,ys[i+1]+.047),arrowstyle="-|>",mutation_scale=8,lw=.8))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.subplots_adjust(left=.19,right=.97,top=.985,bottom=.055)

    stem=out/"Figure_1_IEEE_single_column"
    fig.savefig(stem.with_suffix(".png"),dpi=RASTER_DPI,bbox_inches="tight",pad_inches=.03)
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",pad_inches=.03)
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

    manifest={
        "figure":"Figure 1",
        "freeze_input":str(src),
        "freeze_input_sha256":sha256(src),
        "outputs":[str(stem.with_suffix(x)) for x in [".png",".pdf",".svg"]],
        "scientific_values_loaded":{"P00_jacobian_error":p00,"P01_rhs_error":p01},
        "note":"No scientific result values were manually entered into the plotting script."
    }
    (out/"figure_01_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
