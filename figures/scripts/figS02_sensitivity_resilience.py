#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    figS02_sensitivity_resilience.py

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
os.environ.setdefault("MPLBACKEND","Agg"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
W=3.5; DPI=600
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--freeze-dir",default="results/author_reproducibility_manuscript_freeze/manuscript_freeze")
 ap.add_argument("--output-dir",default="figures/supplementary_figure_S2")
 a=ap.parse_args(); F=Path(a.freeze_dir); O=Path(a.output_dir); O.mkdir(parents=True,exist_ok=True)
 P={"p61r":F/"tables/p61_robustness_summary.csv","p61a":F/"tables/p61_anchor_summary.csv",
    "p62a":F/"tables/p62_anchor_resilience.csv","p62b":F/"tables/p62_first_switch_brackets.csv"}
 for p in P.values():
  if not p.exists(): raise SystemExit(f"Missing frozen input: {p}")
 print("[Supplementary Figure S2] Loading frozen P61/P62 tables...",flush=True)
 r=pd.read_csv(P["p61r"]); a61=pd.read_csv(P["p61a"]); a62=pd.read_csv(P["p62a"]); b=pd.read_csv(P["p62b"])
 assert len(r)==5 and len(a61)==8 and len(a62)==8 and len(b)==35 and b.persistent.all()
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8.4,
  "axes.labelsize":7.8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,
  "axes.linewidth":.8,"lines.linewidth":1})
 fig=plt.figure(figsize=(W,7.5)); gs=fig.add_gridspec(3,1,height_ratios=[1.05,1.35,1.2],hspace=.62)

 ax=fig.add_subplot(gs[0]); ax.text(-.20,1.015,"A",transform=ax.transAxes,fontweight="bold",fontsize=9.5,ha="right",va="bottom",clip_on=False)
 ax.set_title("Finite-time outcome retention under controlled perturbations",fontweight="bold",pad=12)
 rr=r.copy()
 labs=[]
 for _,x in rr.iterrows():
  if x.arm=="baseline": labs.append("Baseline")
  elif x.arm=="ic": labs.append(f"IC\n{x.level:.0e}")
  else: labs.append(f"Noise\n{x.level:.0e}")
 ax.bar(range(len(rr)),rr.reproduction_fraction)
 ax.set_xticks(range(len(rr)),labs); ax.set_ylim(0,1.08); ax.set_ylabel("Retention fraction")
 ax.grid(axis="y",alpha=.2)
 for i,v in enumerate(rr.reproduction_fraction): ax.text(i,v+.025,f"{v:.2f}",ha="center",fontsize=6.7)

 ax=fig.add_subplot(gs[1]); ax.text(-.20,1.015,"B",transform=ax.transAxes,fontweight="bold",fontsize=9.5,ha="right",va="bottom",clip_on=False)
 ax.set_title("First detected switching radii are direction dependent",fontweight="bold",pad=12)
 aa=a62.copy(); aa["short"]=aa.anchor_id.str.replace("e30_","",regex=False).str.replace("e34_","",regex=False)
 y=np.arange(len(aa))
 ax.hlines(y,aa.min_persistent_midpoint,aa.max_persistent_midpoint,lw=1)
 ax.scatter(aa.median_persistent_midpoint,y,s=20,zorder=3)
 ax.set_xscale("log"); ax.set_yticks(y,aa.short); ax.invert_yaxis()
 ax.set_xlabel("First detected persistent-switch radius"); ax.grid(axis="x",alpha=.2)
 ax.text(.02,.02,"Range across six sampled directions per anchor",transform=ax.transAxes,fontsize=6.4)

 ax=fig.add_subplot(gs[2]); ax.text(-.20,1.015,"C",transform=ax.transAxes,fontweight="bold",fontsize=9.5,ha="right",va="bottom",clip_on=False)
 ax.set_title("P- and M-side sampled switching radii differ",fontweight="bold",pad=4)
 vals=[]
 for expected in ["P","M"]:
  q=b[b.expected==expected]
  vals.append(q.midpoint.values)
 bp=ax.boxplot(vals,tick_labels=["P side","M side"],showfliers=True,widths=.5)
 ax.set_yscale("log"); ax.set_ylabel("First detected switch radius"); ax.grid(axis="y",alpha=.2)
 meds=[np.median(v) for v in vals]
 ax.text(.04,.94,f"Median P = {meds[0]:.2e}",transform=ax.transAxes,va="top",fontsize=6.6)
 ax.text(.04,.84,f"Median M = {meds[1]:.2e}",transform=ax.transAxes,va="top",fontsize=6.6)
 ax.text(.04,.08,"Sampled-ray distances; not global basin-boundary distances",transform=ax.transAxes,fontsize=6.4)

 fig.subplots_adjust(left=.25,right=.97,top=.98,bottom=.065)
 stem=O/"Supplementary_Figure_S2_IEEE_single_column"; print("[Supplementary Figure S2] Saving PNG/PDF/SVG...",flush=True)
 fig.savefig(stem.with_suffix(".png"),dpi=DPI,bbox_inches="tight",pad_inches=.03)
 fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",pad_inches=.03)
 fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",pad_inches=.03); plt.close(fig)
 man={"figure":"Supplementary Figure S2","freeze_version":"P64-v2",
  "inputs":{k:{"path":str(v),"sha256":sha(v)} for k,v in P.items()},
  "checks":{"p61_levels":len(r),"p61_anchors":len(a61),"p62_anchors":len(a62),
            "p62_persistent_first_switches":len(b),
            "P_first_switches":int((b.expected=="P").sum()),"M_first_switches":int((b.expected=="M").sum())},
  "outputs":[str(stem.with_suffix(x)) for x in [".png",".pdf",".svg"]],
  "guardrail":"P61 noise is an uncalibrated numerical stress test, so no general physical noise-robustness claim is made. P62 radii are first-detected switching distances along sampled rays, not global distances to a basin boundary."}
 (O/"Supplementary_Figure_S2_manifest.json").write_text(json.dumps(man,indent=2)); print(json.dumps(man,indent=2),flush=True)
if __name__=="__main__": main()

