#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    figS01_temporal_switching_detail.py

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
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load_npz_required(p, required=None):
    z=np.load(p,allow_pickle=False)
    if required:
        miss=set(required)-set(z.files)
        if miss: raise ValueError(f"{p} missing {sorted(miss)}; has {z.files}")
    return z

def state_array(z):
    # P68 NPZ schema can differ; identify only explicit classification/state arrays.
    for k in ("orientation","classification","states","state","labels","classes"):
        if k in z.files: return k,z[k]
    raise ValueError(f"No explicit classification array in {z.files}")

def time_array(z,n):
    for k in ("t_center","time","t","centers","window_centers"):
        if k in z.files and len(z[k])==n: return z[k]
    return np.arange(n)

def normalize_state(x):
    out=[]
    for v in x:
        if isinstance(v,(bytes,np.bytes_)): v=v.decode()
        s=str(v)
        if s in ("P","M","A","ambiguous"): out.append("A" if s=="ambiguous" else s)
        elif s in ("2","2.0"): out.append("P")
        elif s in ("0","0.0","-1","-1.0"): out.append("M")
        elif s in ("1","1.0"): out.append("A")
        else: out.append(s)
    return np.array(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--freeze-dir",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--project-root",default=None)
    a=ap.parse_args()
    F=Path(a.freeze_dir).resolve(); O=Path(a.output_dir).resolve(); O.mkdir(parents=True,exist_ok=True)
    root=Path(a.project_root).resolve() if a.project_root else F.parents[2]

    comb=F/"figure_source_data/fig_temporal_switching_lyapunov.csv"
    if not comb.is_file(): raise FileNotFoundError(comb)
    D=pd.read_csv(comb)
    required={"site_id","zeta","role","verified_temporal_switching",
              "min_switches_across_windows","lle_mean","lle_std"}
    miss=required-set(D.columns)
    if miss: raise ValueError(f"P70 combined table missing {sorted(miss)}")
    sw="min_switches_across_windows"; lle="lle_mean"

    # Representative verified site: nearest to 0.944; the actual zeta comes from frozen data.
    V=D[D["verified_temporal_switching"].astype(str).str.lower().isin(["true","1"])].copy()
    if V.empty: raise ValueError("No verified temporal-switching site in P70 table")
    r=V.iloc[(V["zeta"]-0.944).abs().argsort()[:1]].iloc[0]
    site=str(r["site_id"]); zeta=float(r["zeta"])

    trace=root/"results/author_intermittency_verification_v2"/f"{site}_trace.npz"
    if not trace.is_file(): raise FileNotFoundError(trace)
    T=load_npz_required(trace,["t","q1","q2"])

    p68dir=root/"results/author_temporal_switching_verification"
    winfiles={w:p68dir/f"{site}_w{w}.npz" for w in (50,100,200)}
    for q in winfiles.values():
        if not q.is_file(): raise FileNotFoundError(q)

    # Load explicit P68 classification arrays. Fail rather than infer from q1/q2.
    win={}
    schemas={}
    for w,q in winfiles.items():
        z=load_npz_required(q)
        key,st=state_array(z); st=normalize_state(st)
        tt=time_array(z,len(st))
        win[w]=(tt,st); schemas[str(w)]={"files":list(z.files),"state_key":key}

    fig=plt.figure(figsize=(7.16,9.5),constrained_layout=True)
    gs=fig.add_gridspec(4,2,height_ratios=[1.15,.72,.9,.9])
    a1=fig.add_subplot(gs[0,:]); a2=fig.add_subplot(gs[1,:])
    a3=fig.add_subplot(gs[2,0]); a4=fig.add_subplot(gs[2,1]); a5=fig.add_subplot(gs[3,:])

    a1.plot(T["t"],T["q1"],lw=.55)
    a1.axhline(0,ls="--",lw=.6)
    a1.set(xlabel="Time",ylabel=r"$\dot{\phi}_1$")
    a1.set_title(f"Representative fixed-coupling detector dynamics ($\\zeta={zeta:.3f}$)",fontsize=9,fontweight="bold")

    ymap={"M":0,"A":1,"P":2}
    for w in (50,100,200):
        tt,st=win[w]
        yy=np.array([ymap.get(x,np.nan) for x in st],float)
        a2.step(tt,yy,where="mid",lw=1,label=f"W={w}")
    a2.set_yticks([0,1,2],["M","A","P"])
    a2.set(xlabel="Time",ylabel="Orientation")
    a2.set_title("Two-detector classification across window sizes",fontsize=9,fontweight="bold")
    a2.legend(frameon=False,ncol=3,fontsize=7)

    markers={"candidate":"o","control":"s"}
    role_labels={"candidate":"screen-selected","control":"comparison"}
    for role,d in D.groupby("role"):
        a3.scatter(d["zeta"],d[sw],s=25,marker=markers.get(role,"o"),label=role_labels.get(role,str(role)))
    a3.set(xlabel=r"Coupling $\zeta$",ylabel="Minimum P/M switch count\nacross W = 50, 100, 200")
    a3.set_title("Window-robust P/M switching across coupling",fontsize=9,fontweight="bold")
    a3.legend(frameon=False,fontsize=7)

    for role,d in D.groupby("role"):
        a4.errorbar(d["zeta"],d[lle],yerr=d["lle_std"],fmt=markers.get(role,"o"),
                    ms=4,capsize=2,label=role_labels.get(role,str(role)))
    a4.axhline(0,ls="--",lw=.7)
    a4.set(xlabel=r"Coupling $\zeta$",ylabel="Finite-time largest\nLyapunov exponent")
    a4.set_title("Positive finite-time Lyapunov exponents",fontsize=9,fontweight="bold")

    for role,d in D.groupby("role"):
        a5.scatter(d[sw],d[lle],s=26,marker=markers.get(role,"o"),label=role_labels.get(role,str(role)))
    a5.axhline(0,ls="--",lw=.7)
    a5.set(xlabel="Minimum P/M switch count across W = 50, 100, 200",
           ylabel="Finite-time largest Lyapunov exponent")
    a5.set_title("Switching frequency is not determined by LLE magnitude",fontsize=9,fontweight="bold")

    # Panel labels: use panel-specific positions to avoid titles and axis labels.
    panel_label_pos = {
        "A": (-0.055, 1.06),
        "B": (-0.055, 1.06),
        "C": (-0.12, 1.10),   # farther left/up: clears Panel C title
        "D": (-0.08, 1.10),
        "E": (-0.055, 1.08),  # above-left: clears the vertical y-axis label
    }
    for label,ax in zip("ABCDE",(a1,a2,a3,a4,a5)):
        x,y = panel_label_pos[label]
        ax.text(x,y,label,transform=ax.transAxes,fontweight="bold",fontsize=11,
                va="bottom",ha="right",clip_on=False)
        ax.grid(alpha=.2)

    outputs=[]
    for ext in ("png","pdf","svg"):
        q=O/f"Supplementary_Figure_S1_IEEE_double_column.{ext}"
        fig.savefig(q,dpi=600 if ext=="png" else None,bbox_inches="tight"); outputs.append(str(q))
    plt.close(fig)

    inputs=[comb,trace,*winfiles.values()]
    prov={"figure":"Supplementary Figure S1","format":"IEEE double-column","freeze_version":"P70-v1",
          "representative_site":site,"representative_zeta":zeta,
          "P68_npz_schemas":schemas,
           "visible_role_labels":{"candidate":"screen-selected","control":"comparison"},
          "inputs":{str(q):sha(q) for q in inputs},
          "outputs":outputs,
          "guardrail":"Finite-time P/M orientation switching plus positive finite-time LLE supports sensitive dependence consistent with chaos; it does not by itself establish chaotic itinerancy, attractor hopping, crisis-induced switching, or a classical intermittency type."}
    (O/"Supplementary_Figure_S1_provenance.json").write_text(json.dumps(prov,indent=2)+"\n")
    print(json.dumps(prov,indent=2))

if __name__=="__main__": main()


