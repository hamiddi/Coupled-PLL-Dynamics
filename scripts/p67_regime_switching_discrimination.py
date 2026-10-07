#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p67_regime_switching_discrimination.py

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
import os; os.environ.setdefault("MPLBACKEND","Agg"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def classify(d1,d2=None,z=.05):
 if d2 is None: return "proxy_P" if d1>z else ("proxy_M" if d1<-z else "proxy_A")
 return "P" if d1>z and d2<-z else ("M" if d1<-z and d2>z else "A")
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--p66-dir",default="results/author_intermittency_verification")
 ap.add_argument("--output-dir",default="results/author_regime_switching_discrimination")
 ap.add_argument("--window",type=float,default=100.); ap.add_argument("--step",type=float,default=20.); ap.add_argument("--threshold",type=float,default=.05)
 a=ap.parse_args(); S=Path(a.p66_dir); O=Path(a.output_dir); O.mkdir(parents=True,exist_ok=True)
 met=S/"p66_site_metrics.csv"
 if not met.is_file(): raise FileNotFoundError(met)
 rows=list(csv.DictReader(open(met,newline=""))); R=[]; strictn=0; proxyn=0
 for r in rows:
  if r.get("status")!="complete": continue
  f=S/f"{r['job_id']}_trace.npz"
  if not f.is_file(): continue
  q=np.load(f); t=np.asarray(q["t"]); q1=np.asarray(q["q1"]); q2=np.asarray(q["q2"]) if "q2" in q.files else None
  lam=np.asarray(q["laminar"],bool); dt=float(np.median(np.diff(t))); nw=max(2,round(a.window/dt)); ns=max(1,round(a.step/dt))
  strict=q2 is not None; strictn+=strict; proxyn+=not strict
  tt=[]; d1=[]; d2=[]; lab=[]; bf=[]
  for i in range(0,len(t)-nw+1,ns):
   sl=slice(i,i+nw); x=float(np.mean(q1[sl])); y=float(np.mean(q2[sl])) if strict else None
   tt.append(float(np.mean(t[sl]))); d1.append(x); d2.append(np.nan if y is None else y); lab.append(classify(x,y,a.threshold)); bf.append(float(np.mean(~lam[sl])))
  lab=np.asarray(lab); bf=np.asarray(bf); changes=int(np.sum(lab[1:]!=lab[:-1])); ix=np.flatnonzero(lab[1:]!=lab[:-1])+1
  near=np.zeros(len(lab),bool)
  for k in ix: near[max(0,k-1):min(len(lab),k+2)]=True
  u,c=np.unique(lab,return_counts=True)
  rec=dict(job_id=r["job_id"],ic_id=r["ic_id"],zeta=float(r["zeta"]),role=r["role"],classification_mode="strict_two_detector" if strict else "q1_proxy_screen",
   n_windows=len(lab),n_label_changes=changes,burst_fraction_near_changes=float(np.mean(bf[near])) if near.any() else np.nan,
   burst_fraction_away_changes=float(np.mean(bf[~near])) if (~near).any() else np.nan,dominant_label=str(u[np.argmax(c)]),dominant_fraction=float(c.max()/len(lab)))
  R.append(rec); np.savez_compressed(O/f"{r['job_id']}_windowed.npz",t=tt,drift1=d1,drift2=d2,labels=lab.astype("U16"),burst_fraction=bf)
 fields=list(R[0].keys())
 with open(O/"p67_site_summary.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(R)
 reps=[r for r in R if r["role"]=="candidate"]; rep=min(reps,key=lambda r:abs(r["zeta"]-.944))
 raw=np.load(S/f"{rep['job_id']}_trace.npz"); win=np.load(O/f"{rep['job_id']}_windowed.npz"); t=raw["t"]; q1=raw["q1"]; wt=win["t"]; labels=win["labels"]; mask=t<=t[0]+2500; wm=wt<=t[0]+2500
 fig=plt.figure(figsize=(7.16,6)); gs=fig.add_gridspec(3,1,height_ratios=[2,.48,1.1],hspace=.4)
 ax=fig.add_subplot(gs[0]); ax.plot(t[mask],q1[mask],lw=.5); ax.set_ylabel(r"$\dot{\phi}_1$"); ax.set_title(f"Fixed-parameter trace, zeta={rep['zeta']:.3f}",fontweight="bold"); ax.grid(alpha=.15)
 ax=fig.add_subplot(gs[1]); mp={"P":2,"M":-2,"A":0,"proxy_P":2,"proxy_M":-2,"proxy_A":0}; ax.step(wt[wm],[mp[x] for x in labels[wm]],where="mid"); ax.set_yticks([-2,0,2],["M","A","P"]); ax.set_xlabel("Time"); ax.set_title("Sliding-window orientation"+(" (two-detector)" if rep["classification_mode"]=="strict_two_detector" else " (q1 proxy only)"),fontsize=9)
 ax=fig.add_subplot(gs[2])
 for role,mark in [("candidate","o"),("control","s")]:
  rr=[r for r in R if r["role"]==role]; ax.scatter([x["zeta"] for x in rr],[x["n_label_changes"] for x in rr],marker=mark,label=role)
 ax.set_xlabel(r"Coupling strength $\zeta$"); ax.set_ylabel("Window-label changes"); ax.set_title("Temporal orientation changes across P66 sites",fontweight="bold"); ax.legend(frameon=False); ax.grid(alpha=.15)
 stem=O/"Figure_P67_regime_switching_discrimination"; fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight"); fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight"); fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight"); plt.close(fig)
 report={"scope":"P67 time-resolved discrimination using saved P66 traces","source_sha256":{"p66_site_metrics":sha(met)},"settings":{"window":a.window,"step":a.step,"threshold":a.threshold},"summary":{"sites":len(R),"strict_two_detector_sites":strictn,"q1_proxy_sites":proxyn},"guardrails":["Strict physical P/M classification requires both detector rates; q1-only results are screening proxies.","Sliding-window changes are finite-time events, not bifurcations.","Intermittency requires within-orientation laminar/burst evidence and further scaling if a classical type is claimed."]}
 (O/"p67_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()

