#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p68_temporal_switching_verification.py

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

def cls(d1,d2,z=.05):
 if d1>z and d2<-z:return "P"
 if d1<-z and d2>z:return "M"
 return "A"

def runs(labels,step):
 labels=np.asarray(labels)
 if not len(labels): return []
 st=np.r_[0,1+np.flatnonzero(labels[1:]!=labels[:-1])]
 en=np.r_[st[1:],len(labels)]
 return [{"label":str(labels[a]),"i0":int(a),"i1":int(b),"duration":float((b-a)*step)} for a,b in zip(st,en)]

def persistence_filter(labels,min_windows):
 x=np.asarray(labels).copy()
 # Iteratively absorb short interior runs into equal neighbors; otherwise mark A.
 for _ in range(4):
  rr=runs(x,1.0); changed=False
  for j,r in enumerate(rr):
   n=r["i1"]-r["i0"]
   if n>=min_windows: continue
   left=rr[j-1]["label"] if j>0 else None; right=rr[j+1]["label"] if j+1<len(rr) else None
   new=left if left==right and left in ("P","M") else "A"
   if np.any(x[r["i0"]:r["i1"]]!=new):
    x[r["i0"]:r["i1"]]=new; changed=True
  if not changed: break
 return x

def window_labels(t,q1,q2,window,step,z,min_residence):
 dt=float(np.median(np.diff(t))); nw=max(2,int(round(window/dt))); ns=max(1,int(round(step/dt)))
 tt=[]; lab=[]; d1=[]; d2=[]
 for i in range(0,len(t)-nw+1,ns):
  sl=slice(i,i+nw); a=float(np.mean(q1[sl])); b=float(np.mean(q2[sl]))
  tt.append(float(np.mean(t[sl]))); d1.append(a); d2.append(b); lab.append(cls(a,b,z))
 minw=max(1,int(np.ceil(min_residence/step)))
 lab=persistence_filter(np.array(lab,dtype="U1"),minw)
 return np.asarray(tt),np.asarray(d1),np.asarray(d2),lab

def summarize(tt,lab,step):
 rr=runs(lab,step)
 # Count direct P/M changes allowing an intervening A segment.
 core=[r for r in rr if r["label"] in ("P","M")]
 nsw=sum(core[i]["label"]!=core[i-1]["label"] for i in range(1,len(core)))
 pres=[r["duration"] for r in rr if r["label"]=="P"]; mres=[r["duration"] for r in rr if r["label"]=="M"]; ares=[r["duration"] for r in rr if r["label"]=="A"]
 thirds=[]
 edges=np.linspace(tt.min(),tt.max(),4)
 for k in range(3):
  m=(tt>=edges[k])&(tt<=(edges[k+1] if k==2 else edges[k+1]-1e-12))
  y=lab[m]; c=[v for v in y if v in ("P","M")]
  thirds.append(sum(c[i]!=c[i-1] for i in range(1,len(c))) if len(c)>1 else 0)
 return dict(n_switches=int(nsw),P_fraction=float(np.mean(lab=="P")),M_fraction=float(np.mean(lab=="M")),A_fraction=float(np.mean(lab=="A")),
  P_residence_median=float(np.median(pres)) if pres else 0.,M_residence_median=float(np.median(mres)) if mres else 0.,
  A_duration_median=float(np.median(ares)) if ares else 0.,early_switches=int(thirds[0]),middle_switches=int(thirds[1]),late_switches=int(thirds[2]))

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--p66-dir",default="results/author_intermittency_verification_v2")
 ap.add_argument("--output-dir",default="results/author_temporal_switching_verification")
 ap.add_argument("--windows",default="50,100,200"); ap.add_argument("--step",type=float,default=20.)
 ap.add_argument("--threshold",type=float,default=.05); ap.add_argument("--min-residence",type=float,default=60.)
 a=ap.parse_args(); S=Path(a.p66_dir); O=Path(a.output_dir); O.mkdir(parents=True,exist_ok=True)
 met=S/"p66_site_metrics.csv"
 if not met.is_file(): raise FileNotFoundError(met)
 rows=list(csv.DictReader(open(met,newline=""))); windows=[float(x) for x in a.windows.split(",")]
 recs=[]; missing_q2=[]
 for r in rows:
  if r.get("status")!="complete":continue
  f=S/f"{r['job_id']}_trace.npz"
  if not f.is_file():continue
  q=np.load(f)
  if "q2" not in q.files: missing_q2.append(str(f)); continue
  t=np.asarray(q["t"]); q1=np.asarray(q["q1"]); q2=np.asarray(q["q2"])
  for w in windows:
   tt,d1,d2,lab=window_labels(t,q1,q2,w,a.step,a.threshold,a.min_residence); sm=summarize(tt,lab,a.step)
   rec=dict(job_id=r["job_id"],ic_id=r["ic_id"],zeta=float(r["zeta"]),role=r["role"],window=w,**sm)
   recs.append(rec); np.savez_compressed(O/f"{r['job_id']}_w{int(w)}.npz",t=tt,drift1=d1,drift2=d2,labels=lab)
 if missing_q2: raise RuntimeError("q2 missing from P66-v2 traces: "+missing_q2[0])
 fields=list(recs[0].keys())
 with open(O/"p68_switching_metrics.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(recs)

 # Cross-window verification: >=2 switches at all three window sizes and switching in >=2 temporal thirds at W=100.
 byjob={}
 for r in recs:byjob.setdefault(r["job_id"],[]).append(r)
 ver=[]
 for jid,rr in byjob.items():
  r100=min(rr,key=lambda x:abs(x["window"]-100))
  ok=all(x["n_switches"]>=2 for x in rr) and sum(r100[k]>0 for k in ("early_switches","middle_switches","late_switches"))>=2
  ver.append(dict(job_id=jid,ic_id=r100["ic_id"],zeta=r100["zeta"],role=r100["role"],verified_temporal_switching=ok,
                  min_switches_across_windows=min(x["n_switches"] for x in rr),max_switches_across_windows=max(x["n_switches"] for x in rr),
                  early_switches=r100["early_switches"],middle_switches=r100["middle_switches"],late_switches=r100["late_switches"],
                  P_fraction=r100["P_fraction"],M_fraction=r100["M_fraction"],A_fraction=r100["A_fraction"],
                  P_residence_median=r100["P_residence_median"],M_residence_median=r100["M_residence_median"]))
 with open(O/"p68_site_verification.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(ver[0]));w.writeheader();w.writerows(ver)

 # Figure: representative zeta=.944 plus cross-window/site summaries.
 cand=[v for v in ver if v["role"]=="candidate"]; rep=min(cand,key=lambda x:abs(x["zeta"]-.944))
 raw=np.load(S/f"{rep['job_id']}_trace.npz"); t=raw["t"]; q1=raw["q1"]; mask=t<=t[0]+2500
 fig=plt.figure(figsize=(7.16,7.0)); gs=fig.add_gridspec(3,1,height_ratios=[1.8,1.15,1.15],hspace=.42)
 ax=fig.add_subplot(gs[0]); ax.plot(t[mask],q1[mask],lw=.45); ax.set_ylabel(r"$\dot{\phi}_1$"); ax.set_title(f"Fixed-coupling temporal switching, zeta={rep['zeta']:.3f}",fontweight="bold"); ax.grid(alpha=.15)
 ax=fig.add_subplot(gs[1])
 mp={"P":1,"A":0,"M":-1}
 for off,wv in zip([.12,0,-.12],windows):
  q=np.load(O/f"{rep['job_id']}_w{int(wv)}.npz"); m=q["t"]<=t[0]+2500
  ax.step(q["t"][m],[mp[x]+off for x in q["labels"][m]],where="mid",lw=.8,label=f"W={int(wv)}")
 ax.set_yticks([-1,0,1],["M","A","P"]);ax.set_xlabel("Time");ax.set_title("Two-detector classification across window sizes",fontweight="bold");ax.legend(ncol=3,frameon=False);ax.grid(alpha=.15)
 ax=fig.add_subplot(gs[2])
 for role,mark in [("candidate","o"),("control","s")]:
  rr=[x for x in ver if x["role"]==role];ax.scatter([x["zeta"] for x in rr],[x["min_switches_across_windows"] for x in rr],marker=mark,label=role)
 ax.set_xlabel(r"Coupling strength $\zeta$");ax.set_ylabel("Minimum switches\nacross W=50,100,200");ax.set_title("Robust fixed-coupling P/M switching",fontweight="bold");ax.legend(frameon=False);ax.grid(alpha=.15)
 stem=O/"Figure_P68_temporal_switching_verification";fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight");fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight");fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight");plt.close(fig)

 report={"scope":"P68 fixed-coupling temporal P/M switching verification from P66-v2 traces; no new ODE integrations",
  "source_sha256":{"p66_site_metrics":sha(met)},"settings":{"windows":windows,"step":a.step,"threshold":a.threshold,"min_residence":a.min_residence},
  "summary":{"sites":len(ver),"verified_sites":sum(v["verified_temporal_switching"] for v in ver),
             "candidate_sites":sum(v["role"]=="candidate" for v in ver),"candidate_verified":sum(v["role"]=="candidate" and v["verified_temporal_switching"] for v in ver),
             "control_sites":sum(v["role"]=="control" for v in ver),"control_verified":sum(v["role"]=="control" and v["verified_temporal_switching"] for v in ver)},
  "verification_rule":"At least two persistence-filtered P<->M switches at each of W=50,100,200 and at least one switch in at least two of early/middle/late thirds at W=100.",
  "guardrails":["Finite-time temporal orientation switching is not by itself proof of chaotic itinerancy, attractor hopping, intermittency type, or asymptotic invariant-set switching.",
                "Candidate/control labels originate from the P65 intermittency screen and are not assumed to identify temporal switching.",
                "Window-size and persistence checks test classification robustness, not asymptotic dynamics."]}
 (O/"p68_report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()

