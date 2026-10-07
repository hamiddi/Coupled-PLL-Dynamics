#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p69_lyapunov_temporal_switching.py

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
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def rows(p): return list(csv.DictReader(open(p,newline="")))
def state(row): return np.array([float(row[f"x{k}"]) for k in range(1,7)])
def lle(x0,p,transient,duration,renorm,dt,eps,rtol,atol,maxabs,seed):
 # settle reference trajectory
 x=integrate(x0,p,transient,dt,rtol,atol,maxabs)[1][-1]
 rng=np.random.default_rng(seed); v=rng.normal(size=6); v/=np.linalg.norm(v); y=x+eps*v
 n=int(np.floor(duration/renorm)); logs=[]
 for _ in range(n):
  X=integrate(x,p,renorm,dt,rtol,atol,maxabs)[1][-1]
  Y=integrate(y,p,renorm,dt,rtol,atol,maxabs)[1][-1]
  d=Y-X; nd=float(np.linalg.norm(d))
  if not np.isfinite(nd) or nd<=0: raise RuntimeError("invalid perturbation norm")
  logs.append(np.log(nd/eps)); y=X+eps*d/nd; x=X
 return float(np.sum(logs)/(n*renorm)),np.asarray(logs)/renorm

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--p55-fixed-ics",default="results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv")
 ap.add_argument("--p68-verification",default="results/author_temporal_switching_verification/p68_site_verification.csv")
 ap.add_argument("--output-dir",default="results/author_lyapunov_temporal_switching")
 ap.add_argument("--transient",type=float,default=1600.);ap.add_argument("--duration",type=float,default=6400.)
 ap.add_argument("--renorm",type=float,default=2.0);ap.add_argument("--dt",type=float,default=.25);ap.add_argument("--epsilon",type=float,default=1e-8)
 ap.add_argument("--rtol",type=float,default=1e-11);ap.add_argument("--atol",type=float,default=1e-13);ap.add_argument("--max-abs",type=float,default=1e7)
 ap.add_argument("--seeds",default="17,29,43");ap.add_argument("--resume",action="store_true");ap.add_argument("--diagnostic-only",action="store_true")
 a=ap.parse_args(); I=Path(a.p55_fixed_ics); V=Path(a.p68_verification);O=Path(a.output_dir);O.mkdir(parents=True,exist_ok=True)
 if not I.is_file() or not V.is_file():raise FileNotFoundError("Missing P55 IC or P68 verification table")
 ic={r["ic_id"]:r for r in rows(I)}; vr=rows(V); seeds=[int(x) for x in a.seeds.split(",")]
 # Use all P68 sites, allowing direct switching-frequency/LLE comparison.
 jobs=[(r,s) for r in vr for s in seeds]
 cfg={"source_sha256":{"p55":sha(I),"p68":sha(V)},"settings":{k:v for k,v in vars(a).items() if k not in ("resume","diagnostic_only","output_dir")},"jobs":len(jobs)}
 cf=O/"p69_config.json";cp=O/"p69_checkpoint.json"
 if a.resume and cf.exists() and json.loads(cf.read_text())!=cfg:raise ValueError("Resume config mismatch")
 if not a.resume and cp.exists():raise ValueError("Existing checkpoint; use --resume or another output dir")
 cf.write_text(json.dumps(cfg,indent=2)+"\n")
 if a.diagnostic_only:
  print(json.dumps({"status":"PASS","sites":len(vr),"seeds":seeds,"jobs":len(jobs)},indent=2));return
 done=json.loads(cp.read_text()) if a.resume and cp.exists() else []; keys={x["job_id"] for x in done}
 for j,(r,sd) in enumerate(jobs,1):
  jid=f"{r['job_id']}_seed{sd}"
  if jid in keys:continue
  rec={"job_id":jid,"site_id":r["job_id"],"ic_id":r["ic_id"],"zeta":float(r["zeta"]),"role":r["role"],
       "verified_temporal_switching":str(r["verified_temporal_switching"]).lower()=="true",
       "switch_count_min":int(r["min_switches_across_windows"]),"seed":sd,"status":"failed"}
  try:
   p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=rec["zeta"],omega=1.,W=100.)
   val,local=lle(state(ic[rec["ic_id"]]),p,a.transient,a.duration,a.renorm,a.dt,a.epsilon,a.rtol,a.atol,a.max_abs,sd)
   rec.update(status="complete",lle=val,local_lle_mean=float(np.mean(local)),local_lle_std=float(np.std(local)),
              positive_fraction=float(np.mean(local>0)))
  except Exception as e:rec["error"]=repr(e)
  done.append(rec);keys.add(jid);cp.write_text(json.dumps(done,indent=2)+"\n")
  print(f"P69 {j:03d}/{len(jobs)} zeta={rec['zeta']:.6f} seed={sd} LLE={rec.get('lle')} status={rec['status']}",flush=True)
 fields=["job_id","site_id","ic_id","zeta","role","verified_temporal_switching","switch_count_min","seed","status","lle","local_lle_mean","local_lle_std","positive_fraction","error"]
 with open(O/"p69_lle_runs.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(done)
 good=[r for r in done if r["status"]=="complete"]; site=[]
 for sid in sorted(set(r["site_id"] for r in good)):
  q=[r for r in good if r["site_id"]==sid]; vals=np.array([r["lle"] for r in q]); b=q[0]
  site.append(dict(site_id=sid,ic_id=b["ic_id"],zeta=b["zeta"],role=b["role"],verified_temporal_switching=b["verified_temporal_switching"],
   switch_count_min=b["switch_count_min"],n_lle=len(vals),lle_mean=float(vals.mean()),lle_std=float(vals.std(ddof=1)) if len(vals)>1 else 0.,
   lle_min=float(vals.min()),lle_max=float(vals.max()),all_positive=bool(np.all(vals>0))))
 with open(O/"p69_site_summary.csv","w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(site[0]));w.writeheader();w.writerows(site)
 # Figure
 fig=plt.figure(figsize=(7.16,5.8));gs=fig.add_gridspec(2,1,hspace=.38)
 ax=fig.add_subplot(gs[0])
 for role,mark in [("candidate","o"),("control","s")]:
  q=[r for r in site if r["role"]==role];ax.errorbar([r["zeta"] for r in q],[r["lle_mean"] for r in q],yerr=[r["lle_std"] for r in q],fmt=mark,ls="none",capsize=2,label=role)
 ax.axhline(0,ls="--",lw=.8);ax.set_ylabel("Largest Lyapunov exponent");ax.set_xlabel(r"Coupling strength $\zeta$");ax.set_title("Finite-time LLE across P68 sites",fontweight="bold");ax.legend(frameon=False);ax.grid(alpha=.15)
 ax=fig.add_subplot(gs[1])
 for role,mark in [("candidate","o"),("control","s")]:
  q=[r for r in site if r["role"]==role];ax.scatter([r["switch_count_min"] for r in q],[r["lle_mean"] for r in q],marker=mark,label=role)
 ax.axhline(0,ls="--",lw=.8);ax.set_xlabel("Minimum P/M switches across W=50,100,200");ax.set_ylabel("Largest Lyapunov exponent");ax.set_title("Temporal switching versus sensitive dependence",fontweight="bold");ax.grid(alpha=.15)
 stem=O/"Figure_P69_lyapunov_temporal_switching";fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight");fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight");fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight");plt.close(fig)
 sw=[r for r in site if r["verified_temporal_switching"]];ns=[r for r in site if not r["verified_temporal_switching"]]
 report={"scope":"P69 finite-time LLE characterization of P68 fixed-coupling temporal switching","source_sha256":cfg["source_sha256"],
 "settings":cfg["settings"],"summary":{"sites":len(site),"lle_runs":len(good),"switching_sites":len(sw),"switching_all_positive_lle":sum(r["all_positive"] for r in sw),
 "nonswitching_sites":len(ns),"nonswitching_all_positive_lle":sum(r["all_positive"] for r in ns)},
 "interpretation_guardrails":["Positive finite-time LLE across independent perturbation directions supports sensitive dependence consistent with chaos but does not mathematically prove a chaotic invariant set.",
 "Association between positive LLE and P/M switching does not by itself establish chaotic itinerancy or attractor hopping.",
 "Classical intermittency terminology requires mechanism-specific scaling tests."]}
 (O/"p69_report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()

