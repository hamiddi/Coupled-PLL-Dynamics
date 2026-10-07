#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p65_global_bifurcation_multistability.py

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
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from p00_model_validation import Parameters
from p25_boundary_mapping import integrate

def readcsv(p):
    with open(p,newline="") as f: return list(csv.DictReader(f))

def save_json(p,o):
    p=Path(p); q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(o,indent=2,allow_nan=False)+"\n")
    q.replace(p)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def req(x,m):
    if not x: raise ValueError(m)

def orient(d1,d2,zt=.05):
    if d1 < -zt and d2 > zt: return "M"
    if d1 > zt and d2 < -zt: return "P"
    return "ambiguous"

def classify(t,Y,zeta):
    phi1=Y[:,3]-Y[:,0]
    phi2=zeta*Y[:,0]-Y[:,3]
    dur=float(t[-1]-t[0])
    d1=float((phi1[-1]-phi1[0])/dur)
    d2=float((phi2[-1]-phi2[0])/dur)
    return orient(d1,d2),d1,d2,phi1,phi2

def instantaneous_detector_rates(t,Y,zeta):
    # Derivatives of detector phases obtained numerically from the simulated detector phases.
    phi1=Y[:,3]-Y[:,0]
    phi2=zeta*Y[:,0]-Y[:,3]
    return np.gradient(phi1,t), np.gradient(phi2,t), phi1, phi2

def spectral_entropy(x):
    x=np.asarray(x,float); x=x-np.mean(x)
    if len(x)<16 or np.allclose(x,0): return 0.0
    p=np.abs(np.fft.rfft(x))**2
    p=p[1:]
    s=p.sum()
    if s<=0:return 0.0
    p=p/s
    return float(-(p*np.log(p+1e-300)).sum()/np.log(len(p)))

def intermittency_screen(x):
    """
    Descriptive screening only.
    Split the post-transient signal into blocks and quantify block-to-block variance changes.
    A high variance-ratio + nontrivial spectral entropy marks a candidate for later verification.
    """
    x=np.asarray(x,float)
    nblock=12
    blocks=np.array_split(x,nblock)
    v=np.array([np.var(b) for b in blocks if len(b)>3])
    med=float(np.median(v)) if len(v) else 0.0
    vmax=float(np.max(v)) if len(v) else 0.0
    ratio=float(vmax/(med+1e-15))
    se=spectral_entropy(x)
    return ratio,se,bool(ratio>=8.0 and se>=0.25)

def select_ics(rows,n):
    # Deterministic selection across the frozen P55 IC table.
    keys=sorted(rows)
    req(keys,"No initial conditions found")
    if n>=len(keys): return keys
    idx=np.unique(np.linspace(0,len(keys)-1,n).round().astype(int))
    return [keys[i] for i in idx]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p55-fixed-ics",
        default="results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv")
    ap.add_argument("--output-dir",default="results/author_global_bifurcation_multistability")
    ap.add_argument("--zeta-min",type=float,default=0.40)
    ap.add_argument("--zeta-max",type=float,default=1.05)
    ap.add_argument("--n-zeta",type=int,default=326)
    ap.add_argument("--n-ics",type=int,default=8)
    ap.add_argument("--transient",type=float,default=800.0)
    ap.add_argument("--observe",type=float,default=800.0)
    ap.add_argument("--dt",type=float,default=0.5)
    ap.add_argument("--sample-stride",type=int,default=8,
                    help="Keep every Nth post-transient sample in bifurcation CSV/plot.")
    ap.add_argument("--rtol",type=float,default=1e-11)
    ap.add_argument("--atol",type=float,default=1e-13)
    ap.add_argument("--max-abs",type=float,default=1e7)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--diagnostic-only",action="store_true")
    a=ap.parse_args()

    p55=Path(a.p55_fixed_ics)
    req(p55.is_file(),f"Missing P55 IC table: {p55}")
    req(a.zeta_max>a.zeta_min and a.n_zeta>=5 and a.n_ics>=1,"Invalid sweep settings")
    req(a.sample_stride>=1,"sample-stride must be >=1")

    rows={r["ic_id"]:r for r in readcsv(p55)}
    ids=select_ics(rows,a.n_ics)
    X={k:np.array([float(rows[k][f"x{i}"]) for i in range(1,7)],float) for k in ids}
    zetas=np.linspace(a.zeta_min,a.zeta_max,a.n_zeta)

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/"p65_checkpoint.json"
    config_path=out/"p65_config.json"
    config={
      "source_sha256":{"p55_fixed_ics":sha(p55)},
      "selected_ics":ids,
      "settings":{k:v for k,v in vars(a).items() if k not in ("resume","diagnostic_only","output_dir")},
      "zetas":zetas.tolist(),
      "model_parameters":{"a":.8,"b":.2,"c":.05,"d":.05,"delta1":.02,"delta2":.02,"omega":1.,"W":100.},
      "classification":{"P":"drift1>0.05 and drift2<-0.05","M":"drift1<-0.05 and drift2>0.05"}
    }
    if a.resume and config_path.exists():
        req(json.loads(config_path.read_text())==config,"Resume configuration mismatch")
    if not a.resume and checkpoint.exists():
        raise ValueError("Existing P65 checkpoint; use --resume or another output directory")
    save_json(config_path,config)

    total=len(ids)*len(zetas)
    if a.diagnostic_only:
        save_json(out/"p65_preflight.json",{
          "status":"PASS","n_ics":len(ids),"n_zeta":len(zetas),"integrations":total,
          "zeta_range":[float(zetas[0]),float(zetas[-1])],
          "zeta_step":float(zetas[1]-zetas[0]),"selected_ics":ids})
        print(f"P65 preflight PASS: {len(ids)} ICs x {len(zetas)} zeta = {total} integrations")
        return

    records=json.loads(checkpoint.read_text()) if a.resume and checkpoint.exists() else []
    done={r["job_id"] for r in records}
    sample_file=out/"p65_bifurcation_samples.csv"
    if not sample_file.exists() or not a.resume:
        with open(sample_file,"w",newline="") as f:
            csv.writer(f).writerow(["job_id","ic_id","zeta","orientation","t","detector_rate1","detector_rate2","wrapped_phi1"])

    completed=len(done)
    for ii,ic in enumerate(ids):
        for j,z in enumerate(zetas):
            jid=f"{ic}_z{j:04d}"
            if jid in done: continue
            rec={"job_id":jid,"ic_id":ic,"zeta":float(z),"status":"failed"}
            try:
                p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=float(z),omega=1.,W=100.)
                x=X[ic].copy()
                if a.transient:
                    x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
                t,Y=integrate(x,p,a.observe,a.dt,a.rtol,a.atol,a.max_abs)
                o,d1,d2,phi1,phi2=classify(t,Y,float(z))
                q1,q2,phi1,_=instantaneous_detector_rates(t,Y,float(z))
                vr,se,icand=intermittency_screen(q1)
                rec.update(status="complete",orientation=o,drift1=d1,drift2=d2,
                           detector_rate1_mean=float(np.mean(q1)),
                           detector_rate1_std=float(np.std(q1)),
                           detector_rate1_min=float(np.min(q1)),
                           detector_rate1_max=float(np.max(q1)),
                           spectral_entropy=float(se),block_variance_ratio=float(vr),
                           intermittency_candidate=bool(icand))
                take=np.arange(0,len(t),a.sample_stride)
                with open(sample_file,"a",newline="") as f:
                    w=csv.writer(f)
                    for k in take:
                        w.writerow([jid,ic,float(z),o,float(t[k]),float(q1[k]),float(q2[k]),
                                    float(np.angle(np.exp(1j*phi1[k])))])
            except Exception as ex:
                rec["error"]=repr(ex)
            records.append(rec); done.add(jid); save_json(checkpoint,records)
            completed+=1
            print(f"P65 {completed:05d}/{total} {ic} zeta={z:.7f}: "
                  f"{rec.get('orientation','failed')}",flush=True)

    # Compact classification table.
    fields=["job_id","ic_id","zeta","status","orientation","drift1","drift2",
            "detector_rate1_mean","detector_rate1_std","detector_rate1_min","detector_rate1_max",
            "spectral_entropy","block_variance_ratio","intermittency_candidate","error"]
    with open(out/"p65_regime_classifications.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(records)

    good=[r for r in records if r["status"]=="complete"]
    # Summary per zeta.
    summary_rows=[]
    for z in zetas:
        rr=[r for r in good if abs(r["zeta"]-float(z))<1e-14]
        summary_rows.append({
          "zeta":float(z),"n_complete":len(rr),
          "n_P":sum(r["orientation"]=="P" for r in rr),
          "n_M":sum(r["orientation"]=="M" for r in rr),
          "n_ambiguous":sum(r["orientation"]=="ambiguous" for r in rr),
          "n_intermittency_candidates":sum(bool(r["intermittency_candidate"]) for r in rr)
        })
    with open(out/"p65_summary.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)

    # Publication-style figure.
    print("P65: generating publication figure...",flush=True)
    s=np.genfromtxt(sample_file,delimiter=",",names=True,dtype=None,encoding="utf-8")
    fig=plt.figure(figsize=(7.16,6.2))
    gs=fig.add_gridspec(2,1,height_ratios=[2.2,1.0],hspace=.34)

    ax=fig.add_subplot(gs[0])
    ax.text(-.07,1.02,"A",transform=ax.transAxes,fontweight="bold",fontsize=11)
    for o,marker,alpha in [("P",".",.28),("M",".",.28),("ambiguous",".",.16)]:
        mask=s["orientation"]==o
        if np.any(mask): ax.scatter(s["zeta"][mask],s["detector_rate1"][mask],s=.7,alpha=alpha,label=o,rasterized=True)
    ax.set_ylabel(r"Detector-frequency mismatch $\dot{\phi}_1$")
    ax.set_xlabel(r"Coupling strength $\zeta$")
    ax.set_title("Global trajectory-sampled coupling diagram",fontweight="bold")
    ax.grid(alpha=.15)
    ax.legend(title="Finite-time orientation",ncol=3,loc="upper right",frameon=False)

    ax=fig.add_subplot(gs[1])
    ax.text(-.07,1.04,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)
    mat=np.full((len(ids),len(zetas)),np.nan)
    mp={"M":0.0,"ambiguous":1.0,"P":2.0}
    for r in good:
        mat[ids.index(r["ic_id"]),int(np.argmin(np.abs(zetas-r["zeta"])))]=mp[r["orientation"]]
    im=ax.imshow(mat,aspect="auto",origin="lower",interpolation="nearest",
                 extent=[zetas[0],zetas[-1],-.5,len(ids)-.5],cmap="viridis",vmin=0,vmax=2)
    ax.set_xlabel(r"Coupling strength $\zeta$")
    ax.set_ylabel("Initial-condition index")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels(range(1,len(ids)+1))
    ax.set_title("Multistability-aware regime map",fontweight="bold")
    cb=fig.colorbar(im,ax=ax,pad=.015,aspect=20,ticks=[0,1,2]); cb.ax.set_yticklabels(["M","ambiguous","P"])

    fig.subplots_adjust(left=.11,right=.93,top=.95,bottom=.09)
    stem=out/"Figure_P65_global_bifurcation_multistability"
    fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)

    report={
      "scope":"P65 broad coupling trajectory-sampling diagram with multistability-aware P/M classification",
      "source_sha256":config["source_sha256"],"settings":config["settings"],
      "summary":{"planned_integrations":total,"complete":len(good),
                 "failed":sum(r["status"]!="complete" for r in records),
                 "intermittency_candidate_jobs":sum(bool(r.get("intermittency_candidate")) for r in good)},
      "interpretation_guardrails":[
        "The trajectory-sampled diagram is a numerical bifurcation-style survey; visual branch changes alone do not establish a bifurcation type.",
        "P/M are physical detector-drift orientations and are finite-time classifications.",
        "Intermittency flags are screening diagnostics only and require fixed-parameter time-series verification.",
        "Multiple frozen initial conditions are used to expose coexistence that a single-trajectory continuation could hide."
      ]}
    save_json(out/"p65_report.json",report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    main()

