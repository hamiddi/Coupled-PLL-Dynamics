#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p66_v2_intermittency_verification.py

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
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p00_model_validation import Parameters
from p25_boundary_mapping import integrate

def readcsv(p):
    with open(p,newline="") as f:return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,o):
    p=Path(p); q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(o,indent=2,allow_nan=False)+"\n"); q.replace(p)
def req(x,m):
    if not x: raise ValueError(m)
def orient(d1,d2,zt=.05):
    if d1 < -zt and d2 > zt:return "M"
    if d1 > zt and d2 < -zt:return "P"
    return "ambiguous"
def moving_rms(x,n):
    x=np.asarray(x,float); n=max(3,int(n))
    k=np.ones(n)/n
    return np.sqrt(np.convolve(x*x,k,mode="same"))
def runs(mask,dt):
    mask=np.asarray(mask,bool); z=np.r_[False,mask,False].astype(int)
    a=np.flatnonzero(np.diff(z)==1); b=np.flatnonzero(np.diff(z)==-1)
    return (b-a)*dt
def analyze(t,Y,zeta,env_window=20.0):
    phi1=Y[:,3]-Y[:,0]; phi2=zeta*Y[:,0]-Y[:,3]
    dur=t[-1]-t[0]; d1=(phi1[-1]-phi1[0])/dur; d2=(phi2[-1]-phi2[0])/dur
    q1=np.gradient(phi1,t)
    q2=np.gradient(phi2,t)
    # Activity around robust center; envelope separates quiet and burst epochs.
    center=np.median(q1); dev=q1-center
    env=moving_rms(dev,max(3,round(env_window/np.median(np.diff(t)))))
    lo,hi=np.quantile(env,[.25,.75])
    # Robust threshold between lower/upper activity populations.
    thr=float((lo+hi)/2)
    lam=env<=thr
    ld=runs(lam,float(np.median(np.diff(t))))
    bd=runs(~lam,float(np.median(np.diff(t))))
    # Ignore tiny episodes shorter than 2 output intervals.
    minrun=2*np.median(np.diff(t)); ld=ld[ld>=minrun]; bd=bd[bd>=minrun]
    frac=float(np.mean(lam))
    contrast=float(hi/(lo+1e-15))
    return dict(orientation=orient(d1,d2),drift1=float(d1),drift2=float(d2),
      threshold=thr,laminar_fraction=frac,activity_contrast=contrast,
      n_laminar=int(len(ld)),n_burst=int(len(bd)),
      laminar_mean=float(np.mean(ld)) if len(ld) else 0.,
      laminar_median=float(np.median(ld)) if len(ld) else 0.,
      laminar_max=float(np.max(ld)) if len(ld) else 0.,
      burst_mean=float(np.mean(bd)) if len(bd) else 0.,
      q1=q1,q2=q2,env=env,lam=lam,laminar_durations=ld,burst_durations=bd)

def choose_sites(rows):
    good=[r for r in rows if r.get("status")=="complete"]
    # Predeclared representative candidate couplings from the frozen P65 screen:
    # onset, dense candidate bands, and all-IC candidate site.
    targets=[.926,.940,.944,.952,.972,.980,.998]
    sites=[]
    for z in targets:
        rr=[r for r in good if abs(float(r["zeta"])-z)<1e-12 and str(r["intermittency_candidate"]).lower()=="true"]
        if rr:
            rr.sort(key=lambda r:float(r["block_variance_ratio"]),reverse=True)
            sites.append((rr[0]["ic_id"],z,"candidate"))
    # Controls: same IC, nearest sampled noncandidate zeta within 0.01.
    for ic,z,_ in list(sites):
        rr=[r for r in good if r["ic_id"]==ic and str(r["intermittency_candidate"]).lower()!="true"
            and abs(float(r["zeta"])-z)<=.01 and abs(float(r["zeta"])-z)>0]
        if rr:
            rr.sort(key=lambda r:abs(float(r["zeta"])-z))
            sites.append((ic,float(rr[0]["zeta"]),"control"))
    # unique
    out=[]; seen=set()
    for s in sites:
        if (s[0],s[1]) not in seen: out.append(s); seen.add((s[0],s[1]))
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p55-fixed-ics",default="results/author_corrected_fixed_ic_coupling_continuation/p55_fixed_initial_conditions.csv")
    ap.add_argument("--p65-classifications",default="results/author_global_bifurcation_multistability/p65_regime_classifications.csv")
    ap.add_argument("--output-dir",default="results/author_intermittency_verification_v2")
    ap.add_argument("--transient",type=float,default=1600.)
    ap.add_argument("--observe",type=float,default=12800.)
    ap.add_argument("--dt",type=float,default=.25)
    ap.add_argument("--rtol",type=float,default=1e-11); ap.add_argument("--atol",type=float,default=1e-13)
    ap.add_argument("--max-abs",type=float,default=1e7)
    ap.add_argument("--env-window",type=float,default=20.)
    ap.add_argument("--resume",action="store_true"); ap.add_argument("--diagnostic-only",action="store_true")
    a=ap.parse_args()
    p55=Path(a.p55_fixed_ics); p65=Path(a.p65_classifications); req(p55.is_file(),"Missing P55 ICs"); req(p65.is_file(),"Missing P65 classifications")
    icrows={r["ic_id"]:r for r in readcsv(p55)}; p65rows=readcsv(p65); sites=choose_sites(p65rows); req(sites,"No P66 sites selected")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    cfg={"source_sha256":{"p55":sha(p55),"p65_classifications":sha(p65)},"sites":[{"ic_id":i,"zeta":z,"role":r} for i,z,r in sites],
         "settings":{k:v for k,v in vars(a).items() if k not in ("resume","diagnostic_only","output_dir")}}
    cp=out/"p66_checkpoint.json"; cf=out/"p66_config.json"
    if a.resume and cf.exists(): req(json.loads(cf.read_text())==cfg,"Resume configuration mismatch")
    if not a.resume and cp.exists(): raise ValueError("Existing P66 checkpoint; use --resume or another output directory")
    save_json(cf,cfg)
    if a.diagnostic_only:
        save_json(out/"p66_preflight.json",{"status":"PASS","n_sites":len(sites),"sites":cfg["sites"]})
        print(f"P66 preflight PASS: {len(sites)} fixed-parameter sites"); return
    records=json.loads(cp.read_text()) if a.resume and cp.exists() else []; done={r["job_id"] for r in records}
    traces=[]
    for n,(ic,z,role) in enumerate(sites,1):
        jid=f"{role}_{ic}_z{z:.6f}"
        if jid in done: continue
        row=icrows[ic]; x=np.array([float(row[f"x{k}"]) for k in range(1,7)])
        rec={"job_id":jid,"ic_id":ic,"zeta":z,"role":role,"status":"failed"}
        try:
            p=Parameters(a=.8,b=.2,c=.05,d=.05,delta1=.02,delta2=.02,zeta=z,omega=1.,W=100.)
            x=integrate(x,p,a.transient,a.dt,a.rtol,a.atol,a.max_abs)[1][-1]
            t,Y=integrate(x,p,a.observe,a.dt,a.rtol,a.atol,a.max_abs)
            m=analyze(t,Y,z,a.env_window)
            # Operational verification requires repeated laminar/burst episodes and meaningful activity separation.
            verified=bool(m["n_laminar"]>=5 and m["n_burst"]>=5 and .05<m["laminar_fraction"]<.95 and m["activity_contrast"]>=1.5)
            rec.update({k:v for k,v in m.items() if k not in ("q1","q2","env","lam","laminar_durations","burst_durations")})
            rec["intermittency_like_verified"]=verified; rec["status"]="complete"
            np.savez_compressed(out/f"{jid}_trace.npz",t=t,q1=m["q1"],q2=m["q2"],envelope=m["env"],laminar=m["lam"],
                                laminar_durations=m["laminar_durations"],burst_durations=m["burst_durations"])
        except Exception as ex: rec["error"]=repr(ex)
        records.append(rec); done.add(jid); save_json(cp,records)
        print(f"P66 {n:02d}/{len(sites)} {role} {ic} zeta={z:.6f}: {rec.get('status')} verified={rec.get('intermittency_like_verified')}",flush=True)
    fields=["job_id","ic_id","zeta","role","status","orientation","drift1","drift2","threshold","laminar_fraction","activity_contrast",
            "n_laminar","n_burst","laminar_mean","laminar_median","laminar_max","burst_mean","intermittency_like_verified","error"]
    with open(out/"p66_site_metrics.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(records)
    good=[r for r in records if r["status"]=="complete"]; cand=[r for r in good if r["role"]=="candidate"]; ctrl=[r for r in good if r["role"]=="control"]
    # Figure: representative strongest verified candidate + candidate/control summary.
    fig=plt.figure(figsize=(7.16,6.2)); gs=fig.add_gridspec(2,1,height_ratios=[1.7,1],hspace=.36)
    ax=fig.add_subplot(gs[0]); ax.text(-.07,1.02,"A",transform=ax.transAxes,fontweight="bold",fontsize=11)
    reps=[r for r in cand if r.get("intermittency_like_verified")]
    if not reps: reps=cand
    if reps:
        rep=max(reps,key=lambda r:r.get("activity_contrast",0)); q=np.load(out/f"{rep['job_id']}_trace.npz")
        # display first 2500 time units for legibility
        mask=q["t"]<=q["t"][0]+2500
        ax.plot(q["t"][mask],q["q1"][mask],lw=.45)
        ax.set_title(f"Fixed-parameter detector-rate trace: zeta={rep['zeta']:.3f}",fontweight="bold")
    ax.set_xlabel("Time"); ax.set_ylabel(r"$\dot{\phi}_1$"); ax.grid(alpha=.15)
    ax=fig.add_subplot(gs[1]); ax.text(-.07,1.04,"B",transform=ax.transAxes,fontweight="bold",fontsize=11)
    for role,mark in [("candidate","o"),("control","s")]:
        rr=[r for r in good if r["role"]==role]
        ax.scatter([r["zeta"] for r in rr],[r["activity_contrast"] for r in rr],marker=mark,label=role)
    ax.axhline(1.5,ls="--",lw=.8); ax.set_xlabel(r"Coupling strength $\zeta$"); ax.set_ylabel("Activity contrast (Q75/Q25)")
    ax.set_title("Long-run candidate/control verification",fontweight="bold"); ax.legend(frameon=False); ax.grid(alpha=.15)
    stem=out/"Figure_P66_intermittency_verification"; fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight"); fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight"); plt.close(fig)
    report={"scope":"P66-v2 targeted fixed-parameter verification of P65 intermittency candidates with both detector-rate traces saved",
      "source_sha256":cfg["source_sha256"],"settings":cfg["settings"],
      "summary":{"sites":len(sites),"complete":len(good),"failed":len(records)-len(good),
                 "candidate_sites":len(cand),"candidate_verified":sum(bool(r.get("intermittency_like_verified")) for r in cand),
                 "control_sites":len(ctrl),"control_verified":sum(bool(r.get("intermittency_like_verified")) for r in ctrl)},
      "operational_definition":"Repeated low/high detector-rate activity episodes (>=5 each), laminar fraction 0.05-0.95, and Q75/Q25 activity-envelope contrast >=1.5 during a 12800-unit fixed-parameter observation.",
      "interpretation_guardrails":[
       "A positive result supports intermittency-like dynamics under the stated operational definition; it does not identify a classical intermittency universality class.",
       "Candidate sites were selected from P65 and are not an independent discovery sample; nearby noncandidate controls provide a local comparison.",
       "Finite-time numerical trajectories do not prove asymptotic invariant-set structure.",
       "If classical intermittency type is to be claimed, a subsequent experiment must test the appropriate laminar-length scaling law versus distance from a resolved transition parameter."
      ]}
    save_json(out/"p66_report.json",report); print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()

