#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p70_reproducibility_manuscript_freeze.py

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
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
def H(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def R(p):return list(csv.DictReader(open(p,newline="")))
def B(x):return str(x).lower() in ("true","1","yes")
def W(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,"w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def J(p,x):p.write_text(json.dumps(x,indent=2)+"\n")
def main():
 a=argparse.ArgumentParser();a.add_argument("--results-root",default="results");a.add_argument("--output-dir",default="results/author_reproducibility_manuscript_freeze_v2");q=a.parse_args()
 r=Path(q.results_root);o=Path(q.output_dir);f=o/"manuscript_freeze";t=f/"tables";s=f/"figure_source_data";t.mkdir(parents=True,exist_ok=True);s.mkdir(parents=True,exist_ok=True)
 P={"p65":r/"author_global_bifurcation_multistability"/"p65_regime_classifications.csv",
 "p66":r/"author_intermittency_verification_v2"/"p66_site_metrics.csv",
 "p67":r/"author_regime_switching_discrimination_v2"/"p67_site_summary.csv",
 "p68":r/"author_temporal_switching_verification"/"p68_site_verification.csv",
 "p68m":r/"author_temporal_switching_verification"/"p68_switching_metrics.csv",
 "p69":r/"author_lyapunov_temporal_switching"/"p69_site_summary.csv",
 "p69r":r/"author_lyapunov_temporal_switching"/"p69_lle_runs.csv"}
 miss=[str(x) for x in P.values() if not x.is_file()]
 if miss:print(json.dumps({"status":"FAIL","missing":miss},indent=2));raise SystemExit(2)
 J(f/"p70_source_manifest.json",[{"key":k,"path":str(v),"sha256":H(v),"bytes":v.stat().st_size} for k,v in P.items()])
 p65,p66,p67,p68,p68m,p69,p69r=[R(P[k]) for k in ("p65","p66","p67","p68","p68m","p69","p69r")]
 C=[]
 def ck(n,x,d):C.append({"check":n,"pass":bool(x),"detail":d})
 c=Counter(x["orientation"] for x in p65);ck("P65 2608 complete",len(p65)==2608 and all(x["status"]=="complete" for x in p65),str(len(p65)))
 ck("P65 P and M present",c["P"]>0 and c["M"]>0,str(dict(c)));ck("P66-v2 complete",all(x["status"]=="complete" for x in p66),str(len(p66)))
 modes=Counter(x["classification_mode"] for x in p67);ck("P67 strict two-detector only",modes.get("q1_proxy_screen",0)==0 and modes.get("strict_two_detector",0)==len(p67),str(dict(modes)))
 wins=sorted(set(float(x["window"]) for x in p68m));ck("P68 windows 50/100/200",wins==[50.,100.,200.],str(wins))
 ck("P68 verified switching exists",any(B(x["verified_temporal_switching"]) for x in p68),f"{sum(B(x['verified_temporal_switching']) for x in p68)}/{len(p68)}")
 ck("P69 three LLE/site",all(int(float(x["n_lle"]))==3 for x in p69),str(len(p69)));ck("P69 all mean LLE positive",all(float(x["lle_mean"])>0 for x in p69),str(len(p69)))
 sw=[x for x in p69 if B(x["verified_temporal_switching"])];ck("P69 switching sites min LLE positive",bool(sw) and all(float(x["lle_min"])>0 for x in sw),str(len(sw)))
 a68={x["job_id"]:x for x in p68};a69={x["site_id"]:x for x in p69};common=set(a68)&set(a69)
 ck("P68-P69 site/count consistency",len(common)==len(a68)==len(a69) and all(int(float(a68[k]["min_switches_across_windows"]))==int(float(a69[k]["switch_count_min"])) for k in common),f"{len(common)}/{len(a68)}/{len(a69)}")
 zg=defaultdict(list)
 for x in p65:zg[float(x["zeta"])].append(x)
 glob=[]
 for z in sorted(zg):
  cc=Counter(x["orientation"] for x in zg[z]);glob.append({"zeta":z,"n_ic":len(zg[z]),"P":cc["P"],"M":cc["M"],"ambiguous":cc["ambiguous"],"PM_coexistence":cc["P"]>0 and cc["M"]>0,"intermittency_screen_candidates":sum(B(x["intermittency_candidate"]) for x in zg[z])})
 W(t/"p65_global_regime_summary.csv",glob);W(s/"fig_global_coupling_regimes.csv",glob)
 W(t/"p68_temporal_switching_sites.csv",p68);W(t/"p69_lyapunov_sites.csv",p69)
 combo=[]
 for k in sorted(common,key=lambda k:float(a69[k]["zeta"])):
  x,y=a68[k],a69[k];combo.append({"site_id":k,"ic_id":x["ic_id"],"zeta":x["zeta"],"role":x["role"],"verified_temporal_switching":x["verified_temporal_switching"],"min_switches_across_windows":x["min_switches_across_windows"],"P_fraction":x["P_fraction"],"M_fraction":x["M_fraction"],"A_fraction":x["A_fraction"],"lle_mean":y["lle_mean"],"lle_std":y["lle_std"],"lle_min":y["lle_min"],"lle_max":y["lle_max"]})
 W(s/"fig_temporal_switching_lyapunov.csv",combo)
 G=["P65 is a global trajectory-sampled coupling survey, not proof of the complete global bifurcation structure.","P and M are finite-time physical detector-drift orientations P=(+,-) and M=(-,+).","P65 intermittency_candidate is a screening label, not proof of intermittency.","P68 establishes robust finite-time fixed-coupling P/M orientation switching under its window/persistence tests.","Positive P69 finite-time LLE supports sensitive dependence consistent with chaos, not mathematical proof of a chaotic invariant set.","Do not claim chaotic itinerancy, attractor hopping, crisis-induced switching, or a classical intermittency type without mechanism-specific evidence.","P65 candidate/control labels are not a general classifier of temporal switching.","Finite-time regime transitions are not automatically dynamical bifurcations."]
 J(f/"interpretation_guardrails.json",G);J(f/"p70_checks.json",C);bad=[x for x in C if not x["pass"]]
 rep={"freeze_version":"P70-v1","scope":"P65-P69 freeze; zero new ODE integrations","sources_found":len(P),"checks_total":len(C),"checks_failed":len(bad),"status":"PASS" if not bad else "FAIL","checks":C,"summary":{"p65_rows":len(p65),"p65_orientation_counts":dict(c),"p65_sampled_coexistence_zetas":sum(x["PM_coexistence"] for x in glob),"p68_sites":len(p68),"p68_verified_switching_sites":sum(B(x["verified_temporal_switching"]) for x in p68),"p69_sites":len(p69),"p69_positive_mean_lle_sites":sum(float(x["lle_mean"])>0 for x in p69)},"guardrails":G}
 J(o/"p70_report.json",rep);print(json.dumps(rep,indent=2))
 if bad:raise SystemExit(1)
if __name__=="__main__":main()

