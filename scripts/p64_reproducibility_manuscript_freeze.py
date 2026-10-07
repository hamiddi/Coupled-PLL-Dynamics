#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p64_reproducibility_manuscript_freeze.py

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
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

FREEZE_VERSION = "P64-v1"
GUARDRAILS = [
    "P/M denotes physical detector-drift orientation: M=(-,+), P=(+,-); historical C00/C01 labels may permute with coupling.",
    "Finite-time regime classifications do not prove asymptotic attractor membership or global basin topology.",
    "P60 re-entrant P/M windows are not, by themselves, bifurcations or evidence of fractality.",
    "P59 statistics are candidate transition-associated/precursor-like indicators, not proof of critical slowing down.",
    "P61 stochastic forcing is an uncalibrated numerical stress test, not a calibrated physical PLL noise model.",
    "P62 radii are first-detected switching radii along sampled rays, not global distances to a basin boundary.",
]

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def first_existing(root: Path, patterns):
    for pat in patterns:
        hits=sorted(root.glob(pat))
        if hits: return hits[0]
    return None

def discover(root: Path):
    specs = {
      "p20_folds": ["**/p20_folds.csv"],
      "p21_orbits": ["**/p21_orbits.csv"],
      "p22_equilibria": ["**/p22_equilibria.csv"],
      "p23_windows": ["**/p23_windows.csv"],
      "p44_crossings": ["**/p44_crossings.csv"],
      "p48_crossings": ["**/p48_crossings.csv"],
      "p49_baselines": ["**/p49_baselines.csv"],
      "p50_transition_brackets": ["**/p50_transition_brackets.csv", "**/p50*crossings*.csv"],
      "p51_uncertainty": ["**/p51*uncertainty*.csv", "**/p51*summary*.csv"],
      "p55_fixed_initial_conditions": ["**/p55_fixed_initial_conditions.csv"],
      "p56_transition_brackets": ["**/p56_transition_brackets.csv"],
      "p57_refined_thresholds": ["**/p57_refined_thresholds.csv"],
      "p58_metrics": ["**/p58_metrics.csv"],
      "p58_paired_transition_metrics": ["**/p58_paired_transition_metrics.csv"],
      "p59_metrics": ["**/p59_metrics.csv"],
      "p59_ic_trends": ["**/p59_ic_trends.csv"],
      "p59_distance_summary": ["**/p59_distance_summary.csv"],
      "p60_ic_summary": ["**/p60_ic_summary.csv"],
      "p60_transition_brackets": ["**/p60_transition_brackets.csv"],
      "p61_anchor_summary": ["**/p61_anchor_summary.csv"],
      "p61_robustness_summary": ["**/p61_robustness_summary.csv"],
      "p62_anchor_resilience": ["**/p62_anchor_resilience.csv"],
      "p62_first_switch_brackets": ["**/p62_first_switch_brackets.csv"],
      "p63_metrics": ["**/p63_metrics.csv"],
      "p63_transition_summary": ["**/p63_transition_summary.csv"],
    }
    return {k:first_existing(root,v) for k,v in specs.items()}

def col(df, names):
    low={c.lower():c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    return None

def add_check(checks, name, passed, detail):
    checks.append({"check":name,"pass":bool(passed),"detail":str(detail)})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-root', default='results', help='Root containing authoritative experiment outputs')
    ap.add_argument('--output-dir', default='results/author_reproducibility_manuscript_freeze')
    ap.add_argument('--strict', action='store_true', help='Exit nonzero if a core late-stage source/check is missing/fails')
    args=ap.parse_args()
    root=Path(args.results_root).resolve(); out=Path(args.output_dir).resolve()
    freeze=out/'manuscript_freeze'; tables=freeze/'tables'; figs=freeze/'figure_source_data'
    for d in (out,freeze,tables,figs): d.mkdir(parents=True, exist_ok=True)

    found=discover(root)
    manifest=[]
    for key,p in found.items():
        if p and p.is_file():
            dst=tables/f'{key}.csv'; shutil.copy2(p,dst)
            manifest.append({"key":key,"source":str(p),"frozen":str(dst),"sha256":sha256(p),"bytes":p.stat().st_size})
        else: manifest.append({"key":key,"source":None,"frozen":None,"sha256":None,"bytes":None})
    pd.DataFrame(manifest).to_csv(freeze/'source_manifest.csv',index=False)

    checks=[]
    core=["p55_fixed_initial_conditions","p56_transition_brackets","p57_refined_thresholds","p58_metrics","p59_metrics","p60_transition_brackets","p61_robustness_summary","p62_first_switch_brackets","p63_metrics","p63_transition_summary"]
    for k in core: add_check(checks,f"source_present:{k}",found[k] is not None,found[k] or 'MISSING')

    # P57: refined thresholds should be 23 and narrow if recognizable columns exist.
    if found['p57_refined_thresholds']:
        d=pd.read_csv(found['p57_refined_thresholds']); add_check(checks,'P57_n_thresholds_23',len(d)==23,f'n={len(d)}')
        lc=col(d,['zeta_left','left','left_zeta']); rc=col(d,['zeta_right','right','right_zeta'])
        if lc and rc:
            w=(d[rc]-d[lc]).abs(); add_check(checks,'P57_brackets_tight',float(w.max())<=1.1e-7,f'max_width={w.max():.12g}')
            pd.DataFrame({'threshold_midpoint':(d[lc]+d[rc])/2,'width':w}).to_csv(figs/'fig_threshold_distribution.csv',index=False)

    # P59 candidate indicators: copy compact trend table as figure source.
    if found['p59_ic_trends']:
        d=pd.read_csv(found['p59_ic_trends']); d.to_csv(figs/'fig_candidate_indicator_trends.csv',index=False)
    if found['p59_distance_summary']:
        pd.read_csv(found['p59_distance_summary']).to_csv(figs/'fig_indicator_distance_summary.csv',index=False)

    # P60: count transition brackets and preserve re-entrant topology table.
    if found['p60_transition_brackets']:
        d=pd.read_csv(found['p60_transition_brackets']); add_check(checks,'P60_transition_count_78',len(d)==78,f'n={len(d)}')
        d.to_csv(figs/'fig_coupling_reentrance_brackets.csv',index=False)

    # P61 robustness summary.
    if found['p61_robustness_summary']:
        d=pd.read_csv(found['p61_robustness_summary']); d.to_csv(figs/'fig_perturbation_noise_robustness.csv',index=False)

    # P62 resilience source.
    if found['p62_first_switch_brackets']:
        d=pd.read_csv(found['p62_first_switch_brackets']); add_check(checks,'P62_persistent_switch_brackets_35',len(d)==35,f'n={len(d)}')
        d.to_csv(figs/'fig_local_resilience_radii.csv',index=False)

    # P63 mechanism source and completion check.
    if found['p63_metrics']:
        d=pd.read_csv(found['p63_metrics']); add_check(checks,'P63_n_integrations_36',len(d)==36,f'n={len(d)}')
        d.to_csv(figs/'fig_transition_mechanism_metrics.csv',index=False)
    if found['p63_transition_summary']:
        pd.read_csv(found['p63_transition_summary']).to_csv(figs/'fig_transition_mechanism_summary.csv',index=False)

    # P58 physical fingerprints.
    if found['p58_paired_transition_metrics']:
        pd.read_csv(found['p58_paired_transition_metrics']).to_csv(figs/'fig_PM_physical_fingerprint.csv',index=False)

    cdf=pd.DataFrame(checks); cdf.to_csv(freeze/'consistency_checks.csv',index=False)
    nfail=int((~cdf['pass']).sum()) if len(cdf) else 0
    report={
      "freeze_version":FREEZE_VERSION,
      "created_utc":datetime.now(timezone.utc).isoformat(),
      "results_root":str(root),
      "output_dir":str(out),
      "n_manifest_entries":len(manifest),
      "n_sources_found":sum(m['source'] is not None for m in manifest),
      "n_checks":len(checks),"n_failed_checks":nfail,
      "guardrails":GUARDRAILS,
      "status":"PASS" if nfail==0 else "PASS_WITH_WARNINGS",
      "note":"P64 performs no new integrations; it freezes and audits existing evidence. Missing early-stage optional tables remain visible in source_manifest.csv rather than being guessed."
    }
    (freeze/'p64_report.json').write_text(json.dumps(report,indent=2))
    (freeze/'SCIENTIFIC_GUARDRAILS.md').write_text('# Scientific interpretation guardrails\n\n'+'\n'.join(f'- {x}' for x in GUARDRAILS)+'\n')
    (freeze/'README.md').write_text(f'''# {FREEZE_VERSION} manuscript data freeze\n\nStatus: **{report['status']}**  \nSources found: **{report['n_sources_found']}/{report['n_manifest_entries']}**  \nConsistency checks failed: **{nfail}/{len(checks)}**\n\nThis directory is generated from authoritative experiment outputs and contains no new ODE simulations.\n\n- `source_manifest.csv`: exact source paths and SHA-256 hashes.\n- `consistency_checks.csv`: machine-readable cross-experiment audit.\n- `tables/`: frozen copies of discovered authoritative tables.\n- `figure_source_data/`: normalized manuscript/figure source CSVs.\n- `SCIENTIFIC_GUARDRAILS.md`: claims that must remain qualified in the manuscript.\n''')
    print(json.dumps(report,indent=2))
    if args.strict and nfail: return 2
    return 0
if __name__=='__main__': raise SystemExit(main())

