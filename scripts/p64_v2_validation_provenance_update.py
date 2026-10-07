#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    p64_v2_validation_provenance_update.py

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
import argparse, hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

VERSION='P64-v2'

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def find_one(root: Path, names):
    for name in names:
        hits=sorted(root.rglob(name))
        # Never select files inside the output freeze itself.
        hits=[p for p in hits if 'manuscript_freeze' not in p.parts]
        if hits: return hits[0]
    return None

def require(d, keys, label):
    missing=[k for k in keys if k not in d]
    if missing: raise ValueError(f'{label}: missing required fields: {missing}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--output-dir', default='results/author_reproducibility_manuscript_freeze')
    ap.add_argument('--strict', action='store_true')
    a=ap.parse_args()
    root=Path(a.results_root).resolve(); out=Path(a.output_dir).resolve(); freeze=out/'manuscript_freeze'
    tables=freeze/'tables'; figs=freeze/'figure_source_data'
    if not (freeze/'p64_report.json').is_file():
        raise FileNotFoundError(f'Existing P64-v1 freeze not found: {freeze}')
    tables.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)

    p00=find_one(root,['validation_report.json','p00_model_validation.json','p00_validation_report.json'])
    p01=find_one(root,['p01_rotating_frame_validation.json'])
    missing=[]
    if p00 is None: missing.append('P00 validation report')
    if p01 is None: missing.append('P01 rotating-frame validation report')
    if missing:
        print(json.dumps({'status':'MISSING_INPUTS','missing':missing,
          'searched_under':str(root),
          'note':'No values were reconstructed or guessed. Supply/rerun the existing P00/P01 validation scripts, then rerun P64-v2.'},indent=2))
        return 2

    d0=json.loads(p00.read_text()); d1=json.loads(p01.read_text())
    require(d0,['status','samples','tolerance','max_jacobian_absolute_error','max_time_dependence_column_error','zeta_one_time_dependence_error','seventh_derivative_error'], 'P00')
    require(d1,['status','symbolic_identity','samples','tolerance','max_rhs_error','max_jacobian_error','max_inverse_transform_error','max_trajectory_error'], 'P01')

    # Copy authoritative JSON reports unchanged.
    dst0=tables/'p00_validation_report.json'; dst1=tables/'p01_rotating_frame_validation.json'
    shutil.copy2(p00,dst0); shutil.copy2(p01,dst1)

    # Compact figure source: one row per validation metric; no inferred values.
    rows=[
      ['P00','Analytic vs numerical Jacobian','max absolute error',d0['max_jacobian_absolute_error'],d0['tolerance'],d0['status']],
      ['P00','Explicit-time/Jacobian-column check','max absolute error',d0['max_time_dependence_column_error'],d0['tolerance'],d0['status']],
      ['P00','zeta=1 time-dependence check','max absolute error',d0['zeta_one_time_dependence_error'],d0['tolerance'],d0['status']],
      ['P00',"x7'=1 check",'absolute error',d0['seventh_derivative_error'],d0['tolerance'],d0['status']],
      ['P01','Rotating-frame RHS equivalence','max absolute error',d1['max_rhs_error'],d1['tolerance'],d1['status']],
      ['P01','Rotating-frame Jacobian equivalence','max absolute error',d1['max_jacobian_error'],d1['tolerance'],d1['status']],
      ['P01','Inverse-transform round trip','max absolute error',d1['max_inverse_transform_error'],d1['tolerance'],d1['status']],
      ['P01','Original/reduced trajectory equivalence','max absolute error',d1['max_trajectory_error'],d1['tolerance'],d1['status']],
    ]
    pd.DataFrame(rows,columns=['experiment','validation','metric','value','tolerance','status']).to_csv(figs/'fig_model_validation.csv',index=False)

    # Update manifest without disturbing prior entries.
    mp=freeze/'source_manifest.csv'; m=pd.read_csv(mp) if mp.exists() else pd.DataFrame(columns=['key','source','frozen','sha256','bytes'])
    m=m[~m['key'].isin(['p00_validation_report','p01_rotating_frame_validation'])]
    add=pd.DataFrame([
      {'key':'p00_validation_report','source':str(p00),'frozen':str(dst0),'sha256':sha256(p00),'bytes':p00.stat().st_size},
      {'key':'p01_rotating_frame_validation','source':str(p01),'frozen':str(dst1),'sha256':sha256(p01),'bytes':p01.stat().st_size},
    ])
    m=pd.concat([m,add],ignore_index=True); m.to_csv(mp,index=False)

    # Append explicit validation checks.
    cp=freeze/'consistency_checks.csv'; c=pd.read_csv(cp) if cp.exists() else pd.DataFrame(columns=['check','pass','detail'])
    c=c[~c['check'].astype(str).str.startswith(('P00_','P01_','source_present:p00','source_present:p01'))]
    new=[
      {'check':'source_present:p00_validation_report','pass':True,'detail':str(p00)},
      {'check':'source_present:p01_rotating_frame_validation','pass':True,'detail':str(p01)},
      {'check':'P00_status_PASS','pass':d0['status']=='PASS','detail':d0['status']},
      {'check':'P00_jacobian_below_tolerance','pass':float(d0['max_jacobian_absolute_error'])<float(d0['tolerance']),'detail':f"error={d0['max_jacobian_absolute_error']}; tolerance={d0['tolerance']}"},
      {'check':'P01_status_PASS','pass':d1['status']=='PASS','detail':d1['status']},
      {'check':'P01_symbolic_identity','pass':bool(d1['symbolic_identity']),'detail':str(d1['symbolic_identity'])},
      {'check':'P01_rhs_below_tolerance','pass':float(d1['max_rhs_error'])<float(d1['tolerance']),'detail':f"error={d1['max_rhs_error']}; tolerance={d1['tolerance']}"},
      {'check':'P01_trajectory_below_tolerance','pass':float(d1['max_trajectory_error'])<float(d1['tolerance']),'detail':f"error={d1['max_trajectory_error']}; tolerance={d1['tolerance']}"},
    ]
    c=pd.concat([c,pd.DataFrame(new)],ignore_index=True); c.to_csv(cp,index=False)
    nfail=int((~c['pass'].astype(bool)).sum())

    old=json.loads((freeze/'p64_report.json').read_text())
    report=dict(old)
    report.update({
      'freeze_version':VERSION,
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'n_manifest_entries':int(len(m)),
      'n_sources_found':int(m['source'].notna().sum()),
      'n_checks':int(len(c)),
      'n_failed_checks':nfail,
      'status':'PASS' if nfail==0 else 'PASS_WITH_WARNINGS',
      'validation_provenance':{
        'p00_source':str(p00),'p00_sha256':sha256(p00),
        'p01_source':str(p01),'p01_sha256':sha256(p01),
        'p00_status':d0['status'],'p01_status':d1['status'],
        'p00_max_jacobian_absolute_error':d0['max_jacobian_absolute_error'],
        'p01_max_rhs_error':d1['max_rhs_error'],
      },
      'note':'P64-v2 extends P64-v1 with authoritative P00/P01 validation provenance. No new integrations or reconstructed values are introduced by this freeze update.'
    })
    (freeze/'p64_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (freeze/'README.md').write_text(f'''# {VERSION} manuscript data freeze\n\nStatus: **{report['status']}**  \nSources found: **{report['n_sources_found']}/{report['n_manifest_entries']}**  \nConsistency checks failed: **{nfail}/{len(c)}**\n\nP64-v2 extends P64-v1 by freezing the authoritative P00 model/Jacobian validation and P01 rotating-frame validation reports. It performs no new ODE integrations.\n\nNew Figure 1 source: `figure_source_data/fig_model_validation.csv`.\n''')
    print(json.dumps(report,indent=2))
    return 2 if a.strict and nfail else 0

if __name__=='__main__': raise SystemExit(main())

