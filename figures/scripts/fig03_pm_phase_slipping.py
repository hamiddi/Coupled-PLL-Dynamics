#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    fig03_pm_phase_slipping.py

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
import os; os.environ.setdefault('MPLBACKEND','Agg')
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def H(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--freeze-dir',default='results/author_reproducibility_manuscript_freeze/manuscript_freeze'); ap.add_argument('--output-dir',default='figures/figure_03'); a=ap.parse_args()
 p=Path(a.freeze_dir)/'figure_source_data/fig_PM_physical_fingerprint.csv'; d=pd.read_csv(p); o=Path(a.output_dir); o.mkdir(parents=True,exist_ok=True)
 # transition sides define physical P/M; derive all values from frozen table
 vals=[]
 for side in ['left','right']:
  for ori in ['P','M']:
   q=d[d[f'{side}_orientation']==ori]
   vals.append((ori, q[f'{side}_drift1'].mean(),q[f'{side}_drift2'].mean(),q[f'{side}_mismatch1_rms'].mean(),q[f'{side}_mismatch2_rms'].mean(),q[f'{side}_phi1_R'].mean(),q[f'{side}_phi2_R'].mean()))
 A=pd.DataFrame(vals,columns=['ori','d1','d2','m1','m2','R1','R2']).groupby('ori').mean().loc[['P','M']]
 fig,axs=plt.subplots(3,1,figsize=(3.5,6.8)); x=np.arange(2); w=.34
 axs[0].bar(x-w/2,A.d1,w,label=r'$D_1$');axs[0].bar(x+w/2,A.d2,w,label=r'$D_2$');axs[0].axhline(0,lw=.7);axs[0].set_xticks(x,['P (+,-)','M (-,+)']);axs[0].set_ylabel('Mean detector drift');axs[0].legend(frameon=False,ncol=2,fontsize=7);axs[0].set_title('Physical P/M drift orientations',fontsize=9,fontweight='bold')
 axs[1].bar(x-w/2,A.m1,w,label='RMS mismatch 1');axs[1].bar(x+w/2,A.m2,w,label='RMS mismatch 2');axs[1].set_xticks(x,['P','M']);axs[1].set_ylabel('Mismatch RMS');axs[1].set_ylim(0,1.0);axs[1].legend(frameon=False,ncol=2,fontsize=6.7,loc='upper center',bbox_to_anchor=(0.5,0.98));axs[1].set_title('Phase-slipping synchronization fingerprint',fontsize=9,fontweight='bold')
 axs[2].bar(x-w/2,A.R1,w,label=r'$R_{\varphi_1}$');axs[2].bar(x+w/2,A.R2,w,label=r'$R_{\varphi_2}$');axs[2].set_xticks(x,['P','M']);axs[2].set_ylabel('Circular concentration');axs[2].legend(frameon=False,ncol=2,fontsize=7);axs[2].set_title('Phase concentration',fontsize=9,fontweight='bold')
 for i,ax in enumerate(axs): ax.text(-.18,1.03,chr(65+i),transform=ax.transAxes,fontweight='bold');ax.grid(axis='y',alpha=.15)
 fig.tight_layout(); stem=o/'Figure_3_IEEE_single_column'
 for ext in ['png','pdf','svg']: fig.savefig(stem.with_suffix('.'+ext),dpi=600 if ext=='png' else None,bbox_inches='tight')
 plt.close(fig); (o/'figure_03_manifest.json').write_text(json.dumps({'input':str(p),'sha256':H(p),'rows':len(d),'guardrail':'P and M are finite-time detector-drift orientations; both are phase-slipping/running responses, not conventional locking.'},indent=2)+'\n')
if __name__=='__main__':main()

