#!/usr/bin/env python3
"""
===============================================================================
Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal
Switching in Mutually Coupled Third-Order Phase-Locked Loops
===============================================================================

Script:
    fig05_transition_dynamics.py

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
import os; os.environ.setdefault('MPLBACKEND','Agg'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

IEEE_WIDTH=3.5; DPI=600

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--freeze-dir',default='results/author_reproducibility_manuscript_freeze/manuscript_freeze')
    ap.add_argument('--output-dir',default='figures/figure_05')
    a=ap.parse_args(); F=Path(a.freeze_dir); O=Path(a.output_dir); O.mkdir(parents=True,exist_ok=True)
    P={
      'approach':F/'figure_source_data/fig_indicator_distance_summary.csv',
      'metrics':F/'figure_source_data/fig_transition_mechanism_metrics.csv',
      'summary':F/'figure_source_data/fig_transition_mechanism_summary.csv'}
    for p in P.values():
        if not p.is_file(): raise SystemExit(f'Missing frozen input: {p}')
    A=pd.read_csv(P['approach']); M=pd.read_csv(P['metrics']); S=pd.read_csv(P['summary'])
    assert len(A)==9 and len(M)==36 and len(S)==4
    assert (M.status=='complete').all() and M.orientation.isin(['P','M']).all()
    assert int((M.orientation=='P').sum())==17 and int((M.orientation=='M').sum())==19
    assert (S.n_complete==9).all()

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8.2,
      'axes.labelsize':7.5,'xtick.labelsize':6.8,'ytick.labelsize':6.8,
      'legend.fontsize':6.5,'axes.linewidth':.8,'lines.linewidth':1.0})
    fig,axs=plt.subplots(4,1,figsize=(IEEE_WIDTH,8.2)); fig.subplots_adjust(hspace=.68,left=.20,right=.97,top=.98,bottom=.07)
    x=1e6*A.distance_below_left.to_numpy(float)

    # A: approach-side mismatch RMS. Reverse x so movement rightward means closer to boundary.
    ax=axs[0]; ax.text(-.17,1.06,'A',transform=ax.transAxes,fontweight='bold',fontsize=9.5)
    for k,lab,mk in [('mismatch1','Mismatch 1','o'),('mismatch2','Mismatch 2','s')]:
        ax.errorbar(x,A[f'{k}_rms_mean'],yerr=A[f'{k}_rms_std'],marker=mk,ms=3,capsize=1.5,label=lab)
    ax.invert_xaxis(); ax.set_xlabel(r'Distance below P-side endpoint, $10^6\Delta\zeta$')
    ax.set_ylabel('Mismatch RMS'); ax.set_title('Mismatch decreases toward the switching region',fontweight='bold',pad=4)
    ax.legend(frameon=False,ncol=2); ax.grid(alpha=.2)

    # B: approach-side phase-slip rate.
    ax=axs[1]; ax.text(-.17,1.06,'B',transform=ax.transAxes,fontweight='bold',fontsize=9.5)
    for k,lab,mk in [('slip1','Detector 1','o'),('slip2','Detector 2','s')]:
        ax.errorbar(x,A[f'{k}_rate_mean'],yerr=A[f'{k}_rate_std'],marker=mk,ms=3,capsize=1.5,label=lab)
    ax.invert_xaxis(); ax.set_xlabel(r'Distance below P-side endpoint, $10^6\Delta\zeta$')
    ax.set_ylabel('Phase-slip rate'); ax.ticklabel_format(axis='y',style='plain',useOffset=False); ax.set_title('Phase-slip rate trends downward near switching',fontweight='bold',pad=4)
    ax.legend(frameon=False,ncol=2); ax.grid(alpha=.2)

    # C: detector drift across four representative brackets.
    # Plot the mean across crossings with the full min-max envelope to make the
    # common P->M reversal visible without obscuring it with eight overlapping curves.
    ax=axs[2]; ax.text(-.17,1.06,'C',transform=ax.transAxes,fontweight='bold',fontsize=9.5)
    C=(M.groupby('bracket_coordinate')
         .agg(d1_mean=('drift1','mean'),d1_min=('drift1','min'),d1_max=('drift1','max'),
              d2_mean=('drift2','mean'),d2_min=('drift2','min'),d2_max=('drift2','max'))
         .reset_index().sort_values('bracket_coordinate'))
    sx=C.bracket_coordinate.to_numpy(float)
    ax.fill_between(sx,C.d1_min,C.d1_max,alpha=.16)
    ax.fill_between(sx,C.d2_min,C.d2_max,alpha=.16)
    ax.plot(sx,C.d1_mean,'o-',ms=3,label=r'$D_1$')
    ax.plot(sx,C.d2_mean,'s--',ms=3,label=r'$D_2$')
    ax.axvline(0,ls=':',lw=.8); ax.axvline(1,ls=':',lw=.8); ax.axhline(0,lw=.7)
    ax.set_xlabel(r'Normalized bracket coordinate, $s$'); ax.set_ylabel('Mean detector drift')
    ax.set_title('Detector-drift reversal across P→M switching',fontweight='bold',pad=4)
    ax.grid(alpha=.2)
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,0.98))

    # D: endpoint ratios on a log2 scale, averaged over four crossings.
    # This puts heterogeneous positive-valued synchronization/spectral metrics on one interpretable scale:
    # 0=no change, +1=twofold increase, -1=twofold decrease.
    rows=[]
    for tid,q in M.groupby('transition_id',sort=False):
        L=q[q.label=='left'].iloc[0]; R=q[q.label=='right'].iloc[0]
        for key,label in [('mismatch1_rms','Mismatch 1'),('mismatch2_rms','Mismatch 2'),
                          ('slip1_rate','Slip rate 1'),('slip2_rate','Slip rate 2'),
                          ('phi1_R',r'$R_{\varphi_1}$'),('phi2_R',r'$R_{\varphi_2}$'),
                          ('mismatch1_spectral_entropy','Spectral entropy 1'),
                          ('mismatch2_spectral_entropy','Spectral entropy 2')]:
            rows.append((tid,label,np.log2(float(R[key])/float(L[key]))))
    D=pd.DataFrame(rows,columns=['transition','metric','log2ratio'])
    order=['Mismatch 1','Mismatch 2','Slip rate 1','Slip rate 2',
           r'$R_{\varphi_1}$',r'$R_{\varphi_2}$',
           'Spectral entropy 1','Spectral entropy 2']
    g=D.groupby('metric').log2ratio.agg(['mean','std']).reindex(order)
    ax=axs[3]; ax.text(-.17,1.06,'D',transform=ax.transAxes,fontweight='bold',fontsize=9.5)
    xx=np.arange(len(g)); ax.bar(xx,g['mean'],yerr=g['std'],capsize=2,width=.68); ax.axhline(0,lw=.7)
    ax.set_xticks(xx,order,rotation=35,ha='right'); ax.set_ylabel(r'$\log_2$(M/P endpoint ratio)')
    ax.set_title('Endpoint changes across P→M switching',fontweight='bold',pad=4)
    ax.grid(axis='y',alpha=.2)
    # Label each bar with its mean log2(M/P) value. Positive labels sit above
    # the bar; negative labels sit below it so the sign remains visually clear.
    for xi,val in zip(xx,g['mean'].to_numpy(float)):
        offset = 0.035 if val >= 0 else -0.035
        ax.text(xi, val + offset, f'{val:+.2f}',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=6.3)

    stem=O/'Figure_5_IEEE_single_column'
    for ext in ['png','pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=DPI if ext=='png' else None,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
    manifest={'figure':'Figure 5','freeze_version':'P64-v2','inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in P.items()},
      'checks':{'approach_levels':len(A),'p63_trajectories':len(M),'P_count':int((M.orientation=='P').sum()),'M_count':int((M.orientation=='M').sum()),'representative_crossings':len(S)},
      'outputs':[str(stem.with_suffix('.'+x)) for x in ['png','pdf','svg']],
      'guardrail':'P59 trends are deterministic precursor-like finite-time signatures, not proof of critical slowing down. P63 crossings show a finite-time P/M detector-drift reversal and associated endpoint reorganization at the sampled resolution, but do not by themselves establish a bifurcation; narrower re-entrant windows may remain unresolved.'}
    (O/'figure_05_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

