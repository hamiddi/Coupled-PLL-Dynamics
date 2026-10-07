# Repository audit and pruning decisions

The source archive contained 92 Python files. This publication repository retains the computational chain needed for the final manuscript and the final manuscript/supplementary figure builders.

## Removed as superseded or exploratory

- P05/P05b/P05c/P05d, P06, P07, P08: early exploratory multistability/transition/orbit analyses superseded by the author/final pipelines.
- P10: earlier author bifurcation scan superseded by the refined Hopf/continuation workflow used downstream.
- P52: coupling-dependent basin-topology experiment not required by the final manuscript evidence chain.
- P53 and P54: pre-correction fixed-IC continuation and its audit; P55 is the corrected continuation used by the final analyses.
- P66 (v1): superseded by P66-v2, which supplies the traces used by P68 and the P70 freeze.
- Duplicate/obsolete figure builders (`Oldfig05_*`, earlier alternative Fig. 3/4/6 scripts): replaced by the final figure scripts matched to manuscript Figures 1–7.

No numerical algorithm was rewritten during cleanup. Retained scripts had only project-title text normalized where the old working title remained. Historical filenames containing words such as `basin` or `critical_transition` were left unchanged to preserve provenance and script numbering; the manuscript's conservative interpretation remains controlling.

## Final figure mapping

- Figure 1: `fig01_model_validation_workflow.py`
- Figure 2: `fig02_local_invariant_structure.py`
- Figure 3: `fig03_pm_phase_slipping.py`
- Figure 4: `fig04_regime_selection_geometry.py`
- Figure 5: `fig05_transition_dynamics.py`
- Figure 6: `fig06_global_coupling_landscape.py`
- Figure 7: `fig07_temporal_switching_lyapunov.py`
- Supplementary Figure S1: `figS01_temporal_switching_detail.py`
- Supplementary Figure S2: `figS02_sensitivity_resilience.py`
