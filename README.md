# Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal Switching in Mutually Coupled Third-Order Phase-Locked Loops

This repository contains the cleaned computational code retained to reproduce the numerical study and its final manuscript figures.

## Scientific scope

The code analyzes two mutually coupled third-order phase-locked loops using model validation, an autonomous rotating-frame transformation, equilibrium/Hopf analysis, periodic-orbit continuation and Floquet analysis, finite-time P/M detector-drift classification, local initial-condition regime-selection mapping, coupling-transition refinement, synchronization diagnostics, perturbation/resilience tests, broad coupling surveys, fixed-coupling temporal switching, and finite-time largest Lyapunov exponents.

The physical trajectory classifications are

- **P:** detector-drift orientation `(D1, D2) = (+, -)`
- **M:** detector-drift orientation `(D1, D2) = (-, +)`

These are reproducible **finite-time phase-slipping regimes**. The repository does not treat every P/M transition as a bifurcation, does not claim a proven fractal basin boundary, and does not treat positive finite-time Lyapunov exponents as mathematical proof of an asymptotic chaotic invariant set.

## Repository layout

```text
scripts/             Numerical analysis programs, kept in one directory
figures/scripts/     Final manuscript and supplementary figure builders
figures/output/      Generated figures (not tracked)
results/             Generated numerical outputs (not tracked)
docs/                Audit and reproduction documentation
requirements.txt     Minimal Python dependencies
environment.yml      Conda environment
CITATION.cff         Citation metadata
```

## Computational stages retained

1. **P00–P04:** model/Jacobian validation, rotating-frame validation, equilibrium/Hopf groundwork, periodic-orbit/Floquet utilities.
2. **P09, P11–P23:** final benchmark checks, Hopf verification, periodic-orbit continuation, fold/Floquet diagnostics, fixed-coupling orbit analysis, and long-time trajectory verification.
3. **P24–P51:** local finite-time regime-selection geometry, progressive boundary refinement/validation, resilience, high-resolution verification, and uncertainty-scaling analysis.
4. **P55–P64:** corrected fixed-initial-condition coupling continuation, transition localization/refinement, synchronization/transition diagnostics, re-entrant topology, robustness/resilience, trajectory mechanism, and manuscript-data freeze.
5. **P65–P70:** broad coupling survey, fixed-coupling temporal-switching verification, finite-time Lyapunov characterization, and final freeze/provenance checks.

The missing numbers correspond to exploratory or superseded scripts deliberately removed from the publication repository; see `docs/REPOSITORY_AUDIT.md`.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate coupled-pll-dynamics
```

or with pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Run programs from the repository root so that the shared `scripts/` imports and `results/` paths resolve consistently. For example:

```bash
python scripts/p00_model_validation.py
python scripts/p01_rotating_frame_validation.py
```

Several continuation/refinement programs intentionally require explicit upstream report paths. This prevents a script from silently consuming the wrong historical result. Use `python scripts/<name>.py --help` for the required inputs.

## Manuscript data freezes

After P00–P63 outputs are available:

```bash
python scripts/p64_reproducibility_manuscript_freeze.py --strict
python scripts/p64_v2_validation_provenance_update.py
```

After P65–P69 outputs are available:

```bash
python scripts/p70_reproducibility_manuscript_freeze.py
```

The freeze stages perform provenance/consistency checks and create normalized source tables used by the final figures. They do not run new ODE integrations.

## Build the final figures

After both freezes exist:

```bash
python figures/scripts/make_all_figures.py
```

This rebuilds manuscript Figures 1–7 from the frozen source data. Supplementary figures can be generated directly from `figS01_temporal_switching_detail.py` and `figS02_sensitivity_resilience.py` using their `--help` options.

## Reproducibility note

Some stages are computationally expensive. The scripts preserve explicit output directories, reports, checkpoints, source hashes, and manuscript freezes so a published release can include selected frozen source tables in addition to the code. Before public release, add the final manuscript DOI and, if desired, archive a tagged GitHub release with Zenodo.

## License

No open-source license has been assigned yet. Until the authors select a license, normal copyright restrictions apply.
