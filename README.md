# Coupled-PLL-Dynamics

## Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal Switching in Mutually Coupled Third-Order Phase-Locked Loops

**Reproducibility repository for the numerical analyses and final manuscript figures.**

This repository contains the cleaned computational workflow retained for the study of mutually coupled third-order phase-locked loops (PLLs). It includes model validation, local invariant-structure analysis, finite-time P/M regime classification, regime-selection geometry, coupling-transition analysis, global coupling surveys, temporal-switching analysis, finite-time Lyapunov characterization, and manuscript figure generation.

> **Recommended starting point:** validate the repository with `python run_pipeline.py --check`, inspect the workflow with `python run_pipeline.py --list`, and preview the full run with `python run_pipeline.py --full --dry-run`.

---

## Contents

- [Scientific scope](#scientific-scope)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Reproducibility pipeline](#reproducibility-pipeline)
- [Computational stages](#computational-stages)
- [Manuscript data freezes](#manuscript-data-freezes)
- [Building the final figures](#building-the-final-figures)
- [Outputs and logs](#outputs-and-logs)
- [Reproducibility and interpretation](#reproducibility-and-interpretation)
- [Citation](#citation)
- [License](#license)

---

## Scientific scope

The code analyzes two mutually coupled third-order PLLs using:

- model and Jacobian validation;
- autonomous rotating-frame validation;
- equilibrium and Hopf analysis;
- periodic-orbit continuation and Floquet analysis;
- finite-time P/M detector-drift classification;
- local initial-condition regime-selection mapping;
- coupling-transition localization and refinement;
- synchronization and transition diagnostics;
- perturbation and resilience tests;
- broad coupling-parameter surveys;
- fixed-coupling temporal-switching analysis; and
- finite-time largest Lyapunov exponents.

### Physical P/M convention

The physical trajectory classifications are

| Regime | Detector-drift orientation |
| --- | --- |
| **P** | `(D1, D2) = (+, -)` |
| **M** | `(D1, D2) = (-, +)` |

P and M are reproducible **finite-time phase-slipping regimes**. They should not automatically be interpreted as distinct asymptotic attractors. Likewise, the repository does not treat every P/M transition as a bifurcation, does not claim a proven fractal basin boundary, and does not treat a positive finite-time Lyapunov exponent as mathematical proof of an asymptotic chaotic invariant set.

---

## Repository layout

```text
Coupled-PLL-Dynamics/
|
|-- run_pipeline.py          Top-level reproducibility pipeline
|-- scripts/                 Retained numerical analysis programs
|-- figures/
|   |-- scripts/             Manuscript and supplementary figure builders
|   `-- output/              Generated figures (not tracked)
|-- results/                 Generated numerical outputs (not tracked)
|-- docs/
|   |-- PIPELINE.md          Detailed pipeline documentation
|   `-- REPOSITORY_AUDIT.md  Publication-repository audit
|-- requirements.txt         Minimal Python dependencies
|-- environment.yml          Conda environment
|-- CITATION.cff             Citation metadata
`-- README.md                Repository overview and quick-start guide
```

The numbered P-series scripts are intentionally retained with their historical identifiers so that computational provenance can be traced back to the study workflow. Missing numbers correspond to exploratory or superseded programs deliberately removed from the publication repository; see `docs/REPOSITORY_AUDIT.md`.

---

# Quick start

## 1. Clone the repository

```bash
git clone https://github.com/hamiddi/Coupled-PLL-Dynamics.git
cd Coupled-PLL-Dynamics
```

## 2. Create the environment

### Conda

```bash
conda env create -f environment.yml
conda activate coupled-pll-dynamics
```

### pip / virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Validate the installation

```bash
python run_pipeline.py --check
```

This should be the first pipeline command run on a new system. It checks the computational environment, repository structure, required dependencies, and retained scientific/figure programs before expensive calculations are started.

## 4. Inspect the workflow

```bash
python run_pipeline.py --list
```

## 5. Preview the complete study without running it

```bash
python run_pipeline.py --full --dry-run
```

## 6. Reproduce the complete study

```bash
python run_pipeline.py --full
```

> **Computational cost:** the full workflow includes continuation, refinement, large trajectory surveys, temporal-switching analyses, and Lyapunov calculations. Runtime can therefore be substantial.

---

# Reproducibility pipeline

`run_pipeline.py` is the recommended entry point for reproducing the study. It orchestrates the retained programs while leaving the individual scientific scripts available for direct inspection and execution.

## Common commands

| Goal | Command |
| --- | --- |
| Validate environment/repository | `python run_pipeline.py --check` |
| Show ordered workflow | `python run_pipeline.py --list` |
| Preview the complete run | `python run_pipeline.py --full --dry-run` |
| Run numerical analyses | `python run_pipeline.py --analysis` |
| Build figures from existing results/freezes | `python run_pipeline.py --figures` |
| Run analyses and final figures | `python run_pipeline.py --full` |
| Continue checkpoint-aware stages | `python run_pipeline.py --analysis --resume` |

### Run only part of the analysis sequence

A selected range can be executed without repeating the complete workflow. For example:

```bash
python run_pipeline.py --analysis --from p55 --to p64
```

This is useful for reproducing a particular manuscript stage after the required upstream outputs already exist.

### Recommended sequence for a new user

```bash
python run_pipeline.py --check
python run_pipeline.py --list
python run_pipeline.py --full --dry-run
python run_pipeline.py --full
```

For detailed pipeline behavior, dependencies, restart information, and stage organization, see `docs/PIPELINE.md`.

---

## Computational stages

The retained workflow is organized into the following scientific stages.

| Programs | Principal role |
| --- | --- |
| **P00-P04** | Model/Jacobian validation, rotating-frame validation, equilibrium/Hopf groundwork, and periodic-orbit/Floquet utilities |
| **P09, P11-P23** | Benchmark checks, Hopf verification, continuation, fold/Floquet diagnostics, fixed-coupling orbit analysis, and long-time verification |
| **P24-P51** | Local finite-time regime-selection geometry, progressive boundary refinement/validation, resilience, high-resolution verification, and uncertainty scaling |
| **P55-P64** | Corrected fixed-initial-condition coupling continuation, transition localization/refinement, synchronization diagnostics, re-entrant topology, robustness/resilience, trajectory mechanism, and manuscript-data freeze |
| **P65-P70** | Broad coupling survey, fixed-coupling temporal-switching verification, finite-time Lyapunov characterization, and final freeze/provenance checks |

Individual scripts may also be run directly from the repository root. For example:

```bash
python scripts/p00_model_validation.py
python scripts/p01_rotating_frame_validation.py
```

Some continuation and refinement programs intentionally require explicit upstream report paths. This prevents a program from silently consuming an incorrect historical result. Use:

```bash
python scripts/<script_name>.py --help
```

to inspect script-specific arguments.

---

## Manuscript data freezes

The workflow uses reproducibility freezes to separate expensive numerical computation from final manuscript visualization.

### P00-P63 freeze

After the required P00-P63 outputs are available:

```bash
python scripts/p64_reproducibility_manuscript_freeze.py --strict
python scripts/p64_v2_validation_provenance_update.py
```

### P65-P69 freeze

After the required P65-P69 outputs are available:

```bash
python scripts/p70_reproducibility_manuscript_freeze.py
```

These stages perform provenance and consistency checks and create normalized source tables used by the final figure builders. They do **not** perform new ODE integrations.

---

## Building the final figures

When the required frozen source data are available, all final manuscript figures can be rebuilt with:

```bash
python figures/scripts/make_all_figures.py
```

The figure workflow includes manuscript Figures **1-7**.

Supplementary figures are generated by:

```text
figures/scripts/figS01_temporal_switching_detail.py
figures/scripts/figS02_sensitivity_resilience.py
```

Use each supplementary script's `--help` option for its required inputs.

If the numerical results/freezes already exist, the pipeline provides the simpler entry point:

```bash
python run_pipeline.py --figures
```

Generated figures are written under:

```text
figures/output/
```

---

## Outputs and logs

Generated numerical results are written under:

```text
results/
```

Generated manuscript figures are written under:

```text
figures/output/
```

Pipeline execution logs are stored under:

```text
results/logs/
```

The pipeline records execution information such as commands, status, runtime, and Git revision information where available. Generated numerical outputs and figure files are intentionally excluded from routine Git tracking so that the repository remains code-focused.

---

## Reproducibility and interpretation

The repository is designed to preserve both **computational reproducibility** and **scientific provenance**. The retained scripts include validation, continuation, transition, robustness, global-coupling, temporal-switching, and final-freeze stages used in the study.

Several points are important when interpreting the outputs:

1. **P and M are finite-time detector-phase drift orientations.** Numerical cluster labels are not used as the physical definition of these regimes.
2. **A P/M change is not automatically a bifurcation.** Transition classifications must be interpreted together with the dynamical analyses reported in the manuscript.
3. **Finite-time Lyapunov exponents are finite-time diagnostics.** A positive value is not, by itself, a proof of an asymptotic chaotic invariant set.
4. **Generated results are not tracked by default.** Long calculations can therefore be reproduced locally without filling the Git history with large numerical output files.
5. **Historical script numbering is preserved intentionally.** Gaps in the P-series correspond to exploratory or superseded programs removed during the publication-repository audit.

For additional repository decisions, see `docs/REPOSITORY_AUDIT.md`.

---

## Citation

If you use this repository in research, please cite the associated manuscript and software repository.

Machine-readable citation metadata are provided in:

```text
CITATION.cff
```

The manuscript associated with this repository is:

> **Coupling-Induced Multistability, Re-entrant Dynamics, and Chaotic Temporal Switching in Mutually Coupled Third-Order Phase-Locked Loops**

The final DOI and archival release information can be added after publication/acceptance. A tagged GitHub release may also be archived with Zenodo to provide a persistent software DOI.

---

## License

No open-source license has been assigned yet. Until the authors select a license, normal copyright restrictions apply.

---

## Repository

**GitHub:** `hamiddi/Coupled-PLL-Dynamics`

**Primary entry point:** `run_pipeline.py`

**Detailed pipeline documentation:** `docs/PIPELINE.md`
