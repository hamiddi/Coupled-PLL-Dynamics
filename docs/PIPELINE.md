# Reproducibility pipeline

`run_pipeline.py` is the top-level orchestration entry point for the retained publication code. It does not reimplement the PLL model or numerical algorithms; it executes the retained scripts in their publication-workflow order, records provenance and runtime, and can build the final figures from the manuscript freezes.

## Recommended use

From the repository root, first validate the environment and all retained Python files:

```bash
python run_pipeline.py --check
```

Inspect the execution order without running expensive calculations:

```bash
python run_pipeline.py --list
python run_pipeline.py --full --dry-run
```

Run all retained computational analyses:

```bash
python run_pipeline.py --analysis
```

Run the analyses and then build Figures 1–7 and Supplementary Figures S1–S2:

```bash
python run_pipeline.py --full
```

If authoritative numerical results and the P64/P70 freezes already exist, rebuild only the figures:

```bash
python run_pipeline.py --figures
```

## Restarting long calculations

Many later stages maintain checkpoints. To continue a previously interrupted workflow, use:

```bash
python run_pipeline.py --analysis --resume
```

The runner passes `--resume` only to scripts that expose that option. It does not delete or overwrite checkpoints automatically.

A partial workflow can be selected by retained script prefix:

```bash
python run_pipeline.py --analysis --from p55 --to p64
```

Use partial execution only when all required upstream outputs already exist.

## Logging and provenance

Every execution writes timestamped files under `results/logs/`:

- `pipeline_YYYYMMDD_HHMMSS.log` — combined console output;
- `pipeline_YYYYMMDD_HHMMSS.json` — environment, Git commit, commands, status, and runtime;
- `pipeline_YYYYMMDD_HHMMSS.csv` — compact per-stage execution summary.

Generated logs are excluded from version control by the repository's `results/*` rule.

## Important finite-time interpretation

The pipeline preserves the manuscript convention that P and M are finite-time detector-phase drift orientations: P = `(+, -)` and M = `(-, +)`. Successful completion of the numerical pipeline does not by itself convert these finite-time classifications into claims of distinct asymptotic attractors.

## P67 compatibility normalization

The retained `p67_regime_switching_discrimination.py` contains a historical default path referring to the superseded P66 output directory. The pipeline explicitly supplies the retained P66-v2 directory and writes P67 to the directory consumed by P70. This changes orchestration only; it does not alter P67's numerical calculations.
