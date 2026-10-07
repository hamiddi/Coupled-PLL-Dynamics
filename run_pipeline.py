#!/usr/bin/env python3
"""Reproducibility pipeline for the Coupled-PLL-Dynamics repository.

This runner orchestrates the retained publication scripts without duplicating or
changing their scientific calculations. Run it from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import py_compile
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
FIG_SCRIPTS = ROOT / "figures" / "scripts"
LOG_DIR = ROOT / "results" / "logs"

# Retained computational programs in publication-workflow order.
ANALYSIS = [
    "p00_model_validation.py", "p01_rotating_frame_validation.py",
    "p02_equilibrium_hopf_analysis.py", "p03_hopf_refinement.py",
    "p04_periodic_orbit_floquet.py", "p09_author_parameter_validation.py",
    "p11_author_hopf_verification.py", "p12_author_cycle_shooting.py",
    "p13_pseudo_arclength.py", "p14_extended_floquet.py",
    "p15_mode_tracking.py", "p16_fold_refinement.py",
    "p17_independent_fold.py", "p18_direct_fold.py",
    "p19_multiple_shooting.py", "p20_fold_genericity.py",
    "p21_fixed_coupling.py", "p22_attractor_survey.py",
    "p23_longtime_verification.py", "p24_basin_attraction.py",
    "p25_boundary_mapping.py", "p26_boundary_refinement.py",
    "p27_local_boundary_atlas.py", "p28_adaptive_angular_refinement.py",
    "p29_boundary_convergence.py", "p30_local_2d_boundary.py",
    "p31_adaptive_2d_convergence.py", "p32_boundary_continuation.py",
    "p33_expanded_boundary_search.py", "p34_geometry_validation.py",
    "p35_boundary_resolution.py", "p36_independent_boundary_convergence.py",
    "p37_boundary_interpolation_validation.py",
    "p38_prediction_uncertainty_validation.py",
    "p39_adaptive_boundary_validation.py", "p40_local_boundary_stress_test.py",
    "p41_spatial_error_validation.py",
    "p42_prospective_error_horizon_validation.py",
    "p43_geometry_based_boundary_validation.py",
    "p44_compact_geometry_validation.py", "p45_local_adaptive_uncertainty.py",
    "p46_signed_residual_correction.py", "p47_guarded_signed_correction.py",
    "p48_consolidated_validation.py", "p49_basin_geometry_resilience.py",
    "p50_high_resolution_boundary_verification.py",
    "p51_uncertainty_exponent.py",
    "p55_corrected_fixed_ic_coupling_continuation.py",
    "p56_coupling_transition_localization.py",
    "p57_high_resolution_coupling_threshold_refinement.py",
    "p58_synchronization_transition_mechanism.py",
    "p59_critical_transition_early_warning.py",
    "p60_coupling_axis_topology_audit.py", "p61_reentrant_robustness.py",
    "p62_local_basin_resilience.py", "p63_trajectory_transition_mechanism.py",
    "p64_reproducibility_manuscript_freeze.py",
    "p64_v2_validation_provenance_update.py",
    "p65_global_bifurcation_multistability.py",
    "p66_v2_intermittency_verification.py",
    "p67_regime_switching_discrimination.py",
    "p68_temporal_switching_verification.py",
    "p69_lyapunov_temporal_switching.py",
    "p70_reproducibility_manuscript_freeze.py",
]

# Scripts with checkpoint-aware --resume support.
RESUMABLE = {
    *[f"p{i:02d}" for i in range(24, 52)],
    *[f"p{i:02d}" for i in range(55, 64)],
    "p65", "p66", "p69",
}

# Repository-specific argument corrections/normalization.  P67's retained
# source has a historical default pointing at the superseded P66 directory;
# P70 expects the normalized v2 P67 output directory.
EXTRA_ARGS = {
    "p64_reproducibility_manuscript_freeze.py": ["--strict"],
    "p67_regime_switching_discrimination.py": [
        "--p66-dir", "results/author_intermittency_verification_v2",
        "--output-dir", "results/author_regime_switching_discrimination_v2",
    ],
}

FIGURES = [
    ("fig01_model_validation_workflow.py", "figure_01", "p64"),
    ("fig02_local_invariant_structure.py", "figure_02", "p64"),
    ("fig03_pm_phase_slipping.py", "figure_03", "p64"),
    ("fig04_regime_selection_geometry.py", "figure_04", "p64"),
    ("fig05_transition_dynamics.py", "figure_05", "p64"),
    ("fig06_global_coupling_landscape.py", "figure_06", "p70"),
    ("fig07_temporal_switching_lyapunov.py", "figure_07", "p70"),
    ("figS01_temporal_switching_detail.py", "figure_S1", "p70"),
    ("figS02_sensitivity_resilience.py", "figure_S2", "p64"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def check_repository() -> bool:
    print("[check] repository:", ROOT)
    ok = True
    required = ["numpy", "scipy", "pandas", "matplotlib", "sympy"]
    for package in required:
        try:
            mod = importlib.import_module(package)
            print(f"[check] {package:<10} {getattr(mod, '__version__', 'available')}")
        except Exception as exc:
            ok = False
            print(f"[FAIL ] dependency {package}: {exc}")

    files = [SCRIPTS / x for x in ANALYSIS]
    files += [FIG_SCRIPTS / x[0] for x in FIGURES]
    files += [FIG_SCRIPTS / "figure_style.py", FIG_SCRIPTS / "make_all_figures.py"]
    for path in files:
        if not path.is_file():
            ok = False
            print(f"[FAIL ] missing: {path.relative_to(ROOT)}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            ok = False
            print(f"[FAIL ] syntax: {path.relative_to(ROOT)}: {exc}")
    if ok:
        print(f"[check] PASS: dependencies available and {len(files)} Python files compile.")
    return ok


def run_one(label: str, cmd: list[str], log_handle, dry_run: bool) -> dict:
    shown = " ".join(cmd)
    print(f"\n[run] {label}\n      {shown}", flush=True)
    if dry_run:
        return {"step": label, "status": "DRY-RUN", "seconds": 0.0, "command": shown}
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=os.environ.copy(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()
    rc = proc.wait()
    elapsed = time.perf_counter() - start
    status = "PASS" if rc == 0 else "FAIL"
    print(f"[{status}] {label} ({elapsed:.1f} s)", flush=True)
    return {"step": label, "status": status, "returncode": rc,
            "seconds": round(elapsed, 3), "command": shown}


def selected_analysis(start: str | None, stop: str | None) -> list[str]:
    jobs = ANALYSIS[:]
    if start:
        hits = [i for i, f in enumerate(jobs) if f.startswith(start)]
        if not hits:
            raise SystemExit(f"Unknown --from step: {start}")
        jobs = jobs[hits[0]:]
    if stop:
        hits = [i for i, f in enumerate(jobs) if f.startswith(stop)]
        if not hits:
            raise SystemExit(f"Unknown --to step: {stop}")
        jobs = jobs[:hits[0] + 1]
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Coupled-PLL-Dynamics reproducibility workflow.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate dependencies, files, and Python syntax only.")
    mode.add_argument("--analysis", action="store_true", help="Run the retained P-series computational workflow.")
    mode.add_argument("--figures", action="store_true", help="Build Figures 1-7 and Supplementary Figures S1-S2 from existing freezes/results.")
    mode.add_argument("--full", action="store_true", help="Run all retained analyses, then build all manuscript and supplementary figures.")
    mode.add_argument("--list", action="store_true", help="List the ordered pipeline steps without running them.")
    ap.add_argument("--resume", action="store_true", help="Pass --resume to checkpoint-aware computational stages.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    ap.add_argument("--from", dest="start", metavar="PXX", help="Start analysis at a retained P-series step, e.g. p55.")
    ap.add_argument("--to", dest="stop", metavar="PXX", help="Stop analysis after a retained P-series step, e.g. p64.")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue after a failed stage (not recommended for final reproduction).")
    args = ap.parse_args()

    if args.list:
        print("Analysis steps:")
        for i, f in enumerate(ANALYSIS, 1): print(f"  {i:02d}. {f}")
        print("Figure steps:")
        for f, out, _ in FIGURES: print(f"  - {f} -> figures/output/{out}")
        return 0

    if not check_repository():
        return 2
    if args.check:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_log = LOG_DIR / f"pipeline_{stamp}.log"
    json_log = LOG_DIR / f"pipeline_{stamp}.json"
    csv_log = LOG_DIR / f"pipeline_{stamp}.csv"
    records: list[dict] = []
    metadata = {
        "started_utc": utc_now(), "python": sys.version,
        "executable": sys.executable, "git_commit": git_commit(),
        "mode": "full" if args.full else ("analysis" if args.analysis else "figures"),
        "resume": args.resume, "dry_run": args.dry_run,
    }

    p64 = ROOT / "results/author_reproducibility_manuscript_freeze/manuscript_freeze"
    p70 = ROOT / "results/author_reproducibility_manuscript_freeze_v2/manuscript_freeze"

    with text_log.open("w", encoding="utf-8") as log:
        log.write(json.dumps(metadata, indent=2) + "\n\n")
        if args.analysis or args.full:
            for script in selected_analysis(args.start, args.stop):
                cmd = [sys.executable, str(SCRIPTS / script)] + EXTRA_ARGS.get(script, [])
                prefix = script.split("_", 1)[0]
                if args.resume and prefix in RESUMABLE:
                    cmd.append("--resume")
                rec = run_one(script, cmd, log, args.dry_run)
                records.append(rec)
                if rec["status"] == "FAIL" and not args.continue_on_error:
                    break

        failed = any(r["status"] == "FAIL" for r in records)
        if (args.figures or args.full) and (not failed or args.continue_on_error):
            for script, out_name, freeze_key in FIGURES:
                freeze = p64 if freeze_key == "p64" else p70
                cmd = [sys.executable, str(FIG_SCRIPTS / script),
                       "--freeze-dir", str(freeze),
                       "--output-dir", str(ROOT / "figures/output" / out_name)]
                if script == "figS01_temporal_switching_detail.py":
                    cmd += ["--project-root", str(ROOT)]
                rec = run_one(script, cmd, log, args.dry_run)
                records.append(rec)
                if rec["status"] == "FAIL" and not args.continue_on_error:
                    break

    metadata["finished_utc"] = utc_now()
    metadata["status"] = "PASS" if not any(r["status"] == "FAIL" for r in records) else "FAIL"
    metadata["steps"] = records
    json_log.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if records:
        with csv_log.open("w", newline="", encoding="utf-8") as fh:
            fields = ["step", "status", "returncode", "seconds", "command"]
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(records)

    print("\n" + "=" * 72)
    print("PIPELINE", metadata["status"])
    print("Git commit:", metadata["git_commit"])
    print("Log:       ", text_log.relative_to(ROOT))
    print("JSON:      ", json_log.relative_to(ROOT))
    print("CSV:       ", csv_log.relative_to(ROOT))
    print("=" * 72)
    return 0 if metadata["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
