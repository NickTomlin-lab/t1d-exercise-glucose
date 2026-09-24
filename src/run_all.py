"""
run_all.py - run the session-level pipeline end to end and say where every
output lands. Pure Python (no shell), so it runs the same on every OS.

Default: the synthetic sample (data/synthetic). Steps:
  1. make_synthetic.py      -> aligned_5min.csv, workouts_clean.csv (SYNTHETIC)
  2. build_features_v2.py   -> aligned_5min_v2.csv (curved IOB, COB)
  3. build_sessions.py      -> sessions.csv
  4. build_controls.py      -> control_windows.csv
  5. fit_session_models.py  -> pooled_windows.csv + model report
  6. lead_time.py           -> lead_time_summary.csv + report
  7. anticipation.py        -> anticipation_summary.csv + report
Each step's stdout is saved as PROCESSED_DIR/<step>_report.txt.

Options:
  --data-dir <folder>   run on another data folder (its settings.json applies)
  --skip-synthetic      do not regenerate the aligned table (real data: run
                        build_dataset.py yourself first; it needs the raw exports)
  --forecaster          also run the 5-minute forecaster scripts (baselines,
                        LightGBM point model, quantile band, low classifier,
                        band tuning, final verdict, curved-IOB CV, ablations)

Real data: python run_all.py --data-dir <private folder> --skip-synthetic
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SESSION_STEPS = [
    ("make_synthetic", "make_synthetic.py", ["aligned_5min.csv", "workouts_clean.csv", "train_5min.csv", "test_5min.csv"]),
    ("build_features_v2", "build_features_v2.py", ["aligned_5min_v2.csv"]),
    ("build_sessions", "build_sessions.py", ["sessions.csv"]),
    ("build_controls", "build_controls.py", ["control_windows.csv"]),
    ("fit_session_models", "fit_session_models.py", ["pooled_windows.csv"]),
    ("lead_time", "lead_time.py", ["lead_time_summary.csv"]),
    ("anticipation", "anticipation.py", ["anticipation_summary.csv"]),
]
FORECASTER_STEPS = [
    ("evaluate_baselines", "evaluate_baselines.py", ["baseline_results.csv"]),
    ("train_lgbm", "train_lgbm.py", ["test_predictions.csv"]),
    ("train_quantile", "train_quantile.py", ["test_band_predictions.csv"]),
    ("train_hypo_classifier", "train_hypo_classifier.py", []),
    ("tune_band", "tune_band.py", []),
    ("final_test_verdict", "final_test_verdict.py", ["final_test_outputs.csv"]),
    ("cv_classifier_v2", "cv_classifier_v2.py", []),
    ("cv_ablation_exercise", "cv_ablation_exercise.py", []),
    ("ablation_exercise", "ablation_exercise.py", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--forecaster", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.data_dir:
        env["T1D_DATA_DIR"] = str(Path(args.data_dir).resolve())
    # ask config where things land (in a subprocess so the env applies)
    probe = subprocess.run([sys.executable, "-c", "import config; print(config.PROCESSED_DIR)"],
                           cwd=HERE, env=env, capture_output=True, text=True, check=True)
    processed = Path(probe.stdout.strip())
    processed.mkdir(parents=True, exist_ok=True)
    print(f"processed outputs -> {processed}")

    steps = [s for s in SESSION_STEPS if not (args.skip_synthetic and s[0] == "make_synthetic")]
    if args.forecaster:
        steps += FORECASTER_STEPS
    for name, script, outputs in steps:
        report = processed / f"{name}_report.txt"
        t0 = time.time()
        print(f"\n=== {name} ===")
        with open(report, "w", encoding="utf-8") as fh:
            proc = subprocess.run([sys.executable, script], cwd=HERE, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print(open(report, encoding="utf-8").read()[-3000:])
            raise SystemExit(f"{name} failed (exit {proc.returncode}); see {report}")
        print(f"  ok in {time.time() - t0:.0f} s; report -> {report}")
        for o in outputs:
            p = processed / o
            print(f"  {'wrote' if p.exists() else 'MISSING'} {p}")
    print("\nall steps finished")


if __name__ == "__main__":
    main()
