"""
config.py - the single place every script gets its paths and study settings from.

Nothing in this repository carries a hard-coded path or a real study date. Each
script does `from config import DATA_DIR, RAW_DIR, PROCESSED_DIR, S` and reads
everything else from `S` (the settings dictionary).

Paths (first match wins):
  --data-dir <folder>        command-line argument (removed from sys.argv here,
                             so scripts with their own arguments are not confused)
  T1D_DATA_DIR               environment variable
  <repo>/data/synthetic      the default: the synthetic sample shipped here

  RAW_DIR        = --raw-dir | T1D_RAW_DIR | DATA_DIR/raw
  PROCESSED_DIR  = --processed-dir | T1D_PROCESSED_DIR | DATA_DIR/processed
  FIGURES_DIR    = --figures-dir | T1D_FIGURES_DIR | PROCESSED_DIR/figures

Settings: DEFAULT_SETTINGS below describe the synthetic sample (year 2000
placeholder dates). A JSON file at DATA_DIR/settings.json (or the path in
T1D_SETTINGS) overrides any key. The real study's settings file, which carries
the real window boundaries, the fixed split dates and the person's own workout
relabel rule, lives OUTSIDE this repository and is never published.

Overrides table: S["overrides_csv"] points at a CSV with the dated corrections
(session overrides, workout relabels, real CGM gaps). The repository ships
data/overrides_example.csv with placeholder dates; the real file is private.

Raw export layout expected by build_dataset.py (see docs/data_sources.md):
  RAW_DIR/glooko/*.zip          Glooko CSV exports (one zip per data window)
  RAW_DIR/clarity/*.csv         Dexcom Clarity exports
  RAW_DIR/glooko_api/basal_states_*.json, modes_*.json   pump timeline pulls
  RAW_DIR/health/*.zip          Health export (export.xml inside)
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _take_arg(name):
    """Pop `--name value` (or --name=value) from sys.argv; return value or None."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            val = argv[i + 1]
            del argv[i:i + 2]
            return val
        if a.startswith(name + "="):
            del argv[i]
            return a.split("=", 1)[1]
    return None


_data = _take_arg("--data-dir") or os.environ.get("T1D_DATA_DIR")
DATA_DIR = Path(_data).expanduser().resolve() if _data else REPO_ROOT / "data" / "synthetic"
_raw = _take_arg("--raw-dir") or os.environ.get("T1D_RAW_DIR")
RAW_DIR = Path(_raw).expanduser().resolve() if _raw else DATA_DIR / "raw"
_proc = _take_arg("--processed-dir") or os.environ.get("T1D_PROCESSED_DIR")
PROCESSED_DIR = Path(_proc).expanduser().resolve() if _proc else DATA_DIR / "processed"
_fig = _take_arg("--figures-dir") or os.environ.get("T1D_FIGURES_DIR")
FIGURES_DIR = Path(_fig).expanduser().resolve() if _fig else PROCESSED_DIR / "figures"
_settings_file = _take_arg("--settings") or os.environ.get("T1D_SETTINGS")
SETTINGS_FILE = Path(_settings_file).expanduser().resolve() if _settings_file else DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    # data windows: [start (inclusive), end (exclusive), window number]. Windows are
    # contiguous blocks of the 5-minute grid; nothing is interpolated between them.
    "windows": [["2000-01-03", "2000-01-17", 1]],
    # a window boundary that later rows must never see across (IOB, workouts,
    # past-hour completeness); null = no fence
    "fence_end": None,
    # a window that is quarantined: written to the aligned table, never modelled
    "quarantine_window": None,
    # 5-minute forecaster split (fixed once, never moved): rows at or after
    # test_start are the test set; validation (early stopping) starts at val_start
    "test_start": "2000-01-14 00:00",
    "val_start": "2000-01-12 00:00",
    # session-level split: sessions starting at or before this are `explore`,
    # later ones `prospective` (never printed, never fitted on)
    "explore_last_start": "2000-01-16 23:55",
    # clock-change days excluded from the clean set
    "dst_days": [],
    # pump active insulin time (hours) for the linear IOB columns
    "dia_hours": 2.0,
    # watch "Soccer" workouts are football only inside this weekday/clock rule
    # (Monday = 0; hours inclusive); every other Soccer workout is cricket.
    # null = no rule: every football-typed (Soccer) workout is football. The
    # real rule is private and never published.
    "football_rule": None,
    # indoor cricket season (month, day) inclusive bounds, wrapping the year end
    "indoor_season": {"start": [10, 1], "end": [4, 14]},
    # forward-chaining folds for the session models: "month" (calendar month of
    # the session start) or "week" (7-day blocks from the first window start)
    "fold_by": "week",
    # rolling-origin CV for the low classifier (cv_classifier_v2.py)
    "cv_min_train_days": 5,
    "cv_block_days": 3,
    "cv_min_block_days": 2,
    # evaluation blocks for cv_ablation_exercise.py: [start, end) pairs
    "cv_ablation_folds": [["2000-01-10", "2000-01-12"], ["2000-01-12", "2000-01-14"]],
    # anticipation.py decision grid (null = the whole explore period) and clock range
    "anticipation_grid_start": None,
    "anticipation_grid_end": None,
    "grid_clock": [5, 23],
    # the person's usual training slots, for the false-alarm cost framing only:
    # list of {"weekday": 0-6, "from_hour": h, "season": "any"|"winter"|"summer"}
    # null or [] = none set: anticipation.py reports no usual-day split. The
    # real list is private and never published.
    "usual_training_times": [],
    # dated corrections (see data/overrides_example.csv); relative paths are
    # resolved against DATA_DIR, then the repository root
    "overrides_csv": "overrides.csv",
}


def _load_settings():
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            s.update(json.load(fh))
        s["_source"] = str(SETTINGS_FILE)
    # the two private keys may be null in a settings file: null means "none set"
    if s.get("usual_training_times") is None:
        s["usual_training_times"] = []
    if not s.get("football_rule"):
        s["football_rule"] = None
    else:
        s["_source"] = "defaults (synthetic profile)"
    return s


S = _load_settings()


def ts(key):
    """Setting as a pandas Timestamp (None stays None)."""
    v = S.get(key)
    return pd.Timestamp(v) if v is not None else None


def windows():
    """[(start Timestamp, end Timestamp, number), ...] in time order."""
    return sorted([(pd.Timestamp(a), pd.Timestamp(b), int(n)) for a, b, n in S["windows"]],
                  key=lambda w: w[0])


def data_start():
    return windows()[0][0]


def data_end():
    return windows()[-1][1]


def overrides_path():
    p = Path(S["overrides_csv"])
    if p.is_absolute():
        return p
    for base in (DATA_DIR, REPO_ROOT / "data", REPO_ROOT):
        if (base / p).exists():
            return base / p
    return REPO_ROOT / "data" / "overrides_example.csv"


def load_overrides(kind=None):
    """The overrides table (comment lines start with #). kind filters the `kind`
    column: session_override | workout_relabel | real_gap."""
    p = overrides_path()
    cols = ["kind", "start", "end", "new_cat2", "new_duration_min", "note"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    O = pd.read_csv(p, comment="#", skip_blank_lines=True, dtype=str)
    O = O.reindex(columns=cols)
    O["kind"] = O["kind"].str.strip()
    O["start"] = pd.to_datetime(O["start"].str.strip())
    O["end"] = pd.to_datetime(O["end"].str.strip()) if O["end"].notna().any() else pd.NaT
    O["new_duration_min"] = pd.to_numeric(O["new_duration_min"], errors="coerce")
    if kind:
        O = O[O["kind"] == kind]
    return O.reset_index(drop=True)


def processed(name):
    return PROCESSED_DIR / name


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def describe():
    return (f"DATA_DIR={DATA_DIR}\nRAW_DIR={RAW_DIR}\nPROCESSED_DIR={PROCESSED_DIR}\n"
            f"FIGURES_DIR={FIGURES_DIR}\nsettings: {S['_source']}\noverrides: {overrides_path()}")


if __name__ == "__main__":
    print(describe())
    print(json.dumps({k: v for k, v in S.items() if not k.startswith("_")}, indent=2, default=str))
