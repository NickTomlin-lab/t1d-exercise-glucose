"""
make_public_tables.py - the published session and control tables.

Takes the private sessions.csv and control_windows.csv (PROCESSED_DIR) and
writes results/sessions_public.csv and results/controls_public.csv:

  - usable explore sessions only (start on or before the explore cut-off);
    nothing from the prospective / quarantined period;
  - session_uid: a random 6-character id, NOT chronological (rows shuffled
    with a fixed seed); the private id -> uid map is written OUTSIDE the
    repository (--map-out) for the author only;
  - no start, end, hour, weekday, window, session_id, sensor flag or
    overridden column; hr_mean_session rounded to 5 bpm; steps_session to 100;
  - fold: the forward-chaining fold index of the session start (0 = the first
    month of data), kept because fit_session_models.py cannot re-run its
    time-ordered evaluation without it; it carries no day or clock time.

The published pooled table (results/pooled_windows_public.csv) is then rebuilt
from these two files by fit_session_models.py, not by this script.
"""
import argparse
import string

import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR, REPO_ROOT
from fit_session_models import fold_of   # noqa: E402

SEED = 20260923
SESSION_COLS = ["session_uid", "fold", "cat2", "aerobic", "duration_min", "g0", "slope30", "slope15",
                "min_prev2h", "mins_below4_prev24h", "bolus_iob_u", "est_basal_iob_u", "insulin_total_4h",
                "bolus_sum_2h", "bolus_sum_4h", "carbs_sum_4h", "mins_since_meal", "meal_band",
                "activity_at_start", "activity_lead_min", "hr_mean_session", "steps_session",
                "low_during", "low_post2h", "low_any", "low_late", "below3_any", "first_low_min", "nadir"]
CONTROL_COLS = ["session_uid", "cat2", "duration_min", "g0", "slope30", "mins_since_meal", "meal_band",
                "bolus_iob_u", "est_basal_iob_u", "insulin_total_4h", "low_any", "low_late", "below3_any",
                "nadir", "outcome_missing_frac", "usable"]


def make_uids(n, rng):
    alphabet = string.ascii_lowercase + string.digits
    uids = set()
    while len(uids) < n:
        uids.add("".join(rng.choice(list(alphabet), 6)))
    return sorted(uids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--map-out", help="private csv: session_id,session_uid (write it outside the repository)")
    args = ap.parse_args()
    out_dir = config.Path(args.out_dir) if hasattr(config, "Path") else __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Sess = pd.read_csv(PROCESSED_DIR / "sessions.csv", parse_dates=["start", "end"])
    C = pd.read_csv(PROCESSED_DIR / "control_windows.csv", parse_dates=["start", "end"])
    cutoff = config.ts("explore_last_start")
    U = Sess[(Sess["session_split"] == "explore") & (Sess["usable"] == 1) & (Sess["start"] <= cutoff)].copy()
    assert (U["start"] <= cutoff).all()
    C = C[C["session_id"].isin(U["session_id"]) & (C["start"] <= cutoff)].copy()

    rng = np.random.default_rng(SEED)
    uids = make_uids(len(U), rng)
    rng.shuffle(uids)
    U["session_uid"] = uids
    U["fold"] = fold_of(U["start"]).astype(int)
    U["hr_mean_session"] = (U["hr_mean_session"] / 5).round() * 5
    U["steps_session"] = (U["steps_session"] / 100).round() * 100
    U["mins_since_meal"] = U["mins_since_meal"].clip(upper=1440)
    pub = U[SESSION_COLS].copy()
    pub = pub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)      # shuffle rows
    for c in pub.columns:
        if pub[c].dtype.kind == "f":
            pub[c] = pub[c].round(4)
    pub.to_csv(out_dir / "sessions_public.csv", index=False, lineterminator=chr(10))

    C["session_uid"] = C["session_id"].map(U.set_index("session_id")["session_uid"])
    C["mins_since_meal"] = C["mins_since_meal"].clip(upper=1440)
    C["_ord"] = np.arange(len(C))                       # nearest-date-first order inside a set is kept
    cpub = C.sort_values(["session_uid", "_ord"], kind="stable")[CONTROL_COLS].reset_index(drop=True)
    for c in cpub.columns:
        if cpub[c].dtype.kind == "f":
            cpub[c] = cpub[c].round(4)
    cpub.to_csv(out_dir / "controls_public.csv", index=False, lineterminator=chr(10))

    if args.map_out:
        U[["session_id", "session_uid"]].to_csv(args.map_out, index=False)
    print(f"sessions_public.csv: {len(pub)} usable explore sessions, columns {list(pub.columns)}")
    print(f"controls_public.csv: {len(cpub)} control windows for {cpub['session_uid'].nunique()} sessions")
    print(f"fold values: {sorted(pub['fold'].unique().tolist())}; sessions per fold "
          f"{pub['fold'].value_counts().sort_index().to_dict()}")
    print("cat2:", pub["cat2"].value_counts().to_dict())


if __name__ == "__main__":
    main()
