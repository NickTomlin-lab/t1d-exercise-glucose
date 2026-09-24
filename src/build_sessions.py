"""
build_sessions.py - one row per exercise session from PROCESSED_DIR/aligned_5min.csv.

Personal glucose-analysis project. All glucose in mmol/L. Analysis only:
nothing here informs a dosing decision. Reads aligned_5min.csv (never
modified) and writes sessions.csv, then prints a QC report on the `explore`
sessions only. No models are fitted here.

Session definition:
  1. A workout run = consecutive 5-min rows with workout_min > 0 and the same
     workout_cat.
  2. Keep only gym, cricket, football, cycling (walking ignored).
  3. Runs of the same category separated by an idle gap of <= 30 min are
     merged into one session; n_parts = number of runs merged.
  4. Never merge across a window boundary.

Rules carried over the project's revisions:
  - usable = missing <= 0.20 in start..end+2h, g0 present, no real_gap /
    dst_day in that span. pod_change (any gap in the pump-state record) does
    not affect usable; usable_v1 keeps the older rule for comparison.
  - cat2 splits cricket into cricket_indoor / cricket_outdoor by the indoor
    season in settings (start date only); gym, football, cycling unchanged.
  - aerobic: 1 for cricket_indoor, cricket_outdoor, football, cycling; 0 for
    gym. A category with a single session never gets its own indicator in a
    model; it is carried by aerobic.
  - sessions shorter than MIN_DURATION_MIN on the grid get usable = 0 and
    flag_reason too_short.
  - Session overrides (kind = session_override in the overrides table) are
    applied AFTER sessions are formed: the matching session gets the new
    cat2, the new duration and a recomputed end; every feature and outcome
    is then computed from the corrected window. Column overridden marks them.
  - Feature code lives in compute_features() so build_controls.py applies
    exactly the same definitions to control windows.

Split (fixed, never moved): explore = start <= settings explore_last_start;
prospective = later. Prospective rows are written but no outcome column of
theirs is printed.

Conventions:
  - "Before start" means rows with t < start; "at or before" includes start.
  - A window of N minutes before start is the N/5 rows with t in
    [start - N min, start), selected by timestamp, so nothing bleeds across
    a gap between windows.
  - "The 2 h after end" is rows with t in (end, end + 120 min]; end + 2 h to
    end + 12 h is rows with t in (end + 120, end + 720].
  - Rows that do not exist (past the end of the data) count as missing in
    outcome_missing_frac.
"""
import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR, S

IN_PATH = PROCESSED_DIR / "aligned_5min.csv"
OUT_PATH = PROCESSED_DIR / "sessions.csv"
WK_PATH = PROCESSED_DIR / "workouts_clean.csv"     # raw watch workouts, for n_workouts only

KEEP_CATS = ["gym", "cricket", "football", "cycling"]
AEROBIC = {"cricket_indoor": 1, "cricket_outdoor": 1, "football": 1, "cycling": 1, "gym": 0}
MIN_DURATION_MIN = 15
MERGE_GAP_MIN = 30
STEP = pd.Timedelta(minutes=5)
EXPLORE_LAST_START = config.ts("explore_last_start")
LOW = 4.0
VERY_LOW = 3.0
USABLE_FLAGS = ["real_gap", "dst_day"]
USABLE_FLAGS_V1 = ["real_gap", "dst_day", "pod_change"]
INDOOR = S["indoor_season"]

OUTCOME_COLS = ["low_during", "low_post2h", "low_any", "low_late", "below3_any",
                "first_low_min", "nadir", "outcome_missing_frac", "usable"]


def mins(td):
    return td / pd.Timedelta(minutes=1)


# --------------------------------------------------------------------------
# Shared helpers (imported by build_controls.py, lead_time.py, anticipation.py)
# --------------------------------------------------------------------------
def load_aligned(path=IN_PATH):
    """aligned_5min.csv indexed by row, timestamps parsed, flag columns as bool."""
    df = pd.read_csv(path, parse_dates=["t"], low_memory=False)
    df = df.sort_values("t").reset_index(drop=True)
    assert df["t"].is_unique, "timestamps must be unique"
    for c in ["real_gap", "dst_day", "pod_change"]:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    return df


def rows_between(g, a, b, inclusive="both"):
    if inclusive == "both":
        return g.loc[a:b]
    if inclusive == "left":      # [a, b)
        return g.loc[a:b - STEP]
    if inclusive == "right":     # (a, b]
        return g.loc[a + STEP:b]
    raise ValueError(inclusive)


def meal_band(m):
    if m <= 120:
        return "0-2h"
    if m <= 240:
        return "2-4h"
    if m <= 360:
        return "4-6h"
    return "6h+"


def prevlow_band(m):
    if m == 0:
        return "none"
    if m <= 30:
        return "1-30"
    if m <= 90:
        return "31-90"
    return "90+"


def is_indoor_season(start):
    """Indoor cricket season by (month, day), wrapping the year end."""
    md = (start.month, start.day)
    a, b = tuple(INDOOR["start"]), tuple(INDOOR["end"])
    return (md >= a or md <= b) if a > b else (a <= md <= b)


def cat2_of(cat, start):
    if cat != "cricket":
        return cat
    return "cricket_indoor" if is_indoor_season(start) else "cricket_outdoor"


def compute_features(g, st, en):
    """All per-window features and outcomes for a window [st, en] on the
    5-min grid g (timestamp index). Identical for sessions and controls."""
    f = {}
    back15 = rows_between(g, st - pd.Timedelta(minutes=15), st)["glucose"].dropna()
    if len(back15):
        g0_t = back15.index[-1]
        g0 = float(back15.iloc[-1])
    else:
        g0_t, g0 = None, np.nan
    f["g0"] = g0

    def gl_at(t):
        return float(g.at[t, "glucose"]) if t in g.index else np.nan

    f["slope30"] = g0 - gl_at(g0_t - pd.Timedelta(minutes=30)) if g0_t is not None else np.nan
    f["slope15"] = g0 - gl_at(g0_t - pd.Timedelta(minutes=15)) if g0_t is not None else np.nan

    prev2h = rows_between(g, st - pd.Timedelta(hours=2), st, "left")
    f["min_prev2h"] = prev2h["glucose"].min()
    prev24h = rows_between(g, st - pd.Timedelta(hours=24), st, "left")
    f["mins_below4_prev24h"] = 5.0 * (prev24h["glucose"] < LOW).sum()
    f["any_low_prev24h"] = int(f["mins_below4_prev24h"] > 0)
    f["prevlow_band"] = prevlow_band(f["mins_below4_prev24h"])

    at_start = g.loc[st]
    f["bolus_iob_u"] = float(at_start["bolus_iob_u"])
    f["est_basal_iob_u"] = float(at_start["est_basal_iob_u"])
    for h in (1, 2, 3, 4):
        w = rows_between(g, st - pd.Timedelta(hours=h), st, "left")
        f[f"bolus_sum_{h}h"] = float(w["bolus_u"].sum())
        if h in (2, 4):
            f[f"carbs_sum_{h}h"] = float(w["carbs_g"].sum())
    w4 = rows_between(g, st - pd.Timedelta(hours=4), st, "left")
    f["insulin_total_4h"] = float(w4["bolus_u"].sum() + w4["est_basal_u"].sum())

    prev = rows_between(g, st - pd.Timedelta(minutes=1440), st, "left")
    last_bolus = prev.index[prev["bolus_u"] > 0]
    f["mins_since_bolus"] = min(1440.0, mins(st - last_bolus[-1])) if len(last_bolus) else 1440.0
    last_meal = prev.index[prev["carbs_g"] > 0]
    f["mins_since_meal"] = min(1440.0, mins(st - last_meal[-1])) if len(last_meal) else 1440.0
    f["meal_band"] = meal_band(f["mins_since_meal"])
    f["fed_2h"] = int(f["mins_since_meal"] <= 120)
    f["fed_4h"] = int(f["mins_since_meal"] <= 240)
    f["correction_only_4h"] = int(f["bolus_sum_4h"] > 0 and f["carbs_sum_4h"] == 0)

    f["activity_at_start"] = int(at_start["frac_activity"] > 0)
    lead = 0
    t = st - STEP
    while t in g.index and g.at[t, "frac_activity"] > 0:
        lead += 5
        t -= STEP
    f["activity_lead_min"] = lead
    during = rows_between(g, st, en)
    f["activity_any_during"] = int((during["frac_activity"] > 0).any())
    prev60 = rows_between(g, st - pd.Timedelta(minutes=60), st, "left")
    f["suspend_frac_prev60"] = float(prev60["frac_suspend"].mean())

    f["hr_mean_session"] = float(during["hr_mean"].mean()) if during["hr_mean"].notna().any() else np.nan
    f["steps_session"] = float(during["steps"].sum())

    post2h = rows_between(g, en, en + pd.Timedelta(hours=2), "right")
    late = rows_between(g, en + pd.Timedelta(hours=2), en + pd.Timedelta(hours=12), "right")
    span = rows_between(g, st, en + pd.Timedelta(hours=2))
    f["low_during"] = int((during["glucose"] < LOW).any())
    f["low_post2h"] = int((post2h["glucose"] < LOW).any())
    f["low_any"] = int(f["low_during"] or f["low_post2h"])
    f["low_late"] = int((late["glucose"] < LOW).any())
    f["below3_any"] = int((span["glucose"] < VERY_LOW).any())
    lows = span.index[span["glucose"] < LOW]
    f["first_low_min"] = mins(lows[0] - st) if len(lows) else np.nan
    f["nadir"] = span["glucose"].min()
    expected_rows = int(mins(en + pd.Timedelta(hours=2) - st) / 5) + 1
    n_missing = span["glucose"].isna().sum() + (expected_rows - len(span))
    f["outcome_missing_frac"] = n_missing / expected_rows
    f["rows_missing_in_span"] = int(expected_rows - len(span))
    f["flag_in_span"] = int(span[USABLE_FLAGS].any().any())
    f["flag_reason"] = "+".join(c for c in USABLE_FLAGS_V1 if span[c].any()) or ""
    f["uncovered_min_span"] = 5.0 * float(span["frac_uncovered"].sum())
    f["long_uncovered"] = int(f["uncovered_min_span"] >= 30)
    base_ok = f["outcome_missing_frac"] <= 0.20 and not np.isnan(g0)
    f["usable"] = int(base_ok and not span[USABLE_FLAGS].any().any())
    f["usable_v1"] = int(base_ok and not span[USABLE_FLAGS_V1].any().any())
    return f


# --------------------------------------------------------------------------
# Overrides from the person's memory
# --------------------------------------------------------------------------
OVERRIDE_LOG = []


def apply_overrides(Sess):
    """Apply session_override rows of the overrides table. Match on session
    start (exact timestamp). A start that matches no session or more than one
    is an error, not a silent skip."""
    Sess = Sess.copy()
    Sess["overridden"] = 0
    OVERRIDE_LOG.clear()
    O = config.load_overrides("session_override")
    for _, o in O.iterrows():
        hit = Sess.index[Sess["start"] == o["start"]]
        if len(hit) != 1:
            raise ValueError(f"override start {o['start']} matched {len(hit)} sessions")
        i = hit[0]
        new_dur = float(o["new_duration_min"])
        if new_dur < 5 or new_dur % 5:
            raise ValueError(f"new_duration_min must be a positive multiple of 5, got {new_dur}")
        OVERRIDE_LOG.append(dict(
            session_id=Sess.at[i, "session_id"], start=o["start"],
            cat2_old=Sess.at[i, "cat2"], cat2_new=o["new_cat2"],
            duration_old=Sess.at[i, "duration_min"], duration_new=new_dur,
            end_old=Sess.at[i, "end"], end_new=Sess.at[i, "start"] + pd.Timedelta(minutes=new_dur - 5),
            note=o["note"]))
        Sess.at[i, "cat2"] = o["new_cat2"]
        Sess.at[i, "cat"] = str(o["new_cat2"]).split("_")[0]
        Sess.at[i, "duration_min"] = new_dur
        Sess.at[i, "end"] = Sess.at[i, "start"] + pd.Timedelta(minutes=new_dur - 5)
        Sess.at[i, "overridden"] = 1
    return Sess


# --------------------------------------------------------------------------
# Session table
# --------------------------------------------------------------------------
def build_sessions(df):
    g = df.set_index("t")
    is_wk = (df["workout_min"] > 0) & df["workout_cat"].isin(KEEP_CATS)
    prev_t = df["t"].shift(1)
    new_run = ((df["t"] - prev_t != STEP) | (df["workout_cat"] != df["workout_cat"].shift(1))
               | (is_wk != is_wk.shift(1)) | (df["window"] != df["window"].shift(1)))
    run_id = new_run.cumsum()
    runs = (df[is_wk].assign(run_id=run_id[is_wk]).groupby("run_id")
            .agg(cat=("workout_cat", "first"), start=("t", "min"), end=("t", "max"), window=("window", "first"))
            .sort_values("start").reset_index(drop=True))
    if len(runs) == 0:
        raise SystemExit("no workout runs in the aligned table; nothing to build")

    sessions = []
    for _, r in runs.iterrows():
        if sessions:
            s = sessions[-1]
            idle = mins(r["start"] - (s["end"] + STEP))
            if r["cat"] == s["cat"] and r["window"] == s["window"] and idle <= MERGE_GAP_MIN:
                s["end"] = r["end"]
                s["n_parts"] += 1
                continue
        sessions.append({"cat": r["cat"], "start": r["start"], "end": r["end"],
                         "window": int(r["window"]), "n_parts": 1})
    Sess = pd.DataFrame(sessions).sort_values("start").reset_index(drop=True)
    Sess.insert(0, "session_id", [f"S{i + 1:03d}" for i in range(len(Sess))])
    Sess["duration_min"] = mins(Sess["end"] - Sess["start"]) + 5
    Sess["cat2"] = [cat2_of(c, t) for c, t in zip(Sess["cat"], Sess["start"])]
    Sess = apply_overrides(Sess)

    if WK_PATH.exists():
        W = pd.read_csv(WK_PATH, parse_dates=["s", "e"])
        Sess["n_workouts"] = [int(((W["cat"] == r["cat"]) & (W["s"] <= r["end"] + STEP) & (W["e"] >= r["start"])).sum())
                              for _, r in Sess.iterrows()]
    else:
        Sess["n_workouts"] = np.nan
    Sess["session_split"] = np.where(Sess["start"] <= EXPLORE_LAST_START, "explore", "prospective")
    Sess["hour_start"] = Sess["start"].dt.hour
    Sess["dow"] = [int(g.at[t, "dow"]) for t in Sess["start"]]
    Sess["g7"] = [int(g.at[t, "g7"]) for t in Sess["start"]]

    F = pd.DataFrame([compute_features(g, s["start"], s["end"]) for _, s in Sess.iterrows()])
    Sess = pd.concat([Sess, F], axis=1)
    Sess["hrs_since_prev_session"] = (Sess["start"] - Sess["end"].shift(1)) / pd.Timedelta(hours=1)
    Sess["aerobic"] = Sess["cat2"].map(AEROBIC)
    assert Sess["aerobic"].notna().all(), "cat2 without an aerobic value"

    short = Sess["duration_min"] < MIN_DURATION_MIN
    Sess["too_short"] = short.astype(int)
    Sess.loc[short, ["usable", "usable_v1"]] = 0
    Sess.loc[short, "flag_reason"] = [r + "+too_short" if r else "too_short" for r in Sess.loc[short, "flag_reason"]]

    id_cols = ["session_id", "cat", "cat2", "aerobic", "start", "end", "duration_min", "n_parts",
               "window", "session_split", "hour_start", "dow", "g7", "overridden"]
    other = [c for c in Sess.columns if c not in id_cols + OUTCOME_COLS]
    return Sess[id_cols + other + OUTCOME_COLS]


# --------------------------------------------------------------------------
# QC report (explore rows only for anything involving outcomes)
# --------------------------------------------------------------------------
def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def rate_table(d, by, outcome="low_any"):
    t = d.groupby(by)[outcome].agg(n="size", k="sum", rate="mean")
    t["rate"] = t["rate"].round(3)
    return t


def qc_report(Sess):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 200)
    E = Sess[Sess["session_split"] == "explore"].copy()
    P = Sess[Sess["session_split"] == "prospective"].copy()
    U = E[E["usable"] == 1].copy()
    print(f"Wrote {OUT_PATH}: {len(Sess)} sessions ({len(E)} explore, {len(P)} prospective)")

    hdr("0. Session overrides applied")
    if OVERRIDE_LOG:
        print(pd.DataFrame(OVERRIDE_LOG).to_string(index=False))
        print(f"\noverridden == 1: {int(Sess['overridden'].sum())} sessions "
              f"({int(E['overridden'].sum())} explore); features and outcomes recomputed from the corrected window")
    else:
        print("  none")

    hdr("1. Explore sessions by category and window")
    print(pd.crosstab(E["cat2"], E["window"], margins=True))
    print(f"\nMerged sessions (n_parts > 1): {(E['n_parts'] > 1).sum()} of {len(E)}")
    print(f"Sessions built from >1 raw watch workout (n_workouts > 1): {(E['n_workouts'] > 1).sum()} of {len(E)}")
    print(f"\nNot usable: {(E['usable'] == 0).sum()} of {len(E)}   [v1 rule: {(E['usable_v1'] == 0).sum()}]  (reasons, not exclusive)")
    nu = E[E["usable"] == 0]
    print(f"  outcome_missing_frac > 0.20 : {(nu['outcome_missing_frac'] > 0.20).sum()}")
    print(f"  g0 missing                  : {nu['g0'].isna().sum()}")
    print(f"  real_gap/dst_day in span    : {(nu['flag_in_span'] == 1).sum()}")
    print(f"  too_short (< {MIN_DURATION_MIN} min)        : {(nu['too_short'] == 1).sum()}")
    if len(nu):
        print(nu[["session_id", "cat2", "start", "duration_min", "outcome_missing_frac",
                  "rows_missing_in_span", "g0", "flag_reason"]].to_string(index=False))
    print(f"\nlong_uncovered == 1 (info only): {int(E['long_uncovered'].sum())} explore sessions")
    print("\nUsable explore sessions by category and window")
    print(pd.crosstab(U["cat2"], U["window"], margins=True))
    print("\nDuration (min) of explore sessions by cat2")
    print(E.groupby("cat2")["duration_min"].describe()[["count", "min", "25%", "50%", "75%", "max"]])

    hdr("2. Prospective sessions (count only)")
    print(f"Total prospective: {len(P)}")

    hdr("3. Outcomes, usable explore sessions")
    print(f"n usable = {len(U)}")
    for c in ["low_during", "low_post2h", "low_any", "low_late", "below3_any"]:
        print(f"  {c:12s}: {int(U[c].sum()):3d}  ({U[c].mean():.1%})")
    fl = U["first_low_min"].dropna()
    if len(fl):
        print(f"\nfirst_low_min (n={len(fl)} with a low in start..end+2h): "
              f"median {fl.median():.0f}, Q1 {fl.quantile(.25):.0f}, Q3 {fl.quantile(.75):.0f} min")
    print(f"nadir: median {U['nadir'].median():.1f}, Q1 {U['nadir'].quantile(.25):.1f}, "
          f"Q3 {U['nadir'].quantile(.75):.1f} mmol/L")

    hdr("4. low_any rate by ... (usable explore sessions, n in every cell)")
    for by in ["cat", "cat2", "meal_band", "prevlow_band", "fed_2h", "fed_4h",
               "any_low_prev24h", "activity_at_start"]:
        print(f"\nby {by}")
        print(rate_table(U, by).to_string())

    hdr("5. Fed state vs insulin (usable explore sessions)")
    corr = U["fed_4h"].corr(U["insulin_total_4h"])
    print(f"corr(fed_4h, insulin_total_4h) = {corr:.3f}")
    med = U["insulin_total_4h"].median()
    U["insulin_4h_hi"] = (U["insulin_total_4h"] > med).astype(int)
    print(f"insulin_total_4h median = {med:.2f} U  (hi = above median)")
    t = U.groupby(["fed_4h", "insulin_4h_hi"])["low_any"].agg(n="size", k="sum", rate="mean")
    t["rate"] = t["rate"].round(3)
    print(t.to_string())
    print(f"\ncorrection_only_4h == 1: {int(U['correction_only_4h'].sum())} of {len(U)} usable explore sessions")

    hdr("6. Example explore sessions (all columns, transposed)")
    ex_ids = []
    for cat in ["gym", "cricket_indoor", "cricket_outdoor", "football", "cycling"]:
        c = U[U["cat2"] == cat]
        if len(c):
            ex_ids.append(c.iloc[len(c) // 2]["session_id"])
    if ex_ids:
        print(E[E["session_id"].isin(ex_ids[:5])].set_index("session_id").T.to_string())


if __name__ == "__main__":
    df = load_aligned()
    Sess = build_sessions(df)
    Sess.to_csv(OUT_PATH, index=False)
    qc_report(Sess)
