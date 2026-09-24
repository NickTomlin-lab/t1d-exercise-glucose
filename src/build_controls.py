"""
build_controls.py - matched exercise-free control windows for each usable
explore session, plus the comparison report.

Personal glucose-analysis project. All glucose in mmol/L. Analysis only:
nothing here informs a dosing decision. No models are fitted.

Reads  PROCESSED_DIR/aligned_5min.csv (never modified) and sessions.csv
       (written by build_sessions.py - run that first).
Writes PROCESSED_DIR/control_windows.csv.

Control window rule:
  - Same clock start time and the same length (duration + 2 h outcome span)
    as the session, on a different calendar date.
  - Same data window, within 28 days either side of the session date, and
    entirely on or before the explore cut-off.
  - Exercise-free: no session of any category overlaps
    [control start - 24 h, control end + 2 h].
  - Glucose usable by the session rule: missing <= 0.20, no real_gap /
    dst_day in the span, valid g0.
  - Up to 3 controls per session, nearest in date first (ties: the earlier
    date first). A control date may serve more than one session.
Features are computed by build_sessions.compute_features, so definitions are
identical to the session table, anchored on the control's own start time.
"""
import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR
from build_sessions import STEP, EXPLORE_LAST_START, load_aligned, compute_features, hdr   # noqa: E402

SESS_PATH = PROCESSED_DIR / "sessions.csv"
OUT_PATH = PROCESSED_DIR / "control_windows.csv"
MAX_CONTROLS = 3
MAX_DAYS = 28
EXCL_BEFORE = pd.Timedelta(hours=24)
EXCL_AFTER = pd.Timedelta(hours=2)
BOOT_N = 2000
BOOT_SEED = 20260921
CTRL_COLS = ["session_id", "control_id", "control_date", "days_from_session", "window",
             "cat2", "start", "end", "duration_min",
             "g0", "slope30", "mins_since_meal", "meal_band", "bolus_iob_u",
             "est_basal_iob_u", "insulin_total_4h", "low_any", "low_late", "below3_any",
             "nadir", "outcome_missing_frac", "usable", "overridden"]


def build_controls(g, Sess):
    all_sess = Sess[["start", "end"]].copy()
    U = Sess[(Sess["session_split"] == "explore") & (Sess["usable"] == 1)]
    tmax = g.index.max()
    out = []
    for _, s in U.iterrows():
        st, en = s["start"], s["end"]
        span_len = en - st
        found = 0
        for d in sorted([d for d in range(-MAX_DAYS, MAX_DAYS + 1) if d != 0], key=lambda d: (abs(d), d)):
            cst = st + pd.Timedelta(days=d)
            cen = cst + span_len
            cend2 = cen + EXCL_AFTER
            if cend2 > EXPLORE_LAST_START or cend2 > tmax:
                continue
            if cst not in g.index or cend2 not in g.index:
                continue
            if g.at[cst, "window"] != s["window"] or g.at[cend2, "window"] != s["window"]:
                continue
            lo, hi = cst - EXCL_BEFORE, cend2
            if ((all_sess["start"] <= hi) & (all_sess["end"] >= lo)).any():
                continue
            f = compute_features(g, cst, cen)
            if f["usable"] != 1:
                continue
            f.update({"session_id": s["session_id"], "control_id": f"{s['session_id']}c{found + 1}",
                      "control_date": cst.date(), "days_from_session": d, "window": int(s["window"]),
                      "cat2": s["cat2"], "start": cst, "end": cen, "duration_min": s["duration_min"],
                      "overridden": int(s.get("overridden", 0))})
            out.append(f)
            found += 1
            if found == MAX_CONTROLS:
                break
    C = pd.DataFrame(out)
    return C[CTRL_COLS] if len(C) else pd.DataFrame(columns=CTRL_COLS)


# --------------------------------------------------------------------------
# Stats helpers (also imported by make_figures.py)
# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def rate_table(d, by, outcome="low_any"):
    t = d.groupby(by)[outcome].agg(n="size", k="sum").reset_index()
    t["rate"] = (t["k"] / t["n"]).round(3)
    ci = [wilson(k, n) for k, n in zip(t["k"], t["n"])]
    t["wilson_lo"] = [round(c[0], 3) for c in ci]
    t["wilson_hi"] = [round(c[1], 3) for c in ci]
    return t


def paired_ratio(sess_y, ctrl_mean_y, rng, n_boot=BOOT_N):
    sess_y = np.asarray(sess_y, float)
    ctrl_mean_y = np.asarray(ctrl_mean_y, float)
    n = len(sess_y)
    s_rate, c_rate = sess_y.mean(), ctrl_mean_y.mean()
    ratio = s_rate / c_rate if c_rate > 0 else np.inf
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cs = ctrl_mean_y[idx].mean()
        draws.append(sess_y[idx].mean() / cs if cs > 0 else np.inf)
    draws = np.array(draws)
    finite = np.isfinite(draws)
    lo, hi = (np.percentile(draws[finite], [2.5, 97.5]) if finite.sum() > 0 else (np.nan, np.nan))
    return dict(n=n, sess_rate=s_rate, ctrl_rate=c_rate, ratio=ratio, ci_lo=lo, ci_hi=hi,
                n_inf=int((~finite).sum()))


def fmt_ratio(r):
    ratio = "inf" if not np.isfinite(r["ratio"]) else f"{r['ratio']:.2f}"
    inf_note = f"  ({r['n_inf']} draws with control rate 0 dropped)" if r["n_inf"] else ""
    return (f"n={r['n']:3d}  session {r['sess_rate']:.3f}  control {r['ctrl_rate']:.3f}  "
            f"ratio {ratio}  boot95% [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]{inf_note}")


def report(Sess, C):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 300)
    rng = np.random.default_rng(BOOT_SEED)
    E = Sess[Sess["session_split"] == "explore"].copy()
    U = E[E["usable"] == 1].copy()
    assert (C["start"] <= EXPLORE_LAST_START).all() if len(C) else True

    hdr("1. Usable sessions v1 -> v2, and v2 outcomes (explore)")
    v1, v2 = int(E["usable_v1"].sum()), int(E["usable"].sum())
    print(f"usable v1 (pod_change excluded sessions): {v1} of {len(E)}")
    print(f"usable v2 (pod_change ignored)          : {v2} of {len(E)}   restored: {v2 - v1}")
    print("\nusable v2 by cat2:")
    print(pd.crosstab(U["cat2"], U["window"], margins=True))
    print(f"\nOutcomes, n usable = {len(U)}")
    for c in ["low_during", "low_post2h", "low_any", "low_late", "below3_any"]:
        k = int(U[c].sum())
        lo, hi = wilson(k, len(U))
        print(f"  {c:12s}: {k:3d}  ({k / len(U):.1%})   wilson95 [{lo:.3f}, {hi:.3f}]")

    hdr("2. low_any by cat2 / meal_band / prevlow_band / activity_at_start (usable explore)")
    for by in ["cat2", "meal_band", "prevlow_band", "activity_at_start"]:
        print(f"\nby {by}")
        print(rate_table(U, by).to_string(index=False))

    hdr("3. meal_band by cat2 (usable explore session counts)")
    print(pd.crosstab(U["cat2"], U["meal_band"], margins=True))

    hdr("4. Controls found")
    per = C.groupby("session_id").size().reindex(U["session_id"]).fillna(0).astype(int)
    print("sessions by number of controls:")
    print(per.value_counts().sort_index().rename_axis("n_controls").rename("sessions").to_string())
    print(f"total control windows: {len(C)}   distinct control dates: {C['control_date'].nunique() if len(C) else 0}")
    if len(C):
        print(f"days_from_session: median |d| = {C['days_from_session'].abs().median():.0f}, "
              f"max |d| = {C['days_from_session'].abs().max()}")
    print("\nsessions with 0 controls:")
    zero = U[U["session_id"].isin(per[per == 0].index)]
    print(zero[["session_id", "cat2", "start", "duration_min", "window"]].to_string(index=False) if len(zero) else "  none")

    hdr("5. Baseline comparison: session rate vs matched-control rate (sessions with >= 1 control)")
    if len(C):
        cm = C.groupby("session_id")[["low_any", "low_late", "below3_any"]].mean()
        M = U.set_index("session_id").join(cm, rsuffix="_ctrl", how="inner")
        print(f"sessions with >= 1 control: {len(M)} of {len(U)}")
        print("control rate = mean over sessions of (mean over that session's controls);")
        print(f"bootstrap: resample sessions, {BOOT_N} draws, seed {BOOT_SEED}\n")
        for oc in ["low_any", "low_late", "below3_any"]:
            r = paired_ratio(M[oc], M[f"{oc}_ctrl"], rng)
            print(f"{oc:11s}  {fmt_ratio(r)}")
        print("\nlow_any by cat2:")
        for c2, grp in M.groupby("cat2"):
            r = paired_ratio(grp["low_any"], grp["low_any_ctrl"], rng)
            print(f"  {c2:16s} {fmt_ratio(r)}")
        print("\nraw control-window pooled rates (every control counted once, for reference):")
        for oc in ["low_any", "low_late", "below3_any"]:
            k = int(C[oc].sum())
            lo, hi = wilson(k, len(C))
            print(f"  {oc:11s}: {k:3d} / {len(C)}  ({k / len(C):.3f})  wilson95 [{lo:.3f}, {hi:.3f}]")

        hdr("6. Meal timing without exercise: low_any by meal_band, controls vs sessions")
        ct = rate_table(C, "meal_band").set_index("meal_band")
        st = rate_table(U, "meal_band").set_index("meal_band")
        both = pd.concat({"control": ct, "session": st}, axis=1).reindex(["0-2h", "2-4h", "4-6h", "6h+"])
        print(both.to_string())

    hdr("7. long_uncovered sessions and doubtful cat2")
    lu = E[E["long_uncovered"] == 1]
    print(f"long_uncovered == 1: {len(lu)} explore sessions")
    d1 = E[(E["cat2"] == "cricket_indoor") & (E["duration_min"] > 120)]
    print(f"cricket_indoor longer than 120 min (may be an outdoor game): {len(d1)}")
    d2 = E[(E["cat2"] == "cricket_outdoor") & (E["duration_min"] <= 90)]
    print(f"cricket_outdoor of 90 min or less (may be nets / indoor): {len(d2)}")


if __name__ == "__main__":
    df = load_aligned()
    g = df.set_index("t")
    Sess = pd.read_csv(SESS_PATH, parse_dates=["start", "end"])
    C = build_controls(g, Sess)
    C.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {len(C)} control windows for {C['session_id'].nunique() if len(C) else 0} sessions")
    report(Sess, C)
