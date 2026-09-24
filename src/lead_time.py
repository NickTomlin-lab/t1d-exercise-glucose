"""
lead_time.py - pump lead time around exercise sessions.

Personal glucose-analysis project. All glucose in mmol/L. Analysis only:
nothing here informs a dosing decision. No model is fitted here.

Question: in sessions that included a low, how long before the pump first
reacted had the body already signalled exercise?

Reads  PROCESSED_DIR/aligned_5min.csv (never modified) and sessions.csv
Prints the report to stdout and writes lead_time_summary.csv (quartiles of
each signal and gap, sessions with / without a low; no timestamps).

Universe: usable explore sessions only. Prospective rows are dropped before
anything is computed or printed.

Signals, per session (start = first 5-min row of the session, end = last):
  t_first_low    first row in [start, end + 2 h] with glucose < 4.0
  t_pump_cut     first row t >= start - 30 min with frac_suspend >= 0.5 on t and
                 t + 5 min, where the six rows before t (30 min) all have
                 frac_suspend < 0.5
  t_hr_rise      first row t >= start - 30 min with hr_mean >= baseline + 30 bpm,
                 baseline = median hr_mean over [start - 150, start - 30) min,
                 rows with hr_missing excluded; NaN if that window has no HR
  t_steps_rise   first row t >= start - 30 min with steps >= 200
  t_glucose_fall first row t >= start with glucose(t) - glucose(t - 15) <= -0.5
All times are reported in minutes relative to the session start.
"""
import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR
from build_sessions import load_aligned, hdr, EXPLORE_LAST_START     # noqa: E402

SESS_PATH = PROCESSED_DIR / "sessions.csv"
SUMMARY_PATH = PROCESSED_DIR / "lead_time_summary.csv"
STEP = pd.Timedelta(minutes=5)
LOW = 4.0
PRE = pd.Timedelta(minutes=30)
POST = pd.Timedelta(hours=2)
HR_BASE_WINDOW = pd.Timedelta(minutes=120)
HR_RISE = 30.0
STEPS_RISE = 200.0
FALL = -0.5
SUSPEND_ON = 0.5
SUSPEND_QUIET_ROWS = 6
SIGNALS = ["t_first_low", "t_pump_cut", "t_hr_rise", "t_steps_rise", "t_glucose_fall"]
GAPS = [("pump_cut - hr_rise", "t_pump_cut", "t_hr_rise"),
        ("pump_cut - steps_rise", "t_pump_cut", "t_steps_rise"),
        ("first_low - pump_cut", "t_first_low", "t_pump_cut"),
        ("first_low - hr_rise", "t_first_low", "t_hr_rise")]


def mins(td):
    return td / pd.Timedelta(minutes=1)


def g0_band(g):
    if g < 7.0:
        return "g0 < 7.0"
    if g < 10.0:
        return "g0 7.0-9.9"
    return "g0 >= 10.0"


def signal_times(g, st, en):
    r = {}
    pre = st - PRE
    span_end = en + POST
    span = g.loc[st:span_end]
    lows = span.index[span["glucose"] < LOW]
    r["t_first_low"] = lows[0] if len(lows) else pd.NaT

    fs = g.loc[pre - SUSPEND_QUIET_ROWS * STEP: span_end + STEP, "frac_suspend"]
    r["t_pump_cut"] = pd.NaT
    r["suspended_at_pre"] = int(pre in fs.index and fs[pre] >= SUSPEND_ON)
    r["cut_before_window"] = int(pre in fs.index and (pre - STEP) in fs.index
                                 and fs[pre] >= SUSPEND_ON and fs[pre - STEP] >= SUSPEND_ON)
    for t in g.loc[pre:span_end].index:
        quiet = [t - k * STEP for k in range(1, SUSPEND_QUIET_ROWS + 1)]
        if not all(q in fs.index for q in quiet) or (t + STEP) not in fs.index:
            continue
        if fs[t] >= SUSPEND_ON and fs[t + STEP] >= SUSPEND_ON and all(fs[q] < SUSPEND_ON for q in quiet):
            r["t_pump_cut"] = t
            break

    base = g.loc[pre - HR_BASE_WINDOW: pre - STEP]
    base = base[~base["hr_missing"].astype(bool)]["hr_mean"].dropna()
    r["hr_baseline"] = float(base.median()) if len(base) else np.nan
    r["hr_baseline_n"] = int(len(base))
    r["t_hr_rise"] = pd.NaT
    if len(base):
        w = g.loc[pre:en]
        hit = w.index[(~w["hr_missing"].astype(bool)) & (w["hr_mean"] >= r["hr_baseline"] + HR_RISE)]
        r["t_hr_rise"] = hit[0] if len(hit) else pd.NaT
    r["hr_max_session"] = float(g.loc[pre:en, "hr_mean"].max())

    w = g.loc[pre:en]
    hit = w.index[w["steps"] >= STEPS_RISE]
    r["t_steps_rise"] = hit[0] if len(hit) else pd.NaT

    gl = g["glucose"]
    r["t_glucose_fall"] = pd.NaT
    for t in g.loc[st:span_end].index:
        t15 = t - 3 * STEP
        if t15 in gl.index and pd.notna(gl[t]) and pd.notna(gl[t15]) and gl[t] - gl[t15] <= FALL:
            r["t_glucose_fall"] = t
            break
    return r


def q_table(D, cols):
    rows = []
    for c in cols:
        v = D[c].dropna()
        rows.append(dict(signal=c, n_present=len(v), n_absent=int(D[c].isna().sum()),
                         q1=v.quantile(.25) if len(v) else np.nan, median=v.median() if len(v) else np.nan,
                         q3=v.quantile(.75) if len(v) else np.nan, min=v.min() if len(v) else np.nan,
                         max=v.max() if len(v) else np.nan))
    return pd.DataFrame(rows).set_index("signal")


def gap_table(D):
    rows = []
    for name, a, b in GAPS:
        v = (D[a] - D[b]).dropna()
        rows.append(dict(gap=name, n=len(v), q1=v.quantile(.25) if len(v) else np.nan,
                         median=v.median() if len(v) else np.nan, q3=v.quantile(.75) if len(v) else np.nan,
                         frac_positive=float((v > 0).mean()) if len(v) else np.nan))
    return pd.DataFrame(rows).set_index("gap")


def gap_by_g0(D, a, b):
    rows = []
    for band in ["g0 < 7.0", "g0 7.0-9.9", "g0 >= 10.0"]:
        d = D[D["g0_band"] == band]
        v = (d[a] - d[b]).dropna()
        rows.append(dict(g0_band=band, n_sessions=len(d), n_with_both=len(v),
                         n_cut_before_window=int(d["cut_before_window"].sum()),
                         q1=v.quantile(.25) if len(v) else np.nan, median=v.median() if len(v) else np.nan,
                         q3=v.quantile(.75) if len(v) else np.nan))
    return pd.DataFrame(rows).set_index("g0_band")


def fmt(v):
    return f"{v:.1f}"


def signals_table(g, U):
    """Per-session signal times (absolute) plus the minute version; used by main()
    and by make_figures.py."""
    recs = []
    for _, s in U.iterrows():
        r = signal_times(g, s["start"], s["end"])
        r.update(session_id=s["session_id"], cat2=s["cat2"], start=s["start"], end=s["end"],
                 duration_min=s["duration_min"], g0=s["g0"], low_during=int(s["low_during"]),
                 low_post2h=int(s["low_post2h"]), low_any=int(s["low_any"]),
                 activity_at_start=int(s["activity_at_start"]), activity_lead_min=float(s["activity_lead_min"]))
        recs.append(r)
    D = pd.DataFrame(recs)
    M = D.copy()
    for c in SIGNALS:
        M[c] = mins(D[c] - D["start"])
    M["g0_band"] = M["g0"].map(g0_band)
    return D, M


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 300)
    df = load_aligned()
    g = df.set_index("t")
    Sess = pd.read_csv(SESS_PATH, parse_dates=["start", "end"])
    U = Sess[(Sess["session_split"] == "explore") & (Sess["usable"] == 1)].copy()
    assert (U["start"] <= EXPLORE_LAST_START).all(), "prospective row leaked"
    D_abs, M = signals_table(g, U)
    L = M[(M["low_during"] == 1) | (M["low_post2h"] == 1)]
    NL = M[M["low_any"] == 0]
    assert len(L) + len(NL) == len(M)

    print("Analysis only: nothing here informs a dosing decision. Glucose in mmol/L.")
    print(f"Usable explore sessions: {len(M)}   with a low (low_during or low_post2h): {len(L)}   "
          f"without a low (low_any == 0): {len(NL)}")
    print("All times below are minutes relative to the session start (negative = before start).")

    hdr("1. Sessions with a low: when each signal first appeared (min from start)")
    print(q_table(L, SIGNALS).to_string(float_format=fmt))
    print(f"\nActivity mode already on at start: {int(L['activity_at_start'].sum())} of {len(L)} sessions")
    print(f"Pump already suspended (frac_suspend >= 0.5) at start - 30 min: {int(L['suspended_at_pre'].sum())} of {len(L)}")
    print(f"  of which the suspend run began before start - 30 (cut_before_window): {int(L['cut_before_window'].sum())}")
    print(f"HR baseline unavailable: {int(L['hr_baseline'].isna().sum())} of {len(L)}")
    print("by cat2 (n, n with pump cut, n with HR rise, n with steps rise):")
    print(L.groupby("cat2").agg(n=("session_id", "size"), pump_cut=("t_pump_cut", "count"),
                                hr_rise=("t_hr_rise", "count"), steps_rise=("t_steps_rise", "count"),
                                glucose_fall=("t_glucose_fall", "count")).to_string())

    hdr("2. Sessions with a low: gaps between signals (min; positive = first named event later)")
    print(gap_table(L).to_string(float_format=fmt))
    Lc = L[L["cut_before_window"] == 0]
    print(f"\nsensitivity: the same gaps for the {len(Lc)} sessions with a low where the pump was NOT already "
          f"in a suspend run before start - 30 min")
    print(gap_table(Lc).to_string(float_format=fmt))

    hdr("3. Sessions WITHOUT a low: the same signals and gaps")
    print(q_table(NL, [c for c in SIGNALS if c != "t_first_low"]).to_string(float_format=fmt))
    print(gap_table(NL).loc[["pump_cut - hr_rise", "pump_cut - steps_rise"]].to_string(float_format=fmt))

    hdr("4. Pump lead time by starting glucose (g0 at session start)")
    for name, a, b in GAPS[:2]:
        print(f"\n{name}  (min; positive = pump cut came after the {b[2:]})")
        both = pd.concat({"with a low": gap_by_g0(L, a, b), "without a low": gap_by_g0(NL, a, b)})
        print(both.to_string(float_format=fmt))

    hdr("5. Plain reading")
    v_hr = (L["t_pump_cut"] - L["t_hr_rise"]).dropna()
    v_low = (L["t_first_low"] - L["t_pump_cut"]).dropna()
    v_st = (L["t_pump_cut"] - L["t_steps_rise"]).dropna()
    if len(v_hr) and len(v_low):
        print(f"Across {len(L)} sessions that included a low, the watch showed exercise (heart rate up 30 bpm) "
              f"a median of {v_hr.median():.0f} minutes before the pump first cut delivery "
              f"(n = {len(v_hr)} sessions with both signals); the first low came a median of "
              f"{v_low.median():.0f} minutes after the pump acted (n = {len(v_low)}).")
    if len(v_st):
        print(f"Using steps instead of heart rate: the watch showed movement a median of {v_st.median():.0f} minutes "
              f"before the pump cut (n = {len(v_st)}).")

    # summary csv: quartiles only, no timestamps
    rows = []
    for label, D in (("with_low", L), ("without_low", NL)):
        qt = q_table(D, SIGNALS)
        for sig, r in qt.iterrows():
            rows.append(dict(group=label, kind="signal", name=sig, n=r["n_present"], q1=r["q1"], median=r["median"], q3=r["q3"]))
        gt = gap_table(D)
        for gap, r in gt.iterrows():
            rows.append(dict(group=label, kind="gap", name=gap, n=r["n"], q1=r["q1"], median=r["median"], q3=r["q3"]))
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)
    print(f"\nwrote {SUMMARY_PATH}")

    hdr("6. Sanity print: five sessions with a low, all timestamps (private check against memory)")
    pick = L[L[SIGNALS].notna().all(axis=1)]
    ids = list(pick["session_id"].head(5))
    if ids:
        cols = ["cat2", "start", "end", "duration_min", "g0", "low_during", "low_post2h", "activity_at_start",
                "activity_lead_min", "hr_baseline", "hr_max_session", "suspended_at_pre", "cut_before_window"] + SIGNALS
        print(D_abs[D_abs["session_id"].isin(ids)].set_index("session_id")[cols].T.to_string())

    hdr("Method notes")
    print("""
 1. 'Sessions that included a low' = low_during == 1 or low_post2h == 1 (= low_any).
 2. Session start/end are the 5-min grid rows in sessions.csv (overrides applied).
 3. Pump cut and glucose fall are searched up to end + 2 h; HR rise and steps
    rise up to the session END. A signal not found is NaN and that session drops
    out of any gap using it.
 4. A pump already suspended at start - 30 min has no 'first cut' unless it
    resumes for 30 min and cuts again; cut_before_window marks those and the gap
    tables are repeated without them.
 5. HR baseline = median hr_mean over [start - 150, start - 30) min.
 6. Quartiles are pandas linear-interpolation quantiles over sessions with the
    signal(s) present; gaps are signed.
 7. No model is fitted; no prospective row is read past the initial filter.
""")


if __name__ == "__main__":
    main()
