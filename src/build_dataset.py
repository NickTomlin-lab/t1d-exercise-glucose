"""
build_dataset.py - align the four raw data sources onto one 5-minute grid.

Personal glucose-analysis project. Analysis only: nothing here informs a
dosing decision. Produces PROCESSED_DIR/aligned_5min.csv: one row per
5-minute interval, local (Europe/London) wall time, glucose in mmol/L.

This script needs the RAW exports, which are NOT in the repository (see
docs/data_sources.md). The synthetic sample in data/synthetic/ is written
directly in aligned form by make_synthetic.py, so the rest of the pipeline
runs without this step.

Raw layout (RAW_DIR from config.py):
  glooko/*.zip                 Glooko CSV exports (CGM, bolus + carbs, daily
                               insulin totals, alarms), one zip per window
  clarity/*.csv                Dexcom Clarity exports (EGV rows used)
  glooko_api/basal_states_*.json   pump delivery-state segments (suspend /
                                   automated / max), one file per window
  glooko_api/modes_*.json      pump mode events (Activity / Limited / Manual)
  health/*.zip                 Health export (workouts, heart rate, steps)

Study settings come from config.S (windows, fence, DST days, the football
relabel rule, the fixed test start). Dated corrections (real CGM gaps, watch
workout relabels) come from the overrides table (config.load_overrides).

Quality rules (from the private data dictionary):
  - CGM backbone = union of Glooko and Clarity (Glooko preferred where both).
  - Clarity "Low" -> 2.2 mmol/L with below_range flag; "High" -> 22.2.
  - UK clock-change days flagged dst_day, excluded from clean.
  - Apple Health timestamps carry a fixed export offset -> convert to UTC,
    then to Europe/London wall time.
  - Basal states: suspend = 0 U/h exact; max = exact; automated rate is
    unknown per pulse, calibrated per day from the daily basal totals.
  - Bolus IOB: linear decay over the pump's active insulin time.
  - Windows are fenced where settings say so: nothing from before the fence
    (boluses, workouts, past-hour rows) feeds a row after it.
"""
import glob
import io
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from config import RAW_DIR, PROCESSED_DIR, S

LONDON = ZoneInfo("Europe/London")
DIA_H = float(S["dia_hours"])
WINDOWS = config.windows()
FENCE_END = config.ts("fence_end")
QUARANTINE = S.get("quarantine_window")
TEST_START = config.ts("test_start")
DST_DAYS = {pd.Timestamp(d).date() for d in S.get("dst_days", [])}
OTHER_TO_GYM_MIN = 60          # an "other" workout this close to a gym workout is part of it
HARD_CATS = ["cricket", "football", "gym", "cycling"]   # exercise categories (walking is not)
RELABEL_LOG = []               # every "other" workout and what happened to it


def load_glooko_csv(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if member in n]
        parts = [pd.read_csv(io.BytesIO(z.read(n)), skiprows=1) for n in sorted(names)]
    return pd.concat(parts) if parts else pd.DataFrame()


def build():
    gzips = sorted(glob.glob(os.path.join(RAW_DIR, "glooko", "*.zip")))
    clarity = sorted(glob.glob(os.path.join(RAW_DIR, "clarity", "*.csv")))
    basal_json = sorted(glob.glob(os.path.join(RAW_DIR, "glooko_api", "basal_states_*.json")))
    mode_files = sorted(glob.glob(os.path.join(RAW_DIR, "glooko_api", "modes_*.json")))
    health_zips = sorted(glob.glob(os.path.join(RAW_DIR, "health", "*.zip")), key=os.path.getmtime)
    if not (gzips and clarity and basal_json and health_zips):
        raise SystemExit(f"raw exports not found under {RAW_DIR} (see docs/data_sources.md)")
    health_zip = health_zips[-1]          # newest export wins

    real_gaps = config.load_overrides("real_gap")
    relabels = config.load_overrides("workout_relabel")

    # ---------------- grid ----------------
    frames = []
    for d0, d1, w in WINDOWS:
        idx = pd.date_range(d0, d1, freq="5min", inclusive="left")
        frames.append(pd.DataFrame({"t": idx, "window": w}))
    grid = pd.concat(frames, ignore_index=True)
    gsec = grid.t.values.astype("datetime64[s]").astype("int64")

    # ---------------- CGM union ----------------
    glooko = pd.concat([load_glooko_csv(z, "cgm_data_") for z in gzips])
    glooko["t"] = pd.to_datetime(glooko["Timestamp"], format="%d/%m/%Y %H:%M")
    glooko["g"] = pd.to_numeric(glooko["CGM Glucose Value (mmol/L)"], errors="coerce")
    glooko["below_range"] = glooko.g < 2.2          # Glooko encodes below-range as 0.1
    glooko.loc[glooko.below_range, "g"] = 2.2

    cl = pd.concat([pd.read_csv(f) for f in clarity])
    cl = cl[cl["Event Type"] == "EGV"].copy()
    cl["t"] = pd.to_datetime(cl["Timestamp (YYYY-MM-DDThh:mm:ss)"])
    raw_g = cl["Glucose Value (mmol/L)"].astype(str)
    cl["below_range"] = raw_g.eq("Low")
    cl["above_range"] = raw_g.eq("High")
    cl["g"] = pd.to_numeric(raw_g, errors="coerce")
    cl.loc[cl.below_range, "g"] = 2.2
    cl.loc[cl.above_range, "g"] = 22.2

    def to_interval(df):
        df = df.dropna(subset=["g"]).copy()
        df["iv"] = df.t.dt.floor("5min")
        return df

    gi = to_interval(glooko).groupby("iv").agg(g_glooko=("g", "mean"), br_g=("below_range", "max"))
    ci = to_interval(cl).groupby("iv").agg(g_clarity=("g", "mean"), br_c=("below_range", "max"))
    cgm = gi.join(ci, how="outer")
    cgm["glucose"] = cgm.g_glooko.fillna(cgm.g_clarity)
    cgm["glucose_src"] = np.where(cgm.g_glooko.notna(), "glooko", "clarity")
    cgm["below_range"] = cgm.br_g.fillna(False) | cgm.br_c.fillna(False)
    out = grid.merge(cgm[["glucose", "glucose_src", "below_range"]],
                     left_on="t", right_index=True, how="left")

    # ---------------- boluses, carbs, IOB ----------------
    bolus = pd.concat([load_glooko_csv(z, "bolus_data_") for z in gzips])
    bolus["t"] = pd.to_datetime(bolus["Timestamp"], format="%d/%m/%Y %H:%M")
    bolus["u"] = pd.to_numeric(bolus["Insulin delivered (U)"], errors="coerce").fillna(0)
    bolus["carbs"] = pd.to_numeric(bolus["Carbs input (g)"], errors="coerce").fillna(0)
    bolus["iv"] = bolus.t.dt.floor("5min")
    bsum = bolus.groupby("iv").agg(bolus_u=("u", "sum"), carbs_g=("carbs", "sum"))
    out = out.merge(bsum, left_on="t", right_index=True, how="left")
    out[["bolus_u", "carbs_g"]] = out[["bolus_u", "carbs_g"]].fillna(0)

    bt = bolus.t.values.astype("datetime64[s]").astype("int64")
    bu = bolus.u.values
    order = np.argsort(bt); bt, bu = bt[order], bu[order]
    iob = np.zeros(len(out))
    horizon = int(DIA_H * 3600)
    fence_s = int(FENCE_END.timestamp()) if FENCE_END is not None else None
    for k, t_s in enumerate(gsec):
        if fence_s is not None and t_s >= fence_s and t_s - horizon < fence_s:
            i0 = np.searchsorted(bt, fence_s, "left")
        else:
            i0 = np.searchsorted(bt, t_s - horizon, "right")
        i1 = np.searchsorted(bt, t_s, "right")
        if i1 > i0:
            age = t_s - bt[i0:i1]
            iob[k] = np.sum(bu[i0:i1] * (1 - age / horizon))
    out["bolus_iob_u"] = np.round(iob, 3)

    # ---------------- basal states ----------------
    def segments(arr):
        segs, start, prev = [], None, None
        for p in arr:
            if p["y"] == 1 and prev == 0 and start is None:
                start = p["x"]
            elif p["y"] == 0 and prev == 1 and start is not None:
                segs.append((start, p["x"])); start = None
            prev = p["y"]
        return segs

    segsets = {"suspend": [], "auto": [], "max": []}
    keymap = {"basalBarAutomatedSuspend": "suspend", "basalBarAutomated": "auto",
              "basalBarAutomatedMax": "max"}
    for f in basal_json:
        j = json.load(open(f))
        for k, arr in j["series"].items():
            if k in keymap:
                segsets[keymap[k]].extend(segments(arr))

    def cover(segs):
        cov = np.zeros(len(out))
        for s, e in segs:
            i0 = max(np.searchsorted(gsec, s, "right") - 1, 0)
            i1 = min(np.searchsorted(gsec, e, "left"), len(out))
            for i in range(i0, i1):
                lo = gsec[i]
                cov[i] += max(0, min(e, lo + 300) - max(s, lo))
        return cov

    for name, segs in segsets.items():
        out[f"frac_{name}"] = np.round(cover(segs) / 300, 3)
    out["frac_uncovered"] = np.round((1 - out.frac_suspend - out.frac_auto - out.frac_max).clip(lower=0), 3)

    ins = pd.concat([load_glooko_csv(z, "insulin_data_") for z in gzips])
    ins["date"] = pd.to_datetime(ins["Timestamp"], format="%d/%m/%Y %H:%M").dt.date
    ins = ins.set_index("date")["Total basal (U)"]
    day = out.t.dt.date
    auto_h = out.groupby(day).frac_auto.sum() / 12
    max_h = out.groupby(day).frac_max.sum() / 12
    rate = ((ins - max_h * 1.0) / auto_h).clip(lower=0)   # U/h while in the automated state
    out["est_basal_u"] = np.round(out.frac_auto * day.map(rate).astype(float) / 12
                                  + out.frac_max * 1.0 / 12, 4)

    # ---------------- pump mode timeline (Activity / Limited / Manual) ----------------
    # Event timestamps end in "Z" but are local wall time (verified against workouts).
    mode_events = {"activity": set(), "limited": set(), "manual": set()}
    mkey = {"pumpOp5HypoprotectMode": "activity", "pumpOp5LimitedMode": "limited",
            "pumpOp5ManualMode": "manual"}
    for f in mode_files:
        j = json.load(open(f))
        for k, arr in j["series"].items():
            if k in mkey:
                for p in arr:
                    if p.get("timestamp"):
                        mode_events[mkey[k]].add((p["timestamp"], p["endTimestamp"]))
    for name, evs in mode_events.items():
        segs = [(int(pd.Timestamp(a.replace("Z", "")).timestamp()),
                 int(pd.Timestamp(b.replace("Z", "")).timestamp())) for a, b in evs]
        out[f"frac_{name}"] = np.round(np.minimum(cover(segs) / 300, 1.0), 3)

    # ---------------- Apple Health ----------------
    with zipfile.ZipFile(health_zip) as z:
        xml_name = [n for n in z.namelist() if n.endswith("export.xml")][0]
        xml_file = z.open(xml_name)

    def to_local_naive(s):
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")
        return dt.astimezone(LONDON).replace(tzinfo=None)

    lo_bounds = [(a, b) for a, b, _ in WINDOWS]

    def in_windows(t):
        return any(a <= t < b for a, b in lo_bounds)

    hr_rows, step_rows, workouts = [], [], []
    for _, el in ET.iterparse(xml_file, events=("end",)):
        if el.tag == "Record":
            typ = el.get("type")
            if typ == "HKQuantityTypeIdentifierHeartRate":
                t = to_local_naive(el.get("startDate"))
                if in_windows(t):
                    hr_rows.append((t, float(el.get("value"))))
            elif typ == "HKQuantityTypeIdentifierStepCount":
                t = to_local_naive(el.get("startDate"))
                if in_windows(t):
                    step_rows.append((t, float(el.get("value"))))
            el.clear()
        elif el.tag == "Workout":
            s = to_local_naive(el.get("startDate"))
            e = to_local_naive(el.get("endDate"))
            typ = el.get("workoutActivityType").replace("HKWorkoutActivityType", "")
            workouts.append((s, e, typ))
            el.clear()

    hr = pd.DataFrame(hr_rows, columns=["t", "hr"])
    hr["iv"] = hr.t.dt.floor("5min")
    hrm = hr.groupby("iv").agg(hr_mean=("hr", "mean"), hr_n=("hr", "size"))
    out = out.merge(hrm, left_on="t", right_index=True, how="left")
    out["hr_mean"] = out.hr_mean.round(1)
    out["hr_missing"] = out.hr_mean.isna()

    st = pd.DataFrame(step_rows, columns=["t", "steps"])
    st["iv"] = st.t.dt.floor("5min")
    out = out.merge(st.groupby("iv").steps.sum().rename("steps"), left_on="t", right_index=True, how="left")
    out["steps"] = out.steps.fillna(0)

    # workout categories. The watch's "Soccer" type was used for cricket as well as
    # football; the private settings hold the weekday/clock rule that tells them
    # apart (football_rule). With the rule null (the published example) every
    # football-typed workout is football.
    wk = pd.DataFrame(workouts, columns=["s", "e", "typ"])
    wk = wk[wk.s.apply(in_windows)].copy()
    is_soccer = wk.typ.str.contains("Soccer", case=False)
    rule = S.get("football_rule")
    if rule:
        football = ((wk.s.dt.weekday == int(rule["weekday"]))
                    & wk.s.dt.hour.between(int(rule["hour_lo"]), int(rule["hour_hi"])))
    else:
        football = pd.Series(True, index=wk.index)
    wk.loc[is_soccer & football, "typ"] = "Football"
    wk.loc[is_soccer & ~football, "typ"] = "Cricket"
    gym = ["TraditionalStrengthTraining", "HighIntensityIntervalTraining", "Pilates", "Rowing"]
    wk["cat"] = np.select([wk.typ.eq("Football"), wk.typ.eq("Cricket"), wk.typ.isin(gym), wk.typ.eq("Walking")],
                          ["football", "cricket", "gym", "walking"], "other")

    # proximity rule: an "other" workout becomes gym when it starts within 60 min of
    # the end, or ends within 60 min of the start, of a gym-typed workout on the same
    # day. Applied before the fence only (or everywhere when there is no fence).
    RELABEL_LOG.clear()
    tol = pd.Timedelta(minutes=OTHER_TO_GYM_MIN)
    before_fence = (wk.s < FENCE_END) if FENCE_END is not None else pd.Series(True, index=wk.index)
    gym_wk = wk[(wk.cat == "gym") & before_fence]
    for i in wk.index[(wk.cat == "other") & before_fence]:
        o = wk.loc[i]
        same_day = gym_wk[gym_wk.s.dt.date == o.s.date()]
        d_start = (o.s - same_day.e).abs()
        d_end = (o.e - same_day.s).abs()
        gap = pd.concat([d_start, d_end], axis=1).min(axis=1)
        hit = gap[gap <= tol]
        entry = dict(typ=o.typ, start=o.s, end=o.e)
        if len(hit):
            j = hit.idxmin()
            wk.loc[i, "cat"] = "gym"
            entry.update(new_cat="gym", gap_min=round(hit[j].total_seconds() / 60, 1), via="proximity rule")
        else:
            entry.update(new_cat="other", gap_min=(round(gap.min().total_seconds() / 60, 1) if len(gap) else np.nan),
                         via="")
        RELABEL_LOG.append(entry)

    # dated relabels from the overrides table (matched on the start minute)
    key = wk.s.dt.floor("min")
    for _, r in relabels.iterrows():
        hit = wk.index[key == r["start"].floor("min")]
        if len(hit) != 1:
            raise ValueError(f"workout_relabel {r['start']} matched {len(hit)} workouts")
        wk.loc[hit[0], "cat"] = r["new_cat2"]
        for entry in RELABEL_LOG:
            if entry["start"] == wk.loc[hit[0], "s"]:
                entry.update(new_cat=r["new_cat2"], via="dated override")

    wmin = np.zeros(len(out)); wcat = np.array([""] * len(out), dtype=object)
    ws = wk.s.values.astype("datetime64[s]").astype("int64")
    we = wk.e.values.astype("datetime64[s]").astype("int64")
    for s, e, cat in zip(ws, we, wk.cat.values):
        i0 = max(np.searchsorted(gsec, s, "right") - 1, 0)
        i1 = min(np.searchsorted(gsec, e, "left"), len(out))
        for i in range(i0, i1):
            lo = gsec[i]
            ov = max(0, min(e, lo + 300) - max(s, lo)) / 60
            if ov > 0:
                wmin[i] += ov
                wcat[i] = cat if wcat[i] == "" else wcat[i]
    out["workout_min"] = np.round(wmin, 1)
    out["workout_cat"] = wcat

    # hours since last workout END; workouts before the fence are visible only before it
    if FENCE_END is not None:
        in_fence = (wk.s < FENCE_END).values
        wend_before, wend_after = np.sort(we[in_fence]), np.sort(we[~in_fence])
    else:
        wend_before, wend_after = np.sort(we), np.sort(we)
    hrs = np.full(len(out), np.nan)
    for k, t_s in enumerate(gsec):
        wend_sorted = wend_before if (fence_s is not None and t_s < fence_s) else wend_after
        j = np.searchsorted(wend_sorted, t_s, "right") - 1
        if j >= 0:
            hrs[k] = (t_s - wend_sorted[j]) / 3600
    out["hrs_since_exercise"] = np.round(hrs, 2)

    # estimated basal IOB: same linear decay over est_basal_u, per window
    K = int(DIA_H * 12)
    w = 1 - (np.arange(K) * 300 + 150) / (DIA_H * 3600)
    ebiob = np.zeros(len(out))
    for wnum in out.window.unique():
        m = (out.window == wnum).to_numpy()
        eb = out.loc[m, "est_basal_u"].fillna(0).to_numpy()
        acc = np.zeros(len(eb))
        for k in range(K):
            acc[k + 1:] += eb[: len(eb) - k - 1] * w[k]
        ebiob[m] = acc
    out["est_basal_iob_u"] = np.round(ebiob, 3)

    # ---------------- flags, target, clean definition ----------------
    out["hour"] = out.t.dt.hour
    out["dow"] = out.t.dt.weekday
    out["g7"] = 0
    if len(WINDOWS) > 1:                       # sensor generation flag: the last non-quarantined window
        modelled = [n for _, _, n in WINDOWS if n != QUARANTINE]
        out["g7"] = (out.window == max(modelled)).astype(int)
    out["dst_day"] = out.t.dt.date.isin(DST_DAYS)
    out["real_gap"] = False
    for _, r in real_gaps.iterrows():
        out.loc[(out.t >= r["start"]) & (out.t <= r["end"]), "real_gap"] = True
    out["pod_change"] = out.frac_uncovered > 0
    out["missing_glucose"] = out.glucose.isna()

    out = out.sort_values("t").reset_index(drop=True)
    tgt = out.glucose.shift(-6)
    same = ((out.t.shift(-6) - out.t == pd.Timedelta("30min")) & (out.window.shift(-6) == out.window))
    out["glucose_t30"] = tgt.where(same)              # targets never cross a window boundary

    have = out.glucose.notna().astype(int)
    past12 = have.rolling(12, min_periods=12).sum()
    contiguous = (out.t - out.t.shift(11) == pd.Timedelta("55min"))
    from_fence = pd.Series(False, index=out.index)
    if FENCE_END is not None:
        from_fence = (out.t >= FENCE_END) & (out.t.shift(11) < FENCE_END)
    out["past_hour_complete"] = (past12 == 12) & contiguous & ~from_fence
    out["clean"] = (out.past_hour_complete & out.glucose_t30.notna()
                    & ~out.dst_day & ~out.real_gap & ~out.pod_change)

    # ---------------- train/test split: fixed once, never moved ----------------
    out["split"] = np.where(out.window == QUARANTINE, "candidate_test",
                            np.where(out.t >= TEST_START, "test",
                                     np.where(out.t + pd.Timedelta("30min") > TEST_START,
                                              "boundary_excluded", "train")))

    hard = wk[wk.cat.isin(HARD_CATS)]
    if FENCE_END is not None:
        exdays_before = set(hard.s[hard.s < FENCE_END].dt.date)
        exdays_after = set(hard.s[hard.s >= FENCE_END].dt.date)
    else:
        exdays_before = exdays_after = set(hard.s.dt.date)

    def day_cat(d):
        exdays = exdays_before if (FENCE_END is not None and d < FENCE_END.date()) else exdays_after
        if d in exdays: return "exercise day"
        if (d - timedelta(days=1)) in exdays: return "day after"
        if (d - timedelta(days=2)) in exdays: return "2 days after"
        return "3+ days / none"
    out["exercise_day_cat"] = out.t.dt.date.map(day_cat)
    return out, wk


if __name__ == "__main__":
    config.ensure_dirs()
    out, wk = build()
    csv_kw = dict(index=False, lineterminator=chr(10))   # LF on every OS
    out.to_csv(PROCESSED_DIR / "aligned_5min.csv", **csv_kw)
    wk.to_csv(PROCESSED_DIR / "workouts_clean.csv", **csv_kw)
    out[out.split == "train"].to_csv(PROCESSED_DIR / "train_5min.csv", **csv_kw)
    out[out.split == "test"].to_csv(PROCESSED_DIR / "test_5min.csv", **csv_kw)

    print(config.describe())
    print("rows:", len(out), "| clean:", int(out.clean.sum()), f"({out.clean.mean()*100:.1f}%)")
    print("\nclean intervals by window:")
    print(out[out.clean].groupby("window").size().to_string())
    print("\nclean intervals (and days represented) by exercise category:")
    c = out[out.clean]
    print(c.groupby("exercise_day_cat").agg(intervals=("t", "size"),
                                            days=("t", lambda x: x.dt.date.nunique())).to_string())
    print("\nexercise days by dominant type:")
    print(wk[wk.cat.isin(HARD_CATS)].groupby(wk.s.dt.date).cat.first().value_counts().to_string())
    print("\nmissing-glucose intervals:", int(out.missing_glucose.sum()),
          "| pod-change intervals:", int(out.pod_change.sum()))
    print("\nmode coverage (hours): activity {:.1f}, limited {:.1f}, manual {:.1f}".format(
        out.frac_activity.sum()/12, out.frac_limited.sum()/12, out.frac_manual.sum()/12))
    print("\nsplit (clean intervals):")
    print(out[out.clean].groupby("split").size().to_string())
    if RELABEL_LOG:
        print("\n'other' workouts and what happened to them:")
        print(pd.DataFrame(RELABEL_LOG).to_string(index=False))
    if QUARANTINE is not None:
        q = out[out.window == QUARANTINE]
        print(f"\nQUARANTINED window {QUARANTINE}: {len(q)} rows, glucose coverage "
              f"{q.glucose.notna().mean()*100:.1f}% (structure only; no target is examined)")
    print("\nest_basal_iob_u: mean {:.2f}, min {:.2f}, max {:.2f}".format(
        out.est_basal_iob_u.mean(), out.est_basal_iob_u.min(), out.est_basal_iob_u.max()))
