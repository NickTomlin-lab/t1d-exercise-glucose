"""
make_synthetic.py - a SYNTHETIC 14-day sample in the shape of the real aligned
5-minute table, so the pipeline can be run without any private data.

THIS FILE IS SYNTHETIC. Every row carries synthetic = 1. Nothing in it is a
real reading, dose, workout or timestamp; the published results in results/
come from the real data, not from this sample.

How it is made
  Summary statistics of the REAL train rows (means, standard deviations, the
  lag-1 autocorrelation of glucose, bolus and carb frequencies, workout
  frequency and typical duration by category, heart-rate and step levels,
  pump suspend share) were computed once with `--from-real <aligned_5min.csv>`
  and saved to data/synthetic/summary_stats.json. Those are aggregate numbers
  only: no dates, no clock times, no individual events. The generator then
  draws 14 days on a 5-minute grid from them:
    - glucose: an AR(1) process around the real mean with the real lag-1
      autocorrelation, plus a rise after each carb entry and a fall during and
      after exercise, clipped to the sensor range 2.2-22.2 mmol/L;
    - boluses and carbs: Poisson events at random daytime minutes;
    - workouts: per day and category, Bernoulli with the real frequency; the
      START CLOCK TIME IS UNIFORM BETWEEN 07:00 AND 21:00 so the sample carries
      no weekly routine; two fixed sessions are planted so the example
      overrides file has something to match;
    - heart rate, steps, pump suspend/automated state and Activity mode: noise
      around the real levels, elevated or shifted around workouts;
    - IOB columns, hrs_since_exercise, flags, the 30-min target, the clean rule
      and the train/test split are computed exactly as build_dataset.py does.
Dates are placeholders in the year 2000 (settings default: window 1 =
2000-01-03 to 2000-01-17, test from 2000-01-14).

Usage:
  python make_synthetic.py                       # regenerate data/synthetic from summary_stats.json
  python make_synthetic.py --from-real <csv>     # (private) recompute summary_stats.json first
"""
import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

import config
from config import DATA_DIR, PROCESSED_DIR, S

STATS_PATH = DATA_DIR / "summary_stats.json"
SEED = 2000
CATS = ["gym", "cricket", "football", "walking"]
ALIGNED_COLS = ["t", "window", "glucose", "glucose_src", "below_range", "bolus_u", "carbs_g", "bolus_iob_u",
                "frac_suspend", "frac_auto", "frac_max", "frac_uncovered", "est_basal_u", "frac_activity",
                "frac_limited", "frac_manual", "hr_mean", "hr_n", "hr_missing", "steps", "workout_min",
                "workout_cat", "hrs_since_exercise", "est_basal_iob_u", "hour", "dow", "g7", "dst_day",
                "real_gap", "pod_change", "missing_glucose", "glucose_t30", "past_hour_complete", "clean",
                "split", "exercise_day_cat", "synthetic"]
# planted sessions (start clock time, category, duration min) so the example
# overrides in data/overrides_example.csv match something
PLANTED = [("2000-01-05 18:00", "cricket", 120), ("2000-01-09 17:00", "gym", 90)]


def summarise_real(path):
    """Aggregate statistics of the real TRAIN rows. No dates, no events."""
    df = pd.read_csv(path, parse_dates=["t"], low_memory=False)
    tr = df[df["split"] == "train"].copy()
    g = tr["glucose"]
    lag1 = g.autocorr(lag=1)
    days = tr["t"].dt.date.nunique()
    bol = tr[tr["bolus_u"] > 0]
    wk = tr[(tr["workout_min"] > 0)]
    # workout runs: count starts per category and their duration
    starts = wk[(wk["workout_cat"] != wk["workout_cat"].shift(1)) | (wk["t"] - wk["t"].shift(1) != pd.Timedelta("5min"))]
    runs = {}
    for cat in CATS:
        sub = wk[wk["workout_cat"] == cat]
        n_start = int((starts["workout_cat"] == cat).sum())
        runs[cat] = dict(per_day=n_start / days,
                         duration_min=float(5 * len(sub) / n_start) if n_start else 60.0)
    hr_rest = tr.loc[tr["workout_min"] == 0, "hr_mean"].dropna()
    hr_ex = tr.loc[tr["workout_min"] > 0, "hr_mean"].dropna()
    stats = dict(
        glucose_mean=float(g.mean()), glucose_sd=float(g.std()), glucose_lag1=float(lag1),
        glucose_missing_share=float(g.isna().mean()),
        below4_share=float((g < 4).mean()),
        bolus_per_day=float(len(bol) / days), bolus_u_mean=float(bol["bolus_u"].mean()),
        bolus_u_sd=float(bol["bolus_u"].std()),
        carbs_share_of_boluses=float((bol["carbs_g"] > 0).mean()),
        carbs_g_mean=float(bol.loc[bol["carbs_g"] > 0, "carbs_g"].mean()),
        carbs_g_sd=float(bol.loc[bol["carbs_g"] > 0, "carbs_g"].std()),
        workouts=runs,
        hr_rest_mean=float(hr_rest.mean()), hr_rest_sd=float(hr_rest.std()),
        hr_exercise_mean=float(hr_ex.mean()), hr_missing_share=float(tr["hr_mean"].isna().mean()),
        steps_rest_mean=float(tr.loc[tr["workout_min"] == 0, "steps"].mean()),
        steps_exercise_mean=float(tr.loc[tr["workout_min"] > 0, "steps"].mean()),
        frac_suspend_mean=float(tr["frac_suspend"].mean()), frac_max_mean=float(tr["frac_max"].mean()),
        frac_uncovered_share=float((tr["frac_uncovered"] > 0).mean()),
        est_basal_u_mean=float(tr["est_basal_u"].mean()),
        activity_share_of_workouts=float(
            wk.groupby((wk["t"] - wk["t"].shift(1) != pd.Timedelta("5min")).cumsum())["frac_activity"].max().gt(0).mean()),
        dia_hours=float(S["dia_hours"]),
        note="aggregate statistics of the real train rows; no dates or events",
    )
    return stats


def generate(stats, rng):
    w0, w1, wnum = config.windows()[0]
    idx = pd.date_range(w0, w1, freq="5min", inclusive="left")
    n = len(idx)
    out = pd.DataFrame({"t": idx, "window": wnum})
    days = sorted(set(idx.date))
    dia_rows = int(stats["dia_hours"] * 12)

    # ---- workouts ----
    workouts = []
    for d in days:
        for cat in CATS:
            if rng.random() < stats["workouts"][cat]["per_day"]:
                start_min = int(rng.uniform(7 * 60, 21 * 60)) // 5 * 5
                dur = max(15, int(rng.normal(stats["workouts"][cat]["duration_min"], 10)) // 5 * 5)
                s = pd.Timestamp(d) + pd.Timedelta(minutes=start_min)
                workouts.append((s, s + pd.Timedelta(minutes=dur), cat))
    for s, cat, dur in PLANTED:
        s = pd.Timestamp(s)
        workouts.append((s, s + pd.Timedelta(minutes=dur), cat))
    workouts.sort()
    kept = []
    for s, e, cat in workouts:                       # drop overlaps
        if kept and s < kept[-1][1] + pd.Timedelta(minutes=35):
            continue
        if e > w1:
            continue
        kept.append((s, e, cat))
    wk = pd.DataFrame(kept, columns=["s", "e", "cat"])
    wk["typ"] = wk["cat"].map({"gym": "TraditionalStrengthTraining", "cricket": "Cricket",
                               "football": "Football", "walking": "Walking"})
    wmin = np.zeros(n); wcat = np.array([""] * n, dtype=object)
    for s, e, cat in kept:
        m = (idx >= s) & (idx < e)
        wmin[m] = 5.0
        wcat[m] = cat
    out["workout_min"] = wmin
    out["workout_cat"] = wcat
    hard = wk[wk["cat"].isin(["gym", "cricket", "football"])]

    # ---- boluses and carbs ----
    bolus = np.zeros(n); carbs = np.zeros(n)
    for d in days:
        k = rng.poisson(stats["bolus_per_day"])
        for _ in range(k):
            m = int(rng.uniform(6 * 60, 23 * 60)) // 5
            i = idx.searchsorted(pd.Timestamp(d) + pd.Timedelta(minutes=5 * m))
            if i < n:
                bolus[i] += max(0.1, rng.normal(stats["bolus_u_mean"], stats["bolus_u_sd"]))
                if rng.random() < stats["carbs_share_of_boluses"]:
                    carbs[i] += max(5, rng.normal(stats["carbs_g_mean"], stats["carbs_g_sd"]))
    out["bolus_u"] = np.round(bolus, 2)
    out["carbs_g"] = np.round(carbs, 1)

    # ---- pump state ----
    ex_or_post = np.zeros(n, bool)
    for s, e, cat in kept:
        if cat != "walking":
            ex_or_post |= (idx >= s - pd.Timedelta(minutes=15)) & (idx < e + pd.Timedelta(hours=2))
    # suspend runs persist (the pump pauses delivery in runs, not single pulses);
    # around a hard workout: a quiet stretch before, then a cut after the start
    frac_suspend = np.zeros(n)
    state = rng.random() < stats["frac_suspend_mean"]
    for i in range(n):
        if rng.random() < 0.08:
            state = rng.random() < stats["frac_suspend_mean"]
        frac_suspend[i] = 1.0 if state else 0.0
    for s, e, cat in kept:
        if cat != "walking":
            frac_suspend[(idx >= s - pd.Timedelta(minutes=40)) & (idx < s + pd.Timedelta(minutes=10))] = 0.0
            if rng.random() < 0.85:
                frac_suspend[(idx >= s + pd.Timedelta(minutes=10)) & (idx < e + pd.Timedelta(minutes=30))] = 1.0
    frac_max = np.where(frac_suspend == 0, (rng.random(n) < stats["frac_max_mean"] * 2).astype(float), 0.0)
    frac_uncovered = np.where(rng.random(n) < stats["frac_uncovered_share"] / 3, rng.uniform(0.2, 1.0, n), 0.0)
    frac_auto = np.clip(1 - frac_suspend - frac_max - frac_uncovered, 0, 1)
    out["frac_suspend"] = np.round(frac_suspend * (1 - frac_uncovered), 3)
    out["frac_auto"] = np.round(frac_auto, 3)
    out["frac_max"] = np.round(frac_max * (1 - frac_uncovered), 3)
    out["frac_uncovered"] = np.round(frac_uncovered, 3)
    rate = stats["est_basal_u_mean"] * 12 / max(1 - stats["frac_suspend_mean"], 0.2)
    out["est_basal_u"] = np.round(out["frac_auto"] * rate / 12 + out["frac_max"] * 1.0 / 12, 4)
    act = np.zeros(n)
    for s, e, cat in kept:
        if cat != "walking" and rng.random() < stats["activity_share_of_workouts"]:
            lead = int(rng.choice([0, 30, 60]))
            act[(idx >= s - pd.Timedelta(minutes=lead)) & (idx < e + pd.Timedelta(minutes=60))] = 1.0
    out["frac_activity"] = act
    out["frac_limited"] = 0.0
    out["frac_manual"] = 0.0

    # ---- glucose: AR(1) + meal rise + exercise fall ----
    phi = float(np.clip(stats["glucose_lag1"], 0.5, 0.995))
    mu, sd = stats["glucose_mean"], stats["glucose_sd"]
    eps_sd = sd * np.sqrt(1 - phi ** 2)
    meal = np.zeros(n); exer = np.zeros(n)
    for i in np.flatnonzero(carbs > 0):
        rise = carbs[i] / 12.0
        for k in range(1, 37):
            if i + k < n:
                meal[i + k] += rise * (k / 12 if k <= 12 else max(0, (36 - k) / 24))
    for s, e, cat in kept:
        if cat != "walking":
            span = (idx >= s) & (idx < e + pd.Timedelta(hours=2))
            exer[span] -= rng.uniform(1.0, 3.0)
    g = np.zeros(n)
    g[0] = mu
    for i in range(1, n):
        g[i] = mu + phi * (g[i - 1] - mu) + rng.normal(0, eps_sd) + 0.08 * (meal[i] - meal[i - 1]) * 12 + 0.06 * (exer[i] - exer[i - 1]) * 12
    g = g + 0.35 * meal + 0.5 * exer
    g = np.clip(g, 2.2, 22.2)
    missing = rng.random(n) < stats["glucose_missing_share"]
    real_gaps = config.load_overrides("real_gap")
    real_gap = np.zeros(n, bool)
    for _, r in real_gaps.iterrows():
        real_gap |= np.asarray((idx >= r["start"]) & (idx <= r["end"]))
    missing |= real_gap
    g[missing] = np.nan
    out["glucose"] = np.round(g, 1)
    out["glucose_src"] = np.where(missing, "", "synthetic")
    out["below_range"] = out["glucose"] <= 2.2

    # ---- IOB (linear decay, as build_dataset.py) ----
    w = 1 - np.arange(dia_rows) * 5 / (stats["dia_hours"] * 60)
    iob = np.zeros(n)
    for k in range(dia_rows):
        iob[k:] += bolus[: n - k] * w[k]
    out["bolus_iob_u"] = np.round(iob, 3)
    wb = 1 - (np.arange(dia_rows) * 300 + 150) / (stats["dia_hours"] * 3600)
    eb = out["est_basal_u"].to_numpy()
    ebiob = np.zeros(n)
    for k in range(dia_rows):
        ebiob[k + 1:] += eb[: n - k - 1] * wb[k]
    out["est_basal_iob_u"] = np.round(ebiob, 3)

    # ---- heart rate and steps ----
    hr = stats["hr_rest_mean"] + rng.normal(0, stats["hr_rest_sd"] * 0.6, n)
    hr += 6 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60 - 15) / 24)   # lower at night
    for s, e, cat in kept:
        m = (idx >= s) & (idx < e)
        hr[m] = stats["hr_exercise_mean"] + rng.normal(0, 8, m.sum())
        m2 = (idx >= e) & (idx < e + pd.Timedelta(minutes=30))
        hr[m2] += 15
    hr_missing = rng.random(n) < stats["hr_missing_share"]
    hr_missing &= wmin == 0
    hr[hr_missing] = np.nan
    out["hr_mean"] = np.round(hr, 1)
    out["hr_n"] = np.where(hr_missing, np.nan, rng.integers(1, 6, n)).astype(float)
    out["hr_missing"] = hr_missing
    day_share = np.where((idx.hour >= 7) & (idx.hour < 23), 1.0, 0.05)
    steps = rng.poisson(stats["steps_rest_mean"] * day_share)
    for s, e, cat in kept:
        m = (idx >= s) & (idx < e)
        steps[m] = rng.poisson(max(stats["steps_exercise_mean"], 250), m.sum())
        pre = (idx >= s - pd.Timedelta(minutes=20)) & (idx < s)
        steps[pre] = rng.poisson(220, pre.sum())
    out["steps"] = steps.astype(float)

    # ---- hours since last hard workout end ----
    ends = np.sort(hard["e"].values.astype("datetime64[ns]"))
    hrs = np.full(n, np.nan)
    if len(ends):
        j = np.searchsorted(ends, idx.values, "right") - 1
        ok = j >= 0
        hrs[ok] = (idx.values[ok] - ends[j[ok]]) / np.timedelta64(1, "h")
    out["hrs_since_exercise"] = np.round(hrs, 2)

    # ---- flags, target, clean, split, exercise day ----
    out["hour"] = idx.hour
    out["dow"] = idx.weekday
    out["g7"] = 0
    dst = {pd.Timestamp(d).date() for d in S.get("dst_days", [])}
    out["dst_day"] = pd.Series(idx.date).isin(dst).values
    out["real_gap"] = real_gap
    out["pod_change"] = out["frac_uncovered"] > 0
    out["missing_glucose"] = out["glucose"].isna()
    out["glucose_t30"] = out["glucose"].shift(-6)
    have = out["glucose"].notna().astype(int)
    out["past_hour_complete"] = have.rolling(12, min_periods=12).sum().eq(12)
    out["clean"] = (out["past_hour_complete"] & out["glucose_t30"].notna() & ~out["dst_day"]
                    & ~out["real_gap"] & ~out["pod_change"])
    test_start = config.ts("test_start")
    out["split"] = np.where(out["t"] >= test_start, "test",
                            np.where(out["t"] + pd.Timedelta("30min") > test_start, "boundary_excluded", "train"))
    exdays = set(hard["s"].dt.date)

    def day_cat(d):
        if d in exdays: return "exercise day"
        if (d - timedelta(days=1)) in exdays: return "day after"
        if (d - timedelta(days=2)) in exdays: return "2 days after"
        return "3+ days / none"
    out["exercise_day_cat"] = pd.Series(idx.date).map(day_cat).values
    out["synthetic"] = 1
    return out[ALIGNED_COLS], wk[["s", "e", "typ", "cat"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-real", help="(private) real aligned_5min.csv to recompute summary_stats.json from")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()
    if args.from_real:
        stats = summarise_real(args.from_real)
        with open(STATS_PATH, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"wrote {STATS_PATH} (aggregate statistics only)")
    with open(STATS_PATH, encoding="utf-8") as fh:
        stats = json.load(fh)
    rng = np.random.default_rng(SEED)
    out, wk = generate(stats, rng)
    csv_kw = dict(index=False, lineterminator=chr(10))
    out.to_csv(DATA_DIR / "aligned_5min_synthetic.csv", **csv_kw)     # the shipped copy
    out.to_csv(PROCESSED_DIR / "aligned_5min.csv", **csv_kw)           # what the pipeline reads
    wk.to_csv(PROCESSED_DIR / "workouts_clean.csv", **csv_kw)
    out[out["split"] == "train"].to_csv(PROCESSED_DIR / "train_5min.csv", **csv_kw)
    out[out["split"] == "test"].to_csv(PROCESSED_DIR / "test_5min.csv", **csv_kw)
    print(f"SYNTHETIC sample written to {DATA_DIR / 'aligned_5min_synthetic.csv'} and copied to "
          f"{PROCESSED_DIR / 'aligned_5min.csv'}: {len(out)} rows, "
          f"{out['t'].dt.date.nunique()} days, {len(wk)} workouts "
          f"({', '.join(f'{c}={n}' for c, n in wk['cat'].value_counts().items())})")
    print(f"clean rows {int(out['clean'].sum())}, glucose < 4 share {(out['glucose'] < 4).mean():.3f}, "
          f"split: {out['split'].value_counts().to_dict()}")
    print("every row carries synthetic = 1; nothing here is a real reading")


if __name__ == "__main__":
    main()
