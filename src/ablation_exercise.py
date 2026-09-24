"""
ablation_exercise.py — the Module 1 question: does exercise information
actually improve 30-min glucose prediction, or is the model just a
glucose-trend predictor (a Dexcom-equivalent) with decoration?

VALIDATION ONLY (the test set is retired). Protocol identical for every
variant: fit before the validation window, early stopping on the first half of validation,
all reported numbers from the second half (never seen by any variant).

Variants (cumulative feature sets):
  A  glucose + time only        — the Dexcom-equivalent: lags, slopes,
                                   rolling stats, hour, day-of-week
  B  A + insulin & carbs        — IOB, boluses, carbs, basal states, modes
  C  B + exercise & movement    — the current full model: workout minutes/
                                   recency, HR, steps
  D  C + exercise TYPE split    — aerobic (football/cricket/walking) vs
                                   anaerobic (strength/HIIT/gym) recency and
                                   24h/48h volumes, separately

Read-outs, point model: RMSE on the evaluation half, overall and by
exercise-recency bucket (0-4h / 4-12h / 12-24h / 24-48h / >48h since last
hard workout) and by TYPE of the most recent hard workout.
Read-out, hypo classifier: Brier + low episodes caught for A vs C vs D.

Usage: python ablation_exercise.py [--data-dir <folder>]
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from config import PROCESSED_DIR, S

df = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv",
                 parse_dates=["t"], low_memory=False).sort_values("t").reset_index(drop=True)
wk = pd.read_csv(PROCESSED_DIR / "workouts_clean.csv", parse_dates=["s", "e"])

g = df.glucose
A = pd.DataFrame(index=df.index)                     # glucose + time
for k in range(12):
    A[f"g_lag{k}"] = g.shift(k)
A["slope_15"] = (g - g.shift(3)) / 3
A["slope_30"] = (g - g.shift(6)) / 6
A["slope_60"] = (g - g.shift(11)) / 11
A["accel"] = A.slope_15 - (g.shift(3) - g.shift(6)) / 3
A["g_mean_1h"] = g.rolling(12).mean()
A["g_std_1h"] = g.rolling(12).std()
A["below_range"] = df.below_range.fillna(False).astype(int)
A["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
A["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
A["dow"] = df.dow
A["g7"] = df.g7

B_extra = pd.DataFrame(index=df.index)               # + insulin & carbs
B_extra["bolus_iob"] = df.bolus_iob_u
B_extra["basal_iob"] = df.est_basal_iob_u
B_extra["bolus_30m"] = df.bolus_u.rolling(6, min_periods=1).sum()
B_extra["bolus_2h"] = df.bolus_u.rolling(24, min_periods=1).sum()
B_extra["carbs_30m"] = df.carbs_g.rolling(6, min_periods=1).sum()
B_extra["carbs_2h"] = df.carbs_g.rolling(24, min_periods=1).sum()
B_extra["carbs_4h"] = df.carbs_g.rolling(48, min_periods=1).sum()
B_extra["min_since_carbs"] = (df.t - df.t.where(df.carbs_g > 0).ffill()).dt.total_seconds() / 60
B_extra["min_since_bolus"] = (df.t - df.t.where(df.bolus_u > 0).ffill()).dt.total_seconds() / 60
B_extra["frac_suspend"] = df.frac_suspend
B_extra["frac_max"] = df.frac_max
B_extra["suspend_1h"] = df.frac_suspend.rolling(12, min_periods=1).mean()
B_extra["frac_activity"] = df.frac_activity
B_extra["activity_2h"] = df.frac_activity.rolling(24, min_periods=1).mean()
B_extra["frac_limited"] = df.frac_limited

C_extra = pd.DataFrame(index=df.index)               # + exercise & movement
C_extra["workout_now"] = df.workout_min
C_extra["workout_1h"] = df.workout_min.rolling(12, min_periods=1).sum()
C_extra["workout_24h"] = df.workout_min.rolling(288, min_periods=1).sum()
C_extra["hrs_since_ex"] = df.hrs_since_exercise.clip(upper=72)
C_extra["hr"] = df.hr_mean
C_extra["hr_missing"] = df.hr_missing.astype(int)
C_extra["steps_1h"] = df.steps.rolling(12, min_periods=1).sum()

# type-split exercise minutes on the grid (aerobic vs anaerobic)
AEROBIC = {"football", "cricket", "walking"}
gsec = df.t.values.astype("datetime64[s]").astype("int64")
aer = np.zeros(len(df)); anae = np.zeros(len(df))
ws = wk.s.values.astype("datetime64[s]").astype("int64")
we = wk.e.values.astype("datetime64[s]").astype("int64")
for s, e, cat in zip(ws, we, wk.cat.values):
    tgt = aer if cat in AEROBIC else anae
    i0 = max(np.searchsorted(gsec, s, "right") - 1, 0)
    i1 = min(np.searchsorted(gsec, e, "left"), len(df))
    for i in range(i0, i1):
        tgt[i] += max(0, min(e, gsec[i] + 300) - max(s, gsec[i])) / 60

D_extra = pd.DataFrame(index=df.index)               # + type split
D_extra["aer_min_now"] = aer
D_extra["aer_24h"] = pd.Series(aer).rolling(288, min_periods=1).sum().values
D_extra["aer_48h"] = pd.Series(aer).rolling(576, min_periods=1).sum().values
D_extra["anae_24h"] = pd.Series(anae).rolling(288, min_periods=1).sum().values
D_extra["anae_48h"] = pd.Series(anae).rolling(576, min_periods=1).sum().values
for typ, sub in [("aer", wk[wk.cat.isin(AEROBIC)]), ("anae", wk[~wk.cat.isin(AEROBIC)])]:
    ends = np.sort(sub.e.values.astype("datetime64[s]").astype("int64"))
    hrs = np.full(len(df), 999.0)
    j = np.searchsorted(ends, gsec, "right") - 1
    ok = j >= 0
    hrs[ok] = (gsec[ok] - ends[j[ok]]) / 3600
    D_extra[f"hrs_since_{typ}"] = np.clip(hrs, 0, 72)

variants = {
    "A glucose+time (Dexcom-equiv)": A,
    "B + insulin/carbs": pd.concat([A, B_extra], axis=1),
    "C + exercise (current model)": pd.concat([A, B_extra, C_extra], axis=1),
    "D + aerobic/anaerobic split": pd.concat([A, B_extra, C_extra, D_extra], axis=1),
}

y = df.glucose_t30
train_m = (df.split == "train") & df.clean
VAL_START = config.ts("val_start")                   # validation window start (settings)
fit_m = train_m & (df.t < VAL_START)
val_m = train_m & (df.t >= VAL_START)
vi = df.index[val_m]; half = len(vi) // 2
es_i, ev_i = vi[:half], vi[half:]
ye = y[ev_i].to_numpy()

# recency buckets + last-hard-workout type on the evaluation half
hard = wk[wk.cat.isin(["cricket", "football", "gym"])]
h_ends = np.sort(hard.e.values.astype("datetime64[s]").astype("int64"))
h_cats = hard.sort_values("e").cat.values
esec = df.t[ev_i].values.astype("datetime64[s]").astype("int64")
j = np.searchsorted(h_ends, esec, "right") - 1
hrs_since_hard = np.where(j >= 0, (esec - h_ends[np.clip(j, 0, None)]) / 3600, 999)
last_type = np.where(j >= 0, h_cats[np.clip(j, 0, None)], "none")
last_aer = np.isin(last_type, ["cricket", "football"])
buckets = [("0-4h", hrs_since_hard <= 4), ("4-12h", (hrs_since_hard > 4) & (hrs_since_hard <= 12)),
           ("12-24h", (hrs_since_hard > 12) & (hrs_since_hard <= 24)),
           ("24-48h", (hrs_since_hard > 24) & (hrs_since_hard <= 48)),
           (">48h", hrs_since_hard > 48)]

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

common = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40,
              subsample=0.9, colsample_bytree=0.8, random_state=42, verbosity=-1)

print("POINT MODEL ABLATION — evaluation half of validation "
      f"(n={len(ev_i)}); RMSE in mmol/L")
hdr = "  ".join(f"{b[0]:>7}" for b in buckets)
print(f"{'variant':32s} {'overall':>7}  {hdr}  {'aer<12h':>8} {'anae<12h':>8}")
preds = {}
for name, X in variants.items():
    m = lgb.LGBMRegressor(objective="regression", n_estimators=3000, **common)
    m.fit(X.loc[fit_m], y[fit_m], eval_set=[(X.loc[es_i], y[es_i])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    mf = lgb.LGBMRegressor(objective="regression", n_estimators=m.best_iteration_, **common)
    mf.fit(X.loc[fit_m], y[fit_m])
    p = mf.predict(X.loc[ev_i]); preds[name] = p
    cells = "  ".join(f"{rmse(ye[b[1]], p[b[1]]):7.3f}" for b in buckets)
    a12 = rmse(ye[last_aer & (hrs_since_hard <= 12)], p[last_aer & (hrs_since_hard <= 12)])
    n12 = rmse(ye[~last_aer & (hrs_since_hard <= 12) & (hrs_since_hard < 900)],
               p[~last_aer & (hrs_since_hard <= 12) & (hrs_since_hard < 900)])
    print(f"{name:32s} {rmse(ye, p):7.3f}  {cells}  {a12:8.3f} {n12:8.3f}")
print("bucket sizes:", ", ".join(f"{b[0]} n={int(b[1].sum())}" for b in buckets),
      f"| aer<12h n={int((last_aer & (hrs_since_hard<=12)).sum())}",
      f"anae<12h n={int((~last_aer & (hrs_since_hard<=12)).sum())}")

# ---------- classifier ablation (lows are where exercise should matter) ----------
ylow = (pd.concat({k: g.shift(-k) for k in range(1, 7)}, axis=1).min(axis=1) < 4).astype(int)
contig = (df.t.shift(-6) - df.t) == pd.Timedelta("30min")
yv = ylow[ev_i].to_numpy()
gl, tt = df.glucose[ev_i].to_numpy(), df.t[ev_i].values

def episodes(mask, ts, max_gap_min=15):
    eps, start, last = [], None, None
    for i, on in enumerate(mask):
        if on:
            if start is None or (ts[i] - last) / np.timedelta64(1, "m") > max_gap_min:
                if start is not None: eps.append((start, last))
                start = ts[i]
            last = ts[i]
    if start is not None: eps.append((start, last))
    return eps

low_eps = episodes(gl < 4, tt)
n_days = (tt[-1] - tt[0]) / np.timedelta64(1, "D") + 1
print(f"\nHYPO CLASSIFIER ABLATION — same half; {len(low_eps)} low episodes, ~{n_days:.0f} days")
for name in ["A glucose+time (Dexcom-equiv)", "C + exercise (current model)",
             "D + aerobic/anaerobic split"]:
    X = variants[name]
    c = lgb.LGBMClassifier(objective="binary", n_estimators=3000, **common)
    c.fit(X.loc[fit_m & contig], ylow[fit_m & contig],
          eval_set=[(X.loc[es_i], ylow[es_i])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    cf = lgb.LGBMClassifier(objective="binary", n_estimators=c.best_iteration_, **common)
    cf.fit(X.loc[fit_m & contig], ylow[fit_m & contig])
    p = cf.predict_proba(X.loc[ev_i])[:, 1]
    # pick per-variant threshold giving ~2 alert episodes/day, then count catches
    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr, best = None, None
    for thr in thr_grid:
        weps = episodes(p >= thr, tt)
        if len(weps) / n_days <= 2.1:
            best_thr = thr; break
    warn = p >= (best_thr if best_thr is not None else 0.5)
    caught = sum(1 for s, e in low_eps
                 if warn[(tt >= s - np.timedelta64(30, "m")) & (tt <= e)].any())
    print(f"  {name:32s} Brier {np.mean((p-yv)**2):.4f} | at ~2 alerts/day "
          f"(thr {best_thr}): caught {caught}/{len(low_eps)}")
