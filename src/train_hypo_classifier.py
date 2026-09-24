"""
train_hypo_classifier.py — option (c): a dedicated "low within 30 minutes?"
classifier, judged on VALIDATION ONLY (the test set is not touched).

Target: will ANY reading in the next 30 minutes (t+5 .. t+30) be below 4 mmol/L?
That is the decision-shaped question — a warning is useful if a low is coming
at all, not only if the exact t+30 reading is low.

Outputs (all measured on the validation window, settings val_start to the test start):
  - Brier score (proper scoring rule) vs a constant-prevalence forecaster
  - reliability table: predicted probability bucket vs observed frequency
  - threshold table: warnings/day vs % of low intervals caught, so the
    "which mistake to make" trade-off is an explicit choice, not a default.

Usage: python train_hypo_classifier.py [--data-dir <folder>]
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

import config
from config import PROCESSED_DIR, S

df = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv",
                 parse_dates=["t"], low_memory=False).sort_values("t").reset_index(drop=True)

# ---------- same feature build as the other models ----------
F = pd.DataFrame(index=df.index)
g = df.glucose
for k in range(12):
    F[f"g_lag{k}"] = g.shift(k)
F["slope_15"] = (g - g.shift(3)) / 3
F["slope_30"] = (g - g.shift(6)) / 6
F["slope_60"] = (g - g.shift(11)) / 11
F["accel"] = F.slope_15 - (g.shift(3) - g.shift(6)) / 3
F["g_mean_1h"] = g.rolling(12).mean()
F["g_std_1h"] = g.rolling(12).std()
F["below_range"] = df.below_range.fillna(False).astype(int)
F["bolus_iob"] = df.bolus_iob_u
F["basal_iob"] = df.est_basal_iob_u
F["bolus_30m"] = df.bolus_u.rolling(6, min_periods=1).sum()
F["bolus_2h"] = df.bolus_u.rolling(24, min_periods=1).sum()
F["carbs_30m"] = df.carbs_g.rolling(6, min_periods=1).sum()
F["carbs_2h"] = df.carbs_g.rolling(24, min_periods=1).sum()
F["carbs_4h"] = df.carbs_g.rolling(48, min_periods=1).sum()
last_carb = df.t.where(df.carbs_g > 0).ffill()
F["min_since_carbs"] = (df.t - last_carb).dt.total_seconds() / 60
last_bolus = df.t.where(df.bolus_u > 0).ffill()
F["min_since_bolus"] = (df.t - last_bolus).dt.total_seconds() / 60
F["frac_suspend"] = df.frac_suspend
F["frac_max"] = df.frac_max
F["suspend_1h"] = df.frac_suspend.rolling(12, min_periods=1).mean()
F["frac_activity"] = df.frac_activity
F["activity_2h"] = df.frac_activity.rolling(24, min_periods=1).mean()
F["frac_limited"] = df.frac_limited
F["workout_now"] = df.workout_min
F["workout_1h"] = df.workout_min.rolling(12, min_periods=1).sum()
F["workout_24h"] = df.workout_min.rolling(288, min_periods=1).sum()
F["hrs_since_ex"] = df.hrs_since_exercise.clip(upper=72)
F["hr"] = df.hr_mean
F["hr_missing"] = df.hr_missing.astype(int)
F["steps_1h"] = df.steps.rolling(12, min_periods=1).sum()
F["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
F["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
F["dow"] = df.dow
F["g7"] = df.g7

# ---------- target: any reading below 4 in the next 30 min ----------
fut_min = pd.concat({k: g.shift(-k) for k in range(1, 7)}, axis=1).min(axis=1)
contig = (df.t.shift(-6) - df.t) == pd.Timedelta("30min")   # same window, no gap
ylow = (fut_min < 4).astype(int)

train_m = (df.split == "train") & df.clean & contig
VAL_START = config.ts("val_start")                   # validation window start (settings)
fit_m = train_m & (df.t < VAL_START)
val_m = train_m & (df.t >= VAL_START)

print(f"fit n={int(fit_m.sum())} (lows {ylow[fit_m].mean()*100:.1f}%) | "
      f"val n={int(val_m.sum())} (lows {ylow[val_m].mean()*100:.1f}%)")

params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.9, colsample_bytree=0.8,
              n_estimators=3000, random_state=42, verbosity=-1)
m = lgb.LGBMClassifier(**params)
m.fit(F[fit_m], ylow[fit_m], eval_set=[(F[val_m], ylow[val_m])],
      callbacks=[lgb.early_stopping(100, verbose=False)])
print("early stopping chose", m.best_iteration_, "trees")

p_raw = m.predict_proba(F[val_m])[:, 1]
yv = ylow[val_m].to_numpy()

# isotonic calibration fitted on the FIRST HALF of validation, reported on the second
half = len(p_raw) // 2
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_raw[:half], yv[:half])
p = iso.predict(p_raw[half:])
yh = yv[half:]
n_days = (df[val_m].t.iloc[-1] - df[val_m].t.iloc[half]).days + 1

brier = np.mean((p - yh) ** 2)
prev = yh.mean()
brier_const = np.mean((prev - yh) ** 2)
print(f"\nBrier score {brier:.4f} vs constant-prevalence {brier_const:.4f} "
      f"(lower is better; prevalence {prev*100:.1f}%)")

print("\nreliability (calibrated, second half of validation):")
bins = [0, .05, .1, .2, .4, .6, 1.01]
for a, b in zip(bins[:-1], bins[1:]):
    m_ = (p >= a) & (p < b)
    if m_.sum() > 0:
        print(f"  predicted {a:.2f}-{b:.2f}: observed {yh[m_].mean()*100:5.1f}%  (n={int(m_.sum())})")

print(f"\nthreshold table (second half of validation, ~{n_days} days):")
print(f"  {'threshold':>9} {'warnings/day':>12} {'% low intervals caught':>23} {'precision':>10}")
for thr in (0.5, 0.3, 0.2, 0.1, 0.05):
    warn = p >= thr
    caught = (warn & (yh == 1)).sum() / max(yh.sum(), 1)
    prec = yh[warn].mean() * 100 if warn.sum() else float("nan")
    print(f"  {thr:9.2f} {warn.sum()/n_days:12.1f} {caught*100:23.1f} {prec:9.1f}%")

imp = pd.Series(m.feature_importances_, index=F.columns).sort_values(ascending=False)
print("\ntop 8 features:", ", ".join(imp.head(8).index))

# ---------- episode-level view (the human-scale numbers) ----------
# A LOW EPISODE = maximal run of actual readings < 4 (gaps <= 15 min merged).
# It is CAUGHT if any warning fired in the 30 minutes before it began.
# A WARNING EPISODE = maximal run of warn intervals (so one sustained alert,
# however long, counts once).
vd = df[val_m].iloc[half:].reset_index(drop=True)
gl = vd.glucose.to_numpy()
tt = vd.t.to_numpy()

def episodes(mask, ts, max_gap_min=15):
    eps, start, last = [], None, None
    for i, on in enumerate(mask):
        if on:
            if start is None or (ts[i] - last) / np.timedelta64(1, "m") > max_gap_min:
                if start is not None:
                    eps.append((start, last))
                start = ts[i]
            last = ts[i]
    if start is not None:
        eps.append((start, last))
    return eps

low_eps = episodes(gl < 4, tt)
print(f"\nepisode view (second half of validation, ~{n_days} days):")
print(f"  actual low episodes: {len(low_eps)}")
for thr in (0.5, 0.3, 0.2):
    warn = p >= thr
    warn_eps = episodes(warn, tt)
    caught = 0
    for s, e in low_eps:
        pre = (tt >= s - np.timedelta64(30, "m")) & (tt < s)
        if warn[pre].any() or warn[(tt >= s) & (tt <= e)].any():
            caught += 1
    print(f"  threshold {thr:.2f}: {len(warn_eps)/n_days:.1f} alert episodes/day, "
          f"low episodes caught {caught}/{len(low_eps)}")
