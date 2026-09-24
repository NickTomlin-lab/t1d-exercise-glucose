"""
train_quantile.py — uncertainty step: LightGBM quantile models (10th/50th/90th),
band checked by COVERAGE on the frozen test set, overall and below 4 mmol/L.

The band is an 80% interval: the truth should fall inside q10..q90 about
8 times in 10. Coverage below that = overconfident = the dangerous case.

Comparison band ("honest ruler"): last-value prediction +/- the 10th/90th
residual quantiles measured on TRAIN — the simplest band anyone could build.

Settings (tree counts) chosen on the validation window (last 3 weeks of train);
test scored once. Usage: python train_quantile.py [--data-dir <folder>]
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from config import PROCESSED_DIR, S

df = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv",
                 parse_dates=["t"], low_memory=False).sort_values("t").reset_index(drop=True)

# ---------- same feature build as train_lgbm.py ----------
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

y = df.glucose_t30
train_m = (df.split == "train") & df.clean
test_m = (df.split == "test") & df.clean
VAL_START = config.ts("val_start")                   # validation window start (settings)
fit_m = train_m & (df.t < VAL_START)
val_m = train_m & (df.t >= VAL_START)

base_params = dict(objective="quantile", learning_rate=0.05, num_leaves=63,
                   min_child_samples=40, subsample=0.9, colsample_bytree=0.8,
                   n_estimators=3000, random_state=42, verbosity=-1)

preds = {}
for a in (0.1, 0.5, 0.9):
    m = lgb.LGBMRegressor(**base_params, alpha=a, metric="quantile")
    m.fit(F[fit_m], y[fit_m], eval_set=[(F[val_m], y[val_m])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration_
    mf = lgb.LGBMRegressor(**{**base_params, "n_estimators": best}, alpha=a)
    mf.fit(F[train_m], y[train_m])
    preds[a] = mf.predict(F[test_m])
    print(f"alpha {a}: {best} trees")

lo = np.minimum(preds[0.1], preds[0.9])
hi = np.maximum(preds[0.1], preds[0.9])
med = preds[0.5]
yt = y[test_m].to_numpy()
ev = df[test_m]

# honest ruler: last-value +/- train residual quantiles
tr = df[train_m]
res_train = (tr.glucose_t30 - tr.glucose).to_numpy()
qlo, qhi = np.percentile(res_train, 10), np.percentile(res_train, 90)
base_lo = tr_base = ev.glucose.to_numpy() + qlo
base_hi = ev.glucose.to_numpy() + qhi

low = yt < 4
overnight = (ev.t.dt.hour < 6).to_numpy()
post_ex = (ev.hrs_since_exercise <= 4).to_numpy()

def report(name, l, h):
    inside = (yt >= l) & (yt <= h)
    print(f"\n{name}: width mean {np.mean(h-l):.2f} mmol/L")
    for label, m_ in [("overall", np.ones_like(low, bool)), ("below 4", low),
                      ("overnight", overnight), ("post-exercise 4h", post_ex)]:
        print(f"  coverage {label:17s} {inside[m_].mean()*100:5.1f}%  (n={int(m_.sum())}, target 80%)")

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
print(f"\nmedian (q50) RMSE on test: {rmse(yt, med):.3f}")
report("LightGBM q10-q90 band", lo, hi)
report("last-value +/- train quantiles", base_lo, base_hi)

out = ev[["t"]].copy()
out["actual"] = yt; out["q10"] = np.round(lo, 2); out["q50"] = np.round(med, 2); out["q90"] = np.round(hi, 2)
out.to_csv(PROCESSED_DIR / "test_band_predictions.csv", index=False)
