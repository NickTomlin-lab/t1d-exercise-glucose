"""
train_lgbm.py — first real model: LightGBM on engineered features, judged ONCE
against the frozen test set and the last-value bar (RMSE 1.534).

Discipline (Modelling Principles):
  - Features use ONLY the past (shifts/rolling over earlier rows).
  - Validation (for early stopping only) = last 3 weeks of TRAIN, time-ordered.
    The test set is scored once, at the end, with the settings already fixed.
  - Report by regime, not one number.

Usage: python train_lgbm.py [--data-dir <folder>]
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from config import PROCESSED_DIR, S

df = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv",
                 parse_dates=["t"], low_memory=False).sort_values("t").reset_index(drop=True)

# ---------- features (past-only) ----------
F = pd.DataFrame(index=df.index)
g = df.glucose
for k in range(12):                                   # lags t .. t-55
    F[f"g_lag{k}"] = g.shift(k)
F["slope_15"] = (g - g.shift(3)) / 3                  # per 5-min step
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
F["min_since_carbs"] = (df.carbs_g > 0)[::-1].cumsum()[::-1]  # placeholder, replaced below
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

# ---------- splits ----------
train_m = (df.split == "train") & df.clean
test_m = (df.split == "test") & df.clean
VAL_START = config.ts("val_start")                   # validation window start (settings)
fit_m = train_m & (df.t < VAL_START)
val_m = train_m & (df.t >= VAL_START)

params = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=63, min_child_samples=40, subsample=0.9,
              colsample_bytree=0.8, n_estimators=3000, random_state=42, verbosity=-1)

m = lgb.LGBMRegressor(**params)
m.fit(F[fit_m], y[fit_m], eval_set=[(F[val_m], y[val_m])],
      callbacks=[lgb.early_stopping(100, verbose=False)])
best = m.best_iteration_
print("early stopping chose", best, "trees (val RMSE {:.3f})".format(m.best_score_["valid_0"]["rmse"]))

# refit on ALL clean train rows with the chosen tree count, then score test ONCE
mf = lgb.LGBMRegressor(**{**params, "n_estimators": best})
mf.fit(F[train_m], y[train_m])
pred = mf.predict(F[test_m])
yt = y[test_m].to_numpy()
ev = df[test_m]

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

carbs2h = df.carbs_g.rolling(24, min_periods=1).sum()[test_m].to_numpy()
overnight = ((ev.t.dt.hour < 6)).to_numpy() & (carbs2h == 0)
post_meal = carbs2h > 0
post_ex = (ev.hrs_since_exercise <= 4).to_numpy()
low = yt < 4
base = ev.glucose.to_numpy()

print(f"\nTEST (single evaluation, n={len(ev)}):")
for name, p in [("last-value", base), ("LightGBM", pred)]:
    print(f"  {name:11s} RMSE {rmse(yt,p):.3f} | overnight {rmse(yt[overnight],p[overnight]):.3f}"
          f" | post-meal {rmse(yt[post_meal],p[post_meal]):.3f}"
          f" | post-ex {rmse(yt[post_ex],p[post_ex]):.3f} (n={post_ex.sum()})"
          f" | low<4 {rmse(yt[low],p[low]):.3f} (n={low.sum()})")

r = yt - pred
print(f"\nLightGBM residuals: p5 {np.percentile(r,5):.2f} p50 {np.percentile(r,50):.2f} "
      f"p95 {np.percentile(r,95):.2f} | misses>2: {np.mean(np.abs(r)>2)*100:.1f}%")

imp = pd.Series(mf.feature_importances_, index=F.columns).sort_values(ascending=False)
print("\ntop 12 features:")
print(imp.head(12).to_string())

out = ev[["t"]].copy(); out["actual"] = yt; out["pred_lgbm"] = np.round(pred, 2)
out["pred_lastvalue"] = base
out.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
