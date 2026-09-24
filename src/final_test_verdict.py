"""
final_test_verdict.py — THE single test evaluation of the chosen package.
After this run the test set retires: no further model choices may cite it.

Package (all choices fixed on validation beforehand):
  1. Point model: LightGBM, 135 trees (chosen earlier), trained on all clean train.
  2. Band: q10/q90 quantile models trained on pre-validation data only
     (so the calibration below is honest), patched conformally with risk
     groups defined by the hypo classifier (p >= 0.05); offsets measured on
     the FULL validation window, never on test.
  3. Hypo alert: classifier (73 trees) + isotonic calibration on validation,
     operating threshold 0.60 (validated edge: fewest alerts, no missed lows).

Declared meaning of the band: 80% interval overall; known to under-cover
during lows (~50% on validation) — the alert, not the band, covers lows.

Usage: python final_test_verdict.py [--data-dir <folder>]
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
te_i = df.index[test_m]
yt = y[te_i].to_numpy()
ev = df.loc[te_i]

common = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40,
              subsample=0.9, colsample_bytree=0.8, random_state=42, verbosity=-1)

# 1. point model
pm = lgb.LGBMRegressor(objective="regression", n_estimators=135, **common)
pm.fit(F[train_m], y[train_m])
pred = pm.predict(F.loc[te_i])

# 3. classifier (needed for band risk groups too)
ylow_all = (pd.concat({k: g.shift(-k) for k in range(1, 7)}, axis=1).min(axis=1) < 4).astype(int)
contig = (df.t.shift(-6) - df.t) == pd.Timedelta("30min")
clf = lgb.LGBMClassifier(objective="binary", n_estimators=73, **common)
clf.fit(F[fit_m & contig], ylow_all[fit_m & contig])
p_val_raw = clf.predict_proba(F[val_m])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_val_raw, ylow_all[val_m])
p_te_raw = clf.predict_proba(F.loc[te_i])[:, 1]
p_te = iso.predict(p_te_raw)

# 2. band: quantiles trained pre-validation, conformal offsets from validation
qlo = lgb.LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=269, **common)
qhi = lgb.LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=78, **common)
qlo.fit(F[fit_m], y[fit_m]); qhi.fit(F[fit_m], y[fit_m])
lo_v, hi_v = qlo.predict(F[val_m]), qhi.predict(F[val_m])
yv = y[val_m].to_numpy()
risky_v = p_val_raw >= 0.05
adj = {}
for grp, m_ in [("risky", risky_v), ("calm", ~risky_v)]:
    adj[grp] = max(np.percentile(np.minimum(lo_v, hi_v)[m_] - yv[m_], 90), 0)
    adj[grp + "_hi"] = max(np.percentile(yv[m_] - np.maximum(lo_v, hi_v)[m_], 90), 0)
lo_t = np.minimum(qlo.predict(F.loc[te_i]), qhi.predict(F.loc[te_i]))
hi_t = np.maximum(qlo.predict(F.loc[te_i]), qhi.predict(F.loc[te_i]))
risky_t = p_te_raw >= 0.05
lo_t = lo_t - np.where(risky_t, adj["risky"], adj["calm"])
hi_t = hi_t + np.where(risky_t, adj["risky_hi"], adj["calm_hi"])

# ---------------- the verdict ----------------
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
low = yt < 4
overnight = (ev.t.dt.hour < 6).to_numpy()
post_ex = (ev.hrs_since_exercise <= 4).to_numpy()

print(f"FINAL TEST VERDICT (n={len(ev)}; test set retires after this)\n")
print(f"1. point model RMSE {rmse(yt, pred):.3f}  (compare with the last-value bar from evaluate_baselines.py)")

inside = (yt >= lo_t) & (yt <= hi_t)
print(f"\n2. band (declared 80%): width {np.mean(hi_t - lo_t):.2f} mmol/L")
for lab, m_ in [("overall", np.ones_like(low, bool)), ("below 4", low),
                ("overnight", overnight), ("post-exercise", post_ex)]:
    print(f"   coverage {lab:14s} {inside[m_].mean()*100:5.1f}%  (n={int(m_.sum())})")

ytest_low = ylow_all[te_i].to_numpy()
brier = np.mean((p_te - ytest_low) ** 2)
print(f"\n3. hypo alert at threshold 0.60: Brier {brier:.4f} "
      f"(constant-prevalence {np.mean((ytest_low.mean()-ytest_low)**2):.4f})")

gl, tt = ev.glucose.to_numpy(), ev.t.to_numpy()
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
warn = p_te >= 0.60
warn_eps = episodes(warn, tt)
n_days = (tt[-1] - tt[0]) / np.timedelta64(1, "D") + 1
caught = sum(1 for s, e in low_eps
             if warn[(tt >= s - np.timedelta64(30, "m")) & (tt <= e)].any())
print(f"   low episodes in test: {len(low_eps)} | caught: {caught} | "
      f"alert episodes/day: {len(warn_eps)/n_days:.1f}")

out = ev[["t"]].copy()
out["actual"] = yt; out["pred"] = np.round(pred, 2)
out["band_lo"] = np.round(lo_t, 2); out["band_hi"] = np.round(hi_t, 2)
out["p_low30"] = np.round(p_te, 3); out["alert"] = warn
out.to_csv(PROCESSED_DIR / "final_test_outputs.csv", index=False)
