"""
tune_band.py — fix the overconfident band, VALIDATION ONLY (test untouched).

Two candidate fixes, judged on the second half of validation (first half is
used only for calibration where needed):

(b) REWEIGHTING: retrain the q10/q90 quantile models with extra sample weight
    on low-target rows (glucose_t30 < 4.5), weight w in {1, 3, 10}.
    w=1 reproduces the original band.

(a) CONFORMAL-STYLE PATCH: keep the original band, but on the calibration
    half measure how far q10 must be lowered in "risky" contexts for the band
    to hold, then apply that offset. Risk groups use ONLY information known
    at prediction time: current glucose < 5.5, or 15-min slope < -0.05.

Scoreboard per candidate: coverage overall / below 4 / post-exercise
(target 80%) and mean width. Usage: python tune_band.py [--data-dir <folder>]
"""
import os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from config import PROCESSED_DIR, S

df = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv",
                 parse_dates=["t"], low_memory=False).sort_values("t").reset_index(drop=True)

# ---- features (same build) ----
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
VAL_START = config.ts("val_start")                   # validation window start (settings)
fit_m = train_m & (df.t < VAL_START)
val_m = train_m & (df.t >= VAL_START)

vi = df.index[val_m]
half = len(vi) // 2
cal_i, ev_i = vi[:half], vi[half:]          # calibration half / evaluation half
yc, ye = y[cal_i].to_numpy(), y[ev_i].to_numpy()

low_e = ye < 4
post_ex_e = (df.hrs_since_exercise[ev_i] <= 4).to_numpy()

def score(name, lo, hi):
    inside = (ye >= lo) & (ye <= hi)
    print(f"{name:34s} width {np.mean(hi-lo):4.2f} | overall {inside.mean()*100:5.1f}%"
          f" | below4 {inside[low_e].mean()*100:5.1f}% (n={low_e.sum()})"
          f" | post-ex {inside[post_ex_e].mean()*100:5.1f}%")

params = dict(objective="quantile", learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.9, colsample_bytree=0.8,
              n_estimators=400, random_state=42, verbosity=-1)

print(f"evaluation half: n={len(ev_i)}, lows {low_e.sum()} | target coverage 80%\n")

bands = {}
for w in (1, 3, 10):
    sw = np.where(y[fit_m] < 4.5, w, 1.0)
    lo_m = lgb.LGBMRegressor(**params, alpha=0.1)
    hi_m = lgb.LGBMRegressor(**params, alpha=0.9)
    lo_m.fit(F.loc[fit_m], y[fit_m], sample_weight=sw)
    hi_m.fit(F.loc[fit_m], y[fit_m], sample_weight=sw)
    lo_c, hi_c = lo_m.predict(F.loc[cal_i]), hi_m.predict(F.loc[cal_i])
    lo_e_, hi_e_ = lo_m.predict(F.loc[ev_i]), hi_m.predict(F.loc[ev_i])
    bands[w] = (np.minimum(lo_c, hi_c), np.maximum(lo_c, hi_c),
                np.minimum(lo_e_, hi_e_), np.maximum(lo_e_, hi_e_))
    score(f"(b) reweight w={w}", bands[w][2], bands[w][3])

# (a) conformal patch on the ORIGINAL band (w=1), risk groups from known-now features
lo_c, hi_c, lo_ev, hi_ev = bands[1]
risky_c = (df.glucose[cal_i] < 5.5).to_numpy() | ((F.slope_15[cal_i] < -0.05).to_numpy())
risky_e = (df.glucose[ev_i] < 5.5).to_numpy() | ((F.slope_15[ev_i] < -0.05).to_numpy())
adj = {}
for grp, m_ in [("risky", risky_c), ("calm", ~risky_c)]:
    under = lo_c[m_] - yc[m_]                     # positive where truth fell below floor
    adj[grp] = max(np.percentile(under, 90), 0)   # lower floor so ~10% remain below
    over = yc[m_] - hi_c[m_]
    adj[grp + "_hi"] = max(np.percentile(over, 90), 0)
print(f"\n(a) offsets from calibration half: floor down {adj['risky']:.2f} (risky) / "
      f"{adj['calm']:.2f} (calm); ceiling up {adj['risky_hi']:.2f} / {adj['calm_hi']:.2f}")
lo_a = lo_ev - np.where(risky_e, adj["risky"], adj["calm"])
hi_a = hi_ev + np.where(risky_e, adj["risky_hi"], adj["calm_hi"])
score("(a) conformal patch on w=1 band", lo_a, hi_a)

# (a') conformal patch with risk groups defined by the HYPO CLASSIFIER itself —
# calibrate hardest exactly where the model sees danger.
ylow_all = (pd.concat({k: g.shift(-k) for k in range(1, 7)}, axis=1).min(axis=1) < 4).astype(int)
clf = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=63,
                         min_child_samples=40, subsample=0.9, colsample_bytree=0.8,
                         n_estimators=73, random_state=42, verbosity=-1)
clf.fit(F.loc[fit_m], ylow_all[fit_m])
p_cal = clf.predict_proba(F.loc[cal_i])[:, 1]
p_ev = clf.predict_proba(F.loc[ev_i])[:, 1]
lo_c, hi_c, lo_ev, hi_ev = bands[1]
for cut in (0.05, 0.2):
    risky_c2, risky_e2 = p_cal >= cut, p_ev >= cut
    a = {}
    for grp, m_ in [("risky", risky_c2), ("calm", ~risky_c2)]:
        a[grp] = max(np.percentile(lo_c[m_] - yc[m_], 90), 0)
        a[grp + "_hi"] = max(np.percentile(yc[m_] - hi_c[m_], 90), 0)
    lo_a2 = lo_ev - np.where(risky_e2, a["risky"], a["calm"])
    hi_a2 = hi_ev + np.where(risky_e2, a["risky_hi"], a["calm_hi"])
    score(f"(a') classifier-risk patch cut={cut}", lo_a2, hi_a2)

# and the patch applied on top of the best reweighted band, for completeness
for w in (3, 10):
    lo_c, hi_c, lo_ev, hi_ev = bands[w]
    for grp, m_ in [("risky", risky_c), ("calm", ~risky_c)]:
        adj[grp] = max(np.percentile(lo_c[m_] - yc[m_], 90), 0)
        adj[grp + "_hi"] = max(np.percentile(yc[m_] - hi_c[m_], 90), 0)
    lo_a = lo_ev - np.where(risky_e, adj["risky"], adj["calm"])
    hi_a = hi_ev + np.where(risky_e, adj["risky_hi"], adj["calm_hi"])
    score(f"(a)+(b) patch on w={w} band", lo_a, hi_a)
