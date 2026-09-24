"""
cv_ablation_exercise.py — the exercise question answered with enough data.

Rolling-origin cross-validation: five 14-day evaluation windows spanning both
seasons (so football, cricket and gym recency are all represented). For each
window, every variant trains ONLY on data before the window (200 trees flat,
identical for all variants — comparison, not tuning). Predictions are pooled
across windows, then RMSE is broken down by hours-since-last-hard-workout and
by the TYPE (aerobic vs anaerobic) of that workout.

CV compares model variants; it does not judge the final model (that already
happened, once, on the retired test set).

Variants: A glucose+time (Dexcom-equivalent) | B +insulin/carbs |
          C +exercise (current) | D +aerobic/anaerobic split.
Usage: python cv_ablation_exercise.py [--data-dir <folder>]
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

# ---------- feature blocks (as ablation_exercise.py) ----------
g = df.glucose
A = pd.DataFrame(index=df.index)
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

Bx = pd.DataFrame(index=df.index)
Bx["bolus_iob"] = df.bolus_iob_u
Bx["basal_iob"] = df.est_basal_iob_u
Bx["bolus_30m"] = df.bolus_u.rolling(6, min_periods=1).sum()
Bx["bolus_2h"] = df.bolus_u.rolling(24, min_periods=1).sum()
Bx["carbs_30m"] = df.carbs_g.rolling(6, min_periods=1).sum()
Bx["carbs_2h"] = df.carbs_g.rolling(24, min_periods=1).sum()
Bx["carbs_4h"] = df.carbs_g.rolling(48, min_periods=1).sum()
Bx["min_since_carbs"] = (df.t - df.t.where(df.carbs_g > 0).ffill()).dt.total_seconds() / 60
Bx["min_since_bolus"] = (df.t - df.t.where(df.bolus_u > 0).ffill()).dt.total_seconds() / 60
Bx["frac_suspend"] = df.frac_suspend
Bx["frac_max"] = df.frac_max
Bx["suspend_1h"] = df.frac_suspend.rolling(12, min_periods=1).mean()
Bx["frac_activity"] = df.frac_activity
Bx["activity_2h"] = df.frac_activity.rolling(24, min_periods=1).mean()
Bx["frac_limited"] = df.frac_limited

Cx = pd.DataFrame(index=df.index)
Cx["workout_now"] = df.workout_min
Cx["workout_1h"] = df.workout_min.rolling(12, min_periods=1).sum()
Cx["workout_24h"] = df.workout_min.rolling(288, min_periods=1).sum()
Cx["hrs_since_ex"] = df.hrs_since_exercise.clip(upper=72)
Cx["hr"] = df.hr_mean
Cx["hr_missing"] = df.hr_missing.astype(int)
Cx["steps_1h"] = df.steps.rolling(12, min_periods=1).sum()

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
Dx = pd.DataFrame(index=df.index)
Dx["aer_min_now"] = aer
Dx["aer_24h"] = pd.Series(aer).rolling(288, min_periods=1).sum().values
Dx["aer_48h"] = pd.Series(aer).rolling(576, min_periods=1).sum().values
Dx["anae_24h"] = pd.Series(anae).rolling(288, min_periods=1).sum().values
Dx["anae_48h"] = pd.Series(anae).rolling(576, min_periods=1).sum().values
for typ, sub in [("aer", wk[wk.cat.isin(AEROBIC)]), ("anae", wk[~wk.cat.isin(AEROBIC)])]:
    ends = np.sort(sub.e.values.astype("datetime64[s]").astype("int64"))
    hrs = np.full(len(df), 999.0)
    j = np.searchsorted(ends, gsec, "right") - 1
    ok = j >= 0
    hrs[ok] = (gsec[ok] - ends[j[ok]]) / 3600
    Dx[f"hrs_since_{typ}"] = np.clip(hrs, 0, 72)

variants = {
    "A glucose+time": A,
    "B +insulin/carbs": pd.concat([A, Bx], axis=1),
    "C +exercise": pd.concat([A, Bx, Cx], axis=1),
    "D +type split": pd.concat([A, Bx, Cx, Dx], axis=1),
}

y = df.glucose_t30
usable = (df.split != "test") & df.clean          # never touch test rows
FOLDS = [(a, b) for a, b in S["cv_ablation_folds"]]          # evaluation blocks from settings

common = dict(objective="regression", n_estimators=200, learning_rate=0.05,
              num_leaves=63, min_child_samples=40, subsample=0.9,
              colsample_bytree=0.8, random_state=42, verbosity=-1)

pool = {name: [] for name in variants}
pool_y, pool_i = [], []
for a, b in FOLDS:
    tr = usable & (df.t < a)
    te = usable & (df.t >= a) & (df.t < b)
    for name, X in variants.items():
        m = lgb.LGBMRegressor(**common)
        m.fit(X[tr], y[tr])
        pool[name].append(m.predict(X[te]))
    pool_y.append(y[te].to_numpy())
    pool_i.append(df.index[te].to_numpy())
    print(f"fold {a}: train n={int(tr.sum())}, eval n={int(te.sum())}")

ye = np.concatenate(pool_y)
ei = np.concatenate(pool_i)
preds = {k: np.concatenate(v) for k, v in pool.items()}

# recency + type of last HARD workout at each pooled eval point
hard = wk[wk.cat.isin(["cricket", "football", "gym"])].sort_values("e")
h_ends = hard.e.values.astype("datetime64[s]").astype("int64")
h_aer = np.isin(hard.cat.values, ["cricket", "football"])
esec = df.t[ei].values.astype("datetime64[s]").astype("int64")
j = np.searchsorted(h_ends, esec, "right") - 1
hrs = np.where(j >= 0, (esec - h_ends[np.clip(j, 0, None)]) / 3600, 999.0)
was_aer = np.where(j >= 0, h_aer[np.clip(j, 0, None)], False)

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

buckets = [("0-4h", hrs <= 4), ("4-12h", (hrs > 4) & (hrs <= 12)),
           ("12-24h", (hrs > 12) & (hrs <= 24)), ("24-48h", (hrs > 24) & (hrs <= 48)),
           (">48h", (hrs > 48) & (hrs < 900))]
tsplit = [("aerobic<12h", was_aer & (hrs <= 12)), ("anaerobic<12h", ~was_aer & (hrs <= 12) & (hrs < 900)),
          ("aerobic 12-48h", was_aer & (hrs > 12) & (hrs <= 48)),
          ("anaerobic 12-48h", ~was_aer & (hrs > 12) & (hrs <= 48) & (hrs < 900))]

print(f"\nPOOLED CV RESULTS (n={len(ye)}); RMSE mmol/L")
hdr = "  ".join(f"{b[0]:>7}" for b in buckets)
print(f"{'variant':18s} {'overall':>7}  {hdr}")
for name, p in preds.items():
    cells = "  ".join(f"{rmse(ye[b[1]], p[b[1]]):7.3f}" for b in buckets)
    print(f"{name:18s} {rmse(ye, p):7.3f}  {cells}")
print("bucket n:", ", ".join(f"{b[0]}={int(b[1].sum())}" for b in buckets))

print("\nby type of last hard workout:")
print(f"{'variant':18s} " + "  ".join(f"{t[0]:>16}" for t in tsplit))
for name, p in preds.items():
    cells = "  ".join(f"{rmse(ye[t[1]], p[t[1]]):16.3f}" for t in tsplit)
    print(f"{name:18s} {cells}")
print("type n:", ", ".join(f"{t[0]}={int(t[1].sum())}" for t in tsplit))

# glucose-drop reality check for the Module 1 narrative: how much does glucose
# actually move in the 4h after aerobic vs anaerobic sessions (raw data, no model)
print("\nraw physiology check — median glucose change from workout end to +2h / +4h:")
for lab, sub in [("aerobic (football/cricket)", wk[wk.cat.isin(["football", "cricket"])]),
                 ("anaerobic (gym)", wk[wk.cat == "gym"])]:
    d2, d4 = [], []
    for _, r in sub.iterrows():
        try:
            g_end = df.glucose[df.t.searchsorted(r.e.floor("5min"))]
            g_2h = df.glucose[df.t.searchsorted((r.e + pd.Timedelta("2h")).floor("5min"))]
            g_4h = df.glucose[df.t.searchsorted((r.e + pd.Timedelta("4h")).floor("5min"))]
            if pd.notna(g_end):
                if pd.notna(g_2h): d2.append(g_2h - g_end)
                if pd.notna(g_4h): d4.append(g_4h - g_end)
        except Exception:
            pass
    print(f"  {lab:28s} +2h: {np.median(d2):+.2f} (n={len(d2)}) | +4h: {np.median(d4):+.2f} (n={len(d4)})")
