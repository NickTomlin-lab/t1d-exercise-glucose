"""
evaluate_baselines.py — score the two mandated baselines on the FROZEN test set.

Baselines (data dictionary / project brief):
  1. last-value:   glucose(t+30) = glucose(t)
  2. linear trend: straight line fit to the last hour (12 readings incl. t),
                   extrapolated 30 minutes.

Evaluation: clean test intervals only (split == "test", clean == True).
Lag values for the trend model may reach back before the test boundary —
using PAST data for features is legal; the target never crosses backwards.

Reports RMSE / MAE overall and by regime (overnight fasting, post-meal,
post-exercise, low glucose), plus the residual tails. mmol/L throughout.

Usage: python evaluate_baselines.py [--data-dir <folder>]
"""
import os, sys
import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR, S

df = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv",
                 parse_dates=["t"], low_memory=False).sort_values("t").reset_index(drop=True)

# lag matrix for the trend fit (t, t-5, ..., t-55)
lags = pd.concat({k: df.glucose.shift(k) for k in range(12)}, axis=1)

ev = df[(df.split == "test") & df.clean].copy()
L = lags.loc[ev.index].to_numpy()            # column k = glucose at t - 5k min

y = ev.glucose_t30.to_numpy()
pred_last = ev.glucose.to_numpy()

# least-squares line through the 12 points; extrapolate +30 min (x in 5-min steps)
x = -np.arange(12)                           # 0, -1, ..., -11 (5-min units)
xm, x2 = x.mean(), (x**2).mean()
Lm = L.mean(axis=1)
slope = (L @ x / 12 - Lm * xm) / (x2 - xm**2)
pred_trend = Lm + slope * (6 - xm)           # value at x = +6 (30 min ahead)

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))

# regimes
t = ev.t.dt
carbs_past2h = df.carbs_g.rolling(24, min_periods=1).sum().loc[ev.index].to_numpy()
overnight = ((t.hour >= 0) & (t.hour < 6)).to_numpy() & (carbs_past2h == 0)
post_meal = carbs_past2h > 0
post_ex = (ev.hrs_since_exercise <= 4).to_numpy()
low = y < 4

print(f"clean test points: {len(ev)}  ({ev.t.min()} -> {ev.t.max()})")
rows = []
for name, pred in [("last-value", pred_last), ("linear trend (1h)", pred_trend)]:
    rows.append({"model": name, "RMSE": rmse(y, pred), "MAE": mae(y, pred),
                 "RMSE overnight": rmse(y[overnight], pred[overnight]),
                 "RMSE post-meal(2h)": rmse(y[post_meal], pred[post_meal]),
                 "RMSE post-exercise(4h)": rmse(y[post_ex], pred[post_ex]),
                 "RMSE low(<4)": rmse(y[low], pred[low]),
                 "n low": int(low.sum())})
res = pd.DataFrame(rows).set_index("model").round(3)
print(res.to_string())

for name, pred in [("last-value", pred_last), ("linear trend (1h)", pred_trend)]:
    r = y - pred
    print(f"\n{name}: residuals p5={np.percentile(r,5):.2f} p50={np.percentile(r,50):.2f} "
          f"p95={np.percentile(r,95):.2f} | % misses > 2 mmol/L: {np.mean(np.abs(r)>2)*100:.1f}%")

res.to_csv(PROCESSED_DIR / "baseline_results.csv")
