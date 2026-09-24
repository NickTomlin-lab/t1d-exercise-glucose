"""
forecaster_verdict_from_outputs.py - the final test verdict, recomputed from
the SAVED test outputs (nothing re-fitted, nothing re-scored).

final_test_verdict.py scored the retired test set once and saved every
prediction to final_test_outputs.csv (t, actual, pred, band_lo, band_hi,
p_low30, alert). Its printed verdict was not kept as a file, so this script
recomputes the same summary from that file plus the glucose, hour and
hours-since-exercise columns of aligned_5min.csv for the same rows:
  1. point-model RMSE against the last-value baseline (glucose now = glucose
     in 30 min), overall and by regime;
  2. band width and coverage (declared 80%) overall, below 4 mmol/L,
     overnight and within 4 h after exercise;
  3. the low alert (p_low30 >= 0.60): Brier score against the constant
     prevalence, low episodes, episodes caught, alert episodes per day.
Episode logic as in final_test_verdict.py: a low episode is a run of
readings < 4 with gaps <= 15 min merged; caught if any alert fired in the
30 min before it or during it; an alert episode is a maximal run of alert rows.
"""
import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR

THR = 0.60


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


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


def main():
    path = PROCESSED_DIR / "final_test_outputs.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    F = pd.read_csv(path, parse_dates=["t"]).sort_values("t").reset_index(drop=True)
    A = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv", parse_dates=["t"], low_memory=False,
                    usecols=["t", "glucose", "hour", "hrs_since_exercise"])
    A = A.set_index("t")
    # glucose 30 min ahead of every scored row, for the "any low in the next 30 min" target
    fut = pd.concat({k: A["glucose"].shift(-k) for k in range(1, 7)}, axis=1).min(axis=1)
    F["glucose_now"] = A["glucose"].reindex(F["t"]).to_numpy()
    F["hour"] = A["hour"].reindex(F["t"]).to_numpy()
    F["hrs_since_ex"] = A["hrs_since_exercise"].reindex(F["t"]).to_numpy()
    F["ylow"] = (fut.reindex(F["t"]).to_numpy() < 4).astype(int)
    F["alert"] = F["alert"].astype(str).str.lower().isin(["true", "1", "1.0"])

    yt = F["actual"].to_numpy(dtype=float)
    pred = F["pred"].to_numpy(dtype=float)
    base = F["glucose_now"].to_numpy(dtype=float)
    low = yt < 4
    overnight = F["hour"].to_numpy() < 6
    post_ex = F["hrs_since_ex"].to_numpy() <= 4
    n_days = F["t"].dt.date.nunique()

    print(f"FINAL TEST VERDICT, recomputed from the saved test outputs (n = {len(F)} clean test rows, "
          f"{n_days} days; nothing re-fitted)\n")
    print("1. 30-minute point forecast, RMSE in mmol/L (last-value baseline = glucose now)")
    for name, p in (("last-value", base), ("LightGBM", pred)):
        print(f"   {name:11s} overall {rmse(yt, p):.3f} | overnight {rmse(yt[overnight], p[overnight]):.3f} "
              f"(n={int(overnight.sum())}) | post-exercise 4h {rmse(yt[post_ex], p[post_ex]):.3f} (n={int(post_ex.sum())}) "
              f"| below 4 {rmse(yt[low], p[low]):.3f} (n={int(low.sum())})")
    r = yt - pred
    print(f"   LightGBM residuals: p5 {np.percentile(r, 5):.2f}, p50 {np.percentile(r, 50):.2f}, "
          f"p95 {np.percentile(r, 95):.2f}; misses > 2 mmol/L: {np.mean(np.abs(r) > 2) * 100:.1f}%")

    lo, hi = F["band_lo"].to_numpy(dtype=float), F["band_hi"].to_numpy(dtype=float)
    inside = (yt >= lo) & (yt <= hi)
    print(f"\n2. band (declared 80% interval): mean width {np.mean(hi - lo):.2f} mmol/L")
    for lab, m in (("overall", np.ones_like(low, bool)), ("below 4", low), ("overnight", overnight),
                   ("post-exercise", post_ex)):
        print(f"   coverage {lab:14s} {inside[m].mean() * 100:5.1f}%  (n={int(m.sum())})")

    p = F["p_low30"].to_numpy(dtype=float)
    y = F["ylow"].to_numpy()
    brier = np.mean((p - y) ** 2)
    print(f"\n3. low alert (any reading < 4 in the next 30 min; threshold {THR:.2f}): Brier {brier:.4f} "
          f"(constant-prevalence {np.mean((y.mean() - y) ** 2):.4f}; prevalence {y.mean() * 100:.1f}%)")
    gl, tt = F["glucose_now"].to_numpy(dtype=float), F["t"].to_numpy()
    low_eps = episodes(gl < 4, tt)
    warn = F["alert"].to_numpy()
    warn_eps = episodes(warn, tt)
    caught = sum(1 for s, e in low_eps if warn[(tt >= s - np.timedelta64(30, "m")) & (tt <= e)].any())
    print(f"   low episodes in test: {len(low_eps)} | caught: {caught} | alert episodes per day: "
          f"{len(warn_eps) / n_days:.1f} | alert rows: {int(warn.sum())} ({warn.mean() * 100:.1f}% of rows)")
    print(f"   alert precision (share of alert rows followed by a low within 30 min): "
          f"{y[warn].mean() * 100:.1f}%" if warn.sum() else "")


if __name__ == "__main__":
    main()
