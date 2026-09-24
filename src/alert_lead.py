"""
alert_lead.py - how far ahead of each low did the 30-minute low alert fire?

Existing outputs only: reads PROCESSED_DIR/final_test_outputs.csv (the
retired test set's saved predictions and alert flags, written once by
final_test_verdict.py) and the glucose column of aligned_5min.csv for the
same rows. Nothing is re-scored or re-fitted.

Definitions
  low episode   the first row with glucose < 4.0 mmol/L after at least 15 min
                (three consecutive scored rows) at or above 4.0
  alert lead    time of that first low minus the time of the first row in the
                preceding 60 min (inclusive of the low row) whose alert flag is
                on: 0 if the alert came on the same row, "missed" if none
Reports the median and quartiles of the lead, the share of episodes with
lead >= 20 min and >= 30 min, and the number missed. Writes
PROCESSED_DIR/alert_lead_episodes.csv (one lead value per episode, shuffled,
no timestamps) for the figure.

Context: the CGM's own predictive alert fires when it predicts a reading
below 3.1 mmol/L within 20 minutes, and its low alert fires at the user's
threshold; this analysis uses a 4.0 mmol/L threshold and the model's
30-minute horizon, so it is not a like-for-like comparison.
"""
import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR

LOW = 4.0
LOOKBACK = pd.Timedelta(minutes=60)
QUIET_ROWS = 3          # 15 min at or above 4.0 before the first low
SEED = 20260923


def episodes_and_leads(F):
    """F: rows with t, glucose, alert (sorted by t). Returns a list of leads
    (minutes) with NaN for missed episodes."""
    F = F.sort_values("t").reset_index(drop=True)
    t = F["t"].to_numpy()
    g = F["glucose"].to_numpy(dtype=float)
    a = F["alert"].astype(bool).to_numpy()
    leads = []
    for i in range(QUIET_ROWS, len(F)):
        if not (g[i] < LOW):
            continue
        prev = slice(i - QUIET_ROWS, i)
        ok = np.all(g[prev] >= LOW) and np.all(np.diff(t[i - QUIET_ROWS:i + 1]) == np.timedelta64(5, "m"))
        if not ok:
            continue
        win = (t >= t[i] - LOOKBACK.to_timedelta64()) & (t <= t[i])
        fired = np.flatnonzero(win & a)
        leads.append((t[i] - t[fired[0]]) / np.timedelta64(1, "m") if len(fired) else np.nan)
    return leads


def main():
    path = PROCESSED_DIR / "final_test_outputs.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run final_test_verdict.py first (or use the real outputs)")
    F = pd.read_csv(path, parse_dates=["t"])
    A = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv", parse_dates=["t"], low_memory=False, usecols=["t", "glucose"])
    F = F.merge(A, on="t", how="left")
    if "alert" not in F.columns:
        raise SystemExit("final_test_outputs.csv has no alert column")
    F["alert"] = F["alert"].astype(str).str.lower().isin(["true", "1", "1.0"])
    leads = np.array(episodes_and_leads(F), dtype=float)
    n = len(leads)
    got = leads[~np.isnan(leads)]
    print("Alert lead versus the first low, retired test set (existing outputs only; nothing re-scored)")
    print(f"scored rows: {len(F)}; alert rows: {int(F['alert'].sum())} "
          f"({F['alert'].mean() * 100:.1f}% of rows, {F['alert'].sum() / F['t'].dt.date.nunique():.1f} per day)")
    print(f"low episodes (first reading < {LOW} after >= 15 min at or above {LOW}): {n}")
    if n:
        print(f"  alert fired in the preceding 60 min: {len(got)} of {n}; missed: {n - len(got)}")
    if len(got):
        q1, med, q3 = np.percentile(got, [25, 50, 75])
        print(f"  lead (min), episodes with an alert: median {med:.0f}, Q1 {q1:.0f}, Q3 {q3:.0f}, "
              f"min {got.min():.0f}, max {got.max():.0f}")
        print(f"  lead >= 20 min: {int((got >= 20).sum())} of {n} episodes ({(got >= 20).mean() * len(got) / n * 100:.0f}%)")
        print(f"  lead >= 30 min: {int((got >= 30).sum())} of {n} episodes ({(got >= 30).mean() * len(got) / n * 100:.0f}%)")
        print(f"  lead == 0 (alert on the same row as the low): {int((got == 0).sum())}")
        print("  (shares are of ALL episodes, missed ones counted as no lead)")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    out = pd.DataFrame({"lead_min": leads[order]})
    out["missed"] = out["lead_min"].isna().astype(int)
    out.to_csv(PROCESSED_DIR / "alert_lead_episodes.csv", index=False)
    print(f"wrote {PROCESSED_DIR / 'alert_lead_episodes.csv'} (one row per episode, shuffled, no timestamps)")
    print("\nContext: the CGM's own predictive alert fires when it predicts a reading below 3.1 mmol/L within 20 min,")
    print("and its low alert fires at the user's threshold; this comparison uses a 4.0 mmol/L threshold and the")
    print("model's 30-minute horizon, so it is not a like-for-like head-to-head.")


if __name__ == "__main__":
    main()
