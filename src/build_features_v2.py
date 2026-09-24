"""
build_features_v2.py — curved insulin-on-board, carbs-on-board and two
exercise interactions, written to PROCESSED_DIR/aligned_5min_v2.csv.

aligned_5min.csv is NOT modified. The v2 file carries every existing column
byte-for-byte (the original text lines are copied and the new columns are
appended to each line), plus:

  iob5_bolus_u, iob5_basal_u, iob5_total_u, insulin_activity_u_per_5min
      exponential action curve (Loop / oref "exponential" model), peak 75 min,
      duration 5 h, applied to the interval-level bolus_u and est_basal_u series
  iob3_bolus_u, iob3_total_u
      same curve with a 3 h duration (peak 75 min)
  cob_g, carbs_absorbed_g_per_5min
      linear absorption of each carbs_g entry over 3 h
  iob5_total_x_exercising, iob5_total_x_post_ex_4h
      iob5_total_u times (workout_min > 0), times (hrs_since_exercise <= 4)

Everything is computed per window, so nothing bleeds across a window
fence or a gap between windows. Analysis only: nothing here informs a dosing decision.
Usage: python build_features_v2.py [--data-dir <folder>]
"""
import os, sys
import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR

SRC = PROCESSED_DIR / "aligned_5min.csv"
DST = PROCESSED_DIR / "aligned_5min_v2.csv"
STEP = 5.0                                   # minutes per grid interval


# ---------------------------------------------------------------- the curve
def exp_curve(tp, td):
    """Closed-form exponential insulin action curve (peak tp, duration td, minutes).
    Returns (iob, activity) callables valid on [0, td]; both are 0 after td."""
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    S = 1 / (1 - a + (1 + a) * np.exp(-td / tau))

    def activity(t):                         # fraction of D acting per minute
        t = np.asarray(t, dtype=float)
        v = (S / tau ** 2) * t * (1 - t / td) * np.exp(-t / tau)
        return np.where((t >= 0) & (t <= td), v, 0.0)

    def iob(t):                              # fraction of D still to act
        t = np.asarray(t, dtype=float)
        v = 1 - S * (1 - a) * ((t ** 2 / (tau * td * (1 - a)) - t / tau - 1) * np.exp(-t / tau) + 1)
        return np.where(t < 0, 0.0, np.where(t > td, 0.0, v))

    return iob, activity, dict(tau=tau, a=a, S=S)


def check_curve(name, tp, td):
    iob, act, c = exp_curve(tp, td)
    tt = np.arange(0, td + 1e-9, 0.01)
    integral = np.trapezoid(act(tt), tt)
    tpeak = tt[np.argmax(act(tt))]
    # consistency of the two closed forms: -dIOB/dt should equal activity
    mid = np.arange(1, td - 1, 1.0)
    deriv = -(iob(mid + 0.5) - iob(mid - 0.5)) / 1.0
    max_dev = float(np.max(np.abs(deriv - act(mid))))
    print(f"  {name}: tau={c['tau']:.3f} a={c['a']:.4f} S={c['S']:.4f}")
    print(f"    IOB(0) = {float(iob(0)):.6f}  (want 1)")
    print(f"    IOB(td) = {float(iob(td)):.2e}  (want 0)")
    print(f"    integral of activity over [0, td] = {integral:.6f}  (want 1)")
    print(f"    activity peaks at t = {tpeak:.1f} min (want {tp}); "
          f"max |dIOB/dt + activity| = {max_dev:.2e}")
    print(f"    IOB at 60/120/180/240 min: "
          + " / ".join(f"{float(iob(m)):.3f}" for m in (60, 120, 180, 240)))
    ok = (abs(float(iob(0)) - 1) < 1e-9 and abs(float(iob(td))) < 1e-9
          and abs(integral - 1) < 1e-4 and max_dev < 1e-6)
    print(f"    CHECK {'PASSED' if ok else 'FAILED'}")
    return iob, act


print("Curve checks (unit dose):")
iob5, act5 = check_curve("5 h curve (tp 75, td 300)", 75, 300)
iob3, act3 = check_curve("3 h curve (tp 75, td 180)", 75, 180)

# ------------------------------------------------------------ discrete kernels
# Boluses: dose sits at the interval that holds its timestamp; age at row k is
# (k - j) * 5 min, so the same-interval row carries the full dose (as the
# pump-style bolus_iob_u does for a bolus logged at the interval start).
# Basal: est_basal_u is a delivery spread across interval j; it is seen from the
# next row onward at mid-interval age ((k - j - 1) * 5 + 2.5 min), exactly the
# convention of est_basal_iob_u in build_dataset.py.
def kernels(iob, act, td):
    K = int(td / STEP)
    m = np.arange(K + 1)
    age_bolus = m * STEP
    age_basal = np.where(m >= 1, (m - 1) * STEP + STEP / 2, -1.0)
    return dict(iob_bolus=iob(age_bolus), act_bolus=act(age_bolus) * STEP,
                iob_basal=iob(age_basal), act_basal=act(age_basal) * STEP)


k5 = kernels(iob5, act5, 300)
k3 = kernels(iob3, act3, 180)
mc = np.arange(37)
k_cob = np.clip(1 - mc * STEP / 180, 0, None)                # COB fraction at age m*5
k_abs = np.where(mc * STEP < 180, STEP / 180, 0.0)           # grams absorbed per 5 min

# ----------------------------------------------------------------- compute
df = pd.read_csv(SRC, parse_dates=["t"], low_memory=False)
assert df.t.is_monotonic_increasing, "aligned_5min.csv must be time-ordered"
n = len(df)
new = {c: np.zeros(n) for c in
       ["iob5_bolus_u", "iob5_basal_u", "iob5_total_u", "insulin_activity_u_per_5min",
        "iob3_bolus_u", "iob3_total_u", "cob_g", "carbs_absorbed_g_per_5min"]}


def conv(x, kern):
    return np.convolve(x, kern)[: len(x)]


for wnum in sorted(df.window.unique()):
    m = (df.window == wnum).to_numpy()
    sub = df.loc[m]
    gaps = sub.t.diff().dropna().unique()
    assert len(gaps) == 0 or (gaps == np.timedelta64(5, "m")).all(), f"window {wnum} grid not regular"
    bol = sub.bolus_u.fillna(0).to_numpy()
    bas = sub.est_basal_u.fillna(0).to_numpy()
    carb = sub.carbs_g.fillna(0).to_numpy()
    b5, s5 = conv(bol, k5["iob_bolus"]), conv(bas, k5["iob_basal"])
    new["iob5_bolus_u"][m] = b5
    new["iob5_basal_u"][m] = s5
    new["iob5_total_u"][m] = b5 + s5
    new["insulin_activity_u_per_5min"][m] = conv(bol, k5["act_bolus"]) + conv(bas, k5["act_basal"])
    b3, s3 = conv(bol, k3["iob_bolus"]), conv(bas, k3["iob_basal"])
    new["iob3_bolus_u"][m] = b3
    new["iob3_total_u"][m] = b3 + s3
    new["cob_g"][m] = conv(carb, k_cob)
    new["carbs_absorbed_g_per_5min"][m] = conv(carb, k_abs)

exercising = (df.workout_min.fillna(0) > 0).to_numpy().astype(float)
post4 = (df.hrs_since_exercise <= 4).fillna(False).to_numpy().astype(float)
new["iob5_total_x_exercising"] = new["iob5_total_u"] * exercising
new["iob5_total_x_post_ex_4h"] = new["iob5_total_u"] * post4

NEW_COLS = list(new.keys())
for c in NEW_COLS:
    new[c] = np.round(new[c], 4)

# ------------------------------------------------------------------- write
# Copy the original lines verbatim and append the new columns, so every
# existing column is byte-identical in the v2 file.
with open(SRC, "r", encoding="utf-8", newline="") as fh:
    lines = fh.read().split("\n")
if lines and lines[-1] == "":
    lines = lines[:-1]
assert len(lines) == n + 1, f"line count {len(lines)} vs rows {n}+1"
fmt = {c: [f"{v:g}" if v != 0 else "0.0" for v in new[c]] for c in NEW_COLS}
out_lines = [lines[0] + "," + ",".join(NEW_COLS)]
for i in range(n):
    out_lines.append(lines[i + 1] + "," + ",".join(fmt[c][i] for c in NEW_COLS))
with open(DST, "w", encoding="utf-8", newline="") as fh:
    fh.write("\n".join(out_lines) + "\n")

# verify: original columns identical, new columns round-trip
chk = pd.read_csv(DST, parse_dates=["t"], low_memory=False)
old = pd.read_csv(SRC, low_memory=False)
chk_old = pd.read_csv(DST, low_memory=False)[old.columns]
assert old.equals(chk_old), "existing columns changed in the v2 file"
for c in NEW_COLS:
    assert np.allclose(chk[c].to_numpy(), new[c], atol=1e-6), c
print(f"\nwrote {DST}: {len(chk)} rows, {len(old.columns)} original columns unchanged "
      f"(verified by re-read), {len(NEW_COLS)} new columns appended")

# --------------------------------------------------------------- report
tr = chk[chk.split == "train"]
print(f"\nTRAIN rows only (n={len(tr)}; test and candidate_test rows are never printed)")
cols = ["bolus_iob_u", "iob3_bolus_u", "iob5_bolus_u", "est_basal_iob_u", "iob5_basal_u",
        "iob3_total_u", "iob5_total_u", "insulin_activity_u_per_5min",
        "cob_g", "carbs_absorbed_g_per_5min", "iob5_total_x_exercising", "iob5_total_x_post_ex_4h"]
labels = {"bolus_iob_u": "pump 2h linear (existing)", "est_basal_iob_u": "pump 2h linear (existing)"}
print(f"  {'column':30s} {'mean':>8} {'max':>8} {'frac>0':>7}  note")
for c in cols:
    print(f"  {c:30s} {tr[c].mean():8.3f} {tr[c].max():8.3f} {(tr[c] > 0).mean():7.3f}  {labels.get(c, '')}")
print(f"  (2h total for reference: bolus_iob_u + est_basal_iob_u mean "
      f"{(tr.bolus_iob_u + tr.est_basal_iob_u).mean():.3f}, max {(tr.bolus_iob_u + tr.est_basal_iob_u).max():.3f})")

r_tot = np.corrcoef(tr.iob5_total_u, tr.bolus_iob_u)[0, 1]
r_bol = np.corrcoef(tr.iob5_bolus_u, tr.bolus_iob_u)[0, 1]
r_bas = np.corrcoef(tr.iob5_basal_u, tr.est_basal_iob_u)[0, 1]
r_3 = np.corrcoef(tr.iob3_bolus_u, tr.bolus_iob_u)[0, 1]
print(f"\n  corr(iob5_total_u, bolus_iob_u) = {r_tot:.4f}")
print(f"  corr(iob5_bolus_u, bolus_iob_u) = {r_bol:.4f}   corr(iob3_bolus_u, bolus_iob_u) = {r_3:.4f}")
print(f"  corr(iob5_basal_u, est_basal_iob_u) = {r_bas:.4f}")
print(f"  rows with iob5_bolus_u > 0 but bolus_iob_u == 0 (insulin 2-5 h old only): "
      f"{int(((tr.iob5_bolus_u > 0.05) & (tr.bolus_iob_u == 0)).sum())} "
      f"({((tr.iob5_bolus_u > 0.05) & (tr.bolus_iob_u == 0)).mean()*100:.1f}% of train rows)")
print(f"  interactions: exercising rows {int((tr.iob5_total_x_exercising > 0).sum())}, "
      f"post-exercise-4h rows {int((tr.iob5_total_x_post_ex_4h > 0).sum())}")
