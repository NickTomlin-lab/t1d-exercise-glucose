"""
cv_classifier_v2.py — prompt 07, part 2: does the curved insulin-on-board (and
carbs-on-board) help the "low within 30 minutes?" classifier?

Target and learner exactly as train_hypo_classifier.py: any glucose < 4.0 in the
next 30 min; LightGBM binary with the same fixed hyperparameters and early
stopping; isotonic calibration. No tuning.

Feature sets
  F0  existing list from train_hypo_classifier.py (with the pump's 2 h IOB columns)
  F1  F0 with bolus_iob / basal_iob replaced by iob5_bolus_u, iob5_basal_u,
      insulin_activity_u_per_5min
  F2  F1 + cob_g
  F3  F2 + iob5_total_x_exercising, iob5_total_x_post_ex_4h
  F4  F0 with the 2 h IOB columns replaced by iob3_bolus_u, iob3_total_u

Evaluation: rolling-origin CV over the WHOLE train split: test blocks of
cv_block_days, each trained on everything earlier (>= cv_min_train_days). Inside
each training fold the last 20 % (time-ordered) is the validation fold: it
drives early stopping and fits the isotonic calibrator; the model is fitted on
the first 80 %. The test block is never seen before it is scored. Out-of-fold
predictions are pooled. Test rows and candidate_test rows are never loaded
into any fit or score.
Usage: python cv_classifier_v2.py [--data-dir <folder>]
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore", message=".*eval_set.*")   # lightgbm 4.7 renames it; eval_set kept to match train_hypo_classifier.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

import config
from config import PROCESSED_DIR, S

df = pd.read_csv(PROCESSED_DIR / "aligned_5min_v2.csv",
                 parse_dates=["t"], low_memory=False).sort_values("t").reset_index(drop=True)

# ---------- feature build: verbatim from train_hypo_classifier.py ----------
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
F0_COLS = list(F.columns)

# new columns (prompt 07)
for c in ["iob5_bolus_u", "iob5_basal_u", "insulin_activity_u_per_5min", "cob_g",
          "iob5_total_x_exercising", "iob5_total_x_post_ex_4h", "iob3_bolus_u", "iob3_total_u"]:
    F[c] = df[c]

def replace(cols, drop, add):
    out = [c for c in cols if c not in drop]
    i = cols.index(drop[0])
    return out[:i] + add + out[i:]

OLD_IOB = ["bolus_iob", "basal_iob"]
F1_COLS = replace(F0_COLS, OLD_IOB, ["iob5_bolus_u", "iob5_basal_u", "insulin_activity_u_per_5min"])
F2_COLS = F1_COLS + ["cob_g"]
F3_COLS = F2_COLS + ["iob5_total_x_exercising", "iob5_total_x_post_ex_4h"]
F4_COLS = replace(F0_COLS, OLD_IOB, ["iob3_bolus_u", "iob3_total_u"])
SETS = {"F0": F0_COLS, "F1": F1_COLS, "F2": F2_COLS, "F3": F3_COLS, "F4": F4_COLS}
DESC = {"F0": "existing list (pump 2 h linear IOB)",
        "F1": "F0, 2 h IOB -> iob5_bolus_u + iob5_basal_u + insulin_activity_u_per_5min",
        "F2": "F1 + cob_g",
        "F3": "F2 + iob5_total_x_exercising + iob5_total_x_post_ex_4h",
        "F4": "F0, 2 h IOB -> iob3_bolus_u + iob3_total_u"}

# ---------- target: any reading below 4 in the next 30 min (as there) ----------
fut_min = pd.concat({k: g.shift(-k) for k in range(1, 7)}, axis=1).min(axis=1)
contig = (df.t.shift(-6) - df.t) == pd.Timedelta("30min")
ylow = (fut_min < 4).astype(int)
usable = (df.split == "train") & df.clean & contig
assert not (usable & df.split.isin(["test", "candidate_test", "boundary_excluded"])).any()
assert df.t[usable].max() < config.ts("test_start"), "a test-period row leaked into the CV set"

# ---------- folds: 14-day blocks, each trained on everything earlier ----------
T0 = df.t[usable].min().normalize()
T_END = df.t[usable].max()
MIN_TRAIN_DAYS = int(S["cv_min_train_days"])
BLOCK_DAYS = int(S["cv_block_days"])
MIN_BLOCK_DAYS = int(S["cv_min_block_days"])
blocks = []
start = T0 + pd.Timedelta(days=MIN_TRAIN_DAYS)
while start <= T_END:
    end = start + pd.Timedelta(days=BLOCK_DAYS)
    blocks.append([start, end])
    start = end
# fold a short tail block into the block before it, and drop any block that
# holds no usable rows (a gap between windows)
merged = []
for s, e in blocks:
    n_here = int((usable & (df.t >= s) & (df.t < e)).sum())
    days_here = df.t[usable & (df.t >= s) & (df.t < e)].dt.normalize().nunique()
    if n_here == 0:
        continue
    if days_here < MIN_BLOCK_DAYS and merged and (merged[-1][1] == s):
        merged[-1][1] = e
    else:
        merged.append([s, e])
blocks = merged

params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.9, colsample_bytree=0.8,
              n_estimators=3000, random_state=42, verbosity=-1)

print(f"usable train rows: {int(usable.sum())} ({df.t[usable].min()} to {df.t[usable].max()}), "
      f"lows {ylow[usable].mean()*100:.1f}%")
print("\nfeature sets:")
for k, cols in SETS.items():
    print(f"  {k}: {len(cols)} features: {DESC[k]}")

print(f"\nfolds ({len(blocks)}): {BLOCK_DAYS}-day test blocks; train = all usable rows before the block, "
      f"last 20% of the train fold = validation (early stopping + isotonic)")
print(f"  {'block':>2} {'test start':>10} {'test end':>10} {'n_train':>8} {'n_val':>6} {'n_test':>6} {'test lows':>9}  best_iter by set")
pred = {k: np.full(len(df), np.nan) for k in SETS}
block_id = np.full(len(df), -1)
t_start = time.time()
for b, (s, e) in enumerate(blocks):
    te = usable & (df.t >= s) & (df.t < e)
    tr_all = usable & (df.t < s)
    tr_idx = df.index[tr_all].to_numpy()
    cut = int(len(tr_idx) * 0.8)
    fit_idx, val_idx = tr_idx[:cut], tr_idx[cut:]
    te_idx = df.index[te].to_numpy()
    block_id[te_idx] = b
    iters = []
    for k, cols in SETS.items():
        X = F[cols]
        m = lgb.LGBMClassifier(**params)
        m.fit(X.loc[fit_idx], ylow[fit_idx], eval_set=[(X.loc[val_idx], ylow[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration_)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(m.predict_proba(X.loc[val_idx])[:, 1], ylow[val_idx])
        pred[k][te_idx] = iso.predict(m.predict_proba(X.loc[te_idx])[:, 1])
    print(f"  {b:2d} {s.date()} {(e - pd.Timedelta(days=1)).date()} {len(fit_idx):8d} {len(val_idx):6d} "
          f"{len(te_idx):6d} {ylow[te_idx].mean()*100:8.1f}%  {' '.join(f'{k}={i}' for k, i in zip(SETS, iters))}")
print(f"  ({time.time() - t_start:.0f} s)")

# ---------- pooled out-of-fold evaluation ----------
scored = block_id >= 0
idx = df.index[scored].to_numpy()
y = ylow[idx].to_numpy()
P = {k: pred[k][idx] for k in SETS}
bid = block_id[idx]
tt = df.t[idx].to_numpy()
gl = df.glucose[idx].to_numpy()
n_days = df.t[idx].dt.normalize().nunique()
THR = 0.60


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


def episode_stats(p):
    warn = p >= THR
    warn_eps = episodes(warn, tt)
    caught = 0
    for s_, e_ in low_eps:
        pre = (tt >= s_ - np.timedelta64(30, "m")) & (tt < s_)
        if warn[pre].any() or warn[(tt >= s_) & (tt <= e_)].any():
            caught += 1
    prec = y[warn].mean() if warn.sum() else float("nan")
    return caught, len(warn_eps) / n_days, prec, warn.sum() / n_days


def brier(p, yy): return float(np.mean((p - yy) ** 2))


prev = y.mean()
print(f"\nPOOLED OUT-OF-FOLD (n={len(y)} rows, {n_days} scored days, lows {prev*100:.1f}%, "
      f"{len(low_eps)} actual low episodes)")
print(f"  constant-prevalence Brier {brier(np.full(len(y), prev), y):.4f}")
print(f"  {'set':3s} {'Brier':>7} {'logloss':>8} {'AUC':>6} | at p>={THR:.2f}: {'episodes caught':>15} "
      f"{'alert eps/day':>13} {'precision':>9} {'warn iv/day':>11}")
for k in SETS:
    c, epd, prec, wpd = episode_stats(P[k])
    print(f"  {k:3s} {brier(P[k], y):7.4f} {log_loss(y, np.clip(P[k], 1e-9, 1-1e-9)):8.4f} "
          f"{roc_auc_score(y, P[k]):6.4f} | {c:>7d}/{len(low_eps):<7d} {epd:13.2f} {prec*100:8.1f}% {wpd:11.1f}")

# ---------- paired Brier differences vs F0, block bootstrap ----------
SEED = 20260923
B = 2000
nb = len(blocks)
rng = np.random.default_rng(SEED)
draws = rng.integers(0, nb, size=(B, nb))
per_block = {k: np.array([np.sum((P[k][bid == b] - y[bid == b]) ** 2) for b in range(nb)]) for k in SETS}
block_n = np.array([np.sum(bid == b) for b in range(nb)])


def boot_diff(kA, kB, sq_a=None, sq_b=None, nn=None):
    sq_a = per_block[kA] if sq_a is None else sq_a
    sq_b = per_block[kB] if sq_b is None else sq_b
    nn = block_n if nn is None else nn
    d = (sq_a[draws].sum(1) - sq_b[draws].sum(1)) / np.maximum(nn[draws].sum(1), 1)
    point = (sq_a.sum() - sq_b.sum()) / nn.sum()
    return point, np.percentile(d, 2.5), np.percentile(d, 97.5), float(np.mean(d < 0))


print(f"\npaired Brier differences vs F0 (block bootstrap over the {nb} test blocks, {B} draws, seed {SEED}); "
      f"negative = new set better:")
print(f"  {'comparison':10s} {'diff':>8} {'ci_lo':>8} {'ci_hi':>8} {'frac<0':>7}")
for k in [k for k in SETS if k != "F0"]:
    d, lo, hi, fr = boot_diff(k, "F0")
    print(f"  {k + ' - F0':10s} {d:+8.4f} {lo:+8.4f} {hi:+8.4f} {fr:7.3f}")

# ---------- Brier by regime ----------
hour = df.hour[idx].to_numpy()
msc = F.min_since_carbs[idx].to_numpy()
hse = df.hrs_since_exercise[idx].to_numpy()
regimes = [
    ("all", np.ones(len(y), bool)),
    ("overnight (hours 0-6)", (hour >= 0) & (hour < 6)),
    ("post-meal (<= 3 h since carbs)", msc <= 180),
    ("post-exercise (<= 4 h since end)", hse <= 4),
    ("below 4 now", gl < 4),
    ("2-5 h after last carb bolus", (msc >= 120) & (msc <= 300)),
]
print("\nBrier by regime (pooled out-of-fold rows):")
print(f"  {'regime':34s} {'n':>6} {'lows':>6} " + " ".join(f"{k:>7}" for k in SETS))
for name, m in regimes:
    print(f"  {name:34s} {int(m.sum()):6d} {y[m].mean()*100:5.1f}% "
          + " ".join(f"{brier(P[k][m], y[m]):7.4f}" for k in SETS))

print("\npaired Brier differences vs F0 by regime (same block bootstrap):")
print(f"  {'regime':34s} " + " ".join(f"{k + '-F0':>24}" for k in SETS if k != "F0"))
reg_ci = {}
for name, m in regimes:
    cells = []
    for k in [k for k in SETS if k != "F0"]:
        sq_k = np.array([np.sum((P[k][(bid == b) & m] - y[(bid == b) & m]) ** 2) for b in range(nb)])
        sq_0 = np.array([np.sum((P["F0"][(bid == b) & m] - y[(bid == b) & m]) ** 2) for b in range(nb)])
        nn = np.array([np.sum((bid == b) & m) for b in range(nb)])
        d, lo, hi, fr = boot_diff(k, "F0", sq_k, sq_0, nn)
        reg_ci[(name, k)] = (d, lo, hi)
        cells.append(f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print(f"  {name:34s} " + " ".join(f"{c:>24}" for c in cells))

# ---------- feature importance (gain) for F1 on the last fold, for the record ----------
imp_cols = ["iob5_bolus_u", "iob5_basal_u", "insulin_activity_u_per_5min", "cob_g",
            "iob5_total_x_exercising", "iob5_total_x_post_ex_4h"]
last_s = blocks[-1][0]
tr_idx = df.index[usable & (df.t < last_s)].to_numpy()
cut = int(len(tr_idx) * 0.8)
m3 = lgb.LGBMClassifier(**params)
m3.fit(F[F3_COLS].loc[tr_idx[:cut]], ylow[tr_idx[:cut]],
       eval_set=[(F[F3_COLS].loc[tr_idx[cut:]], ylow[tr_idx[cut:]])],
       callbacks=[lgb.early_stopping(100, verbose=False)])
gain = pd.Series(m3.booster_.feature_importance("gain"), index=F3_COLS)
rank = gain.rank(ascending=False).astype(int)
print(f"\nF3 gain-importance rank (of {len(F3_COLS)}), last fold's model, for the record:")
for c in imp_cols:
    print(f"  {c:30s} rank {rank[c]:2d}  share {gain[c]/gain.sum()*100:5.1f}%")
print("  top 8 by gain:", ", ".join(gain.sort_values(ascending=False).head(8).index))

# ---------- the one plain sentence ----------
d_all, lo_all, hi_all, _ = boot_diff("F1", "F0")
d_tr, lo_tr, hi_tr = reg_ci[("2-5 h after last carb bolus", "F1")]
b0 = brier(P["F0"], y)
print(f"\nReplacing the 2-hour straight-line IOB with the 5-hour curve changed the low classifier's "
      f"Brier score by {d_all:+.4f} (interval {lo_all:+.4f} to {hi_all:+.4f}; F0 Brier {b0:.4f}), "
      f"and in the 2 to 5 h post-meal window by {d_tr:+.4f} (interval {lo_tr:+.4f} to {hi_tr:+.4f}).")
print("Negative means the curve helped; the person's logged prediction was 'not at all'.")

print("""
Method notes:
 1. Doses are taken from aligned_5min.csv at interval level (bolus_u, est_basal_u,
    carbs_g): a bolus is aged from the start of the 5-minute interval that holds it
    (full dose in that row), and basal delivery in an interval is seen from the next
    row at mid-interval age, matching est_basal_iob_u.
 2. insulin_activity_u_per_5min = activity(t) x 5 min summed over bolus AND basal doses;
    carbs_absorbed_g_per_5min = C x 5/180 for every carb entry younger than 3 h.
 3. Folds: blocks of cv_block_days starting cv_min_train_days after the first usable
    row; a tail block shorter than cv_min_block_days is folded into the block before
    it; a block with no usable rows (a gap between windows) is skipped.
 4. Inside each training fold the last 20 % of rows (time order) is the validation
    fold; it drives early stopping (patience 100) and fits the isotonic calibrator;
    the model is fitted on the first 80 %. The test block never influences its own model.
 5. A low episode is a run of readings < 4 with gaps <= 15 min merged; caught if any
    warning fired in the 30 min before it or during it; an alert episode is a maximal
    run of warned intervals. Days = distinct calendar days with scored rows.
 6. "Interval precision" = share of warned intervals (p >= 0.60) whose target is 1.
 7. Regimes: post-meal = min_since_carbs <= 180; post-exercise = hrs_since_exercise
    <= 4 (workout END); 2-5 h window = 120 <= min_since_carbs <= 300.
 8. Block bootstrap: resample the test blocks with replacement (same draws for every
    comparison), Brier difference = pooled difference over the resampled rows; 2,000
    draws, fixed seed, percentile interval.
 9. Nothing crosses a window fence: the v2 features are computed per window and
    `clean` already excludes rows whose past hour spans a window boundary.
""")
