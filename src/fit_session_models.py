"""
fit_session_models.py - pooled windows, baselines and the small logistic models.

Personal glucose-analysis project. All glucose in mmol/L. Analysis only:
nothing here informs a dosing decision.

Reads  sessions.csv and control_windows.csv from PROCESSED_DIR, or any pair
       of files given with --sessions / --controls (the published
       results/sessions_public.csv and results/controls_public.csv work: they
       carry session_uid instead of session_id and a `fold` column instead of
       a start time).
Writes pooled_windows.csv (PROCESSED_DIR, or --pooled-out).

Explore only. Usable explore sessions and their control windows are the
whole universe of this script; prospective sessions are filtered out before
anything is built, so nothing prospective is printed, summarised or fitted on.

Models (fixed list, no tuning, no selection, no other model types):
  weighted L2 logistic regression, C = 1.0, lbfgs, predictors standardised
  (mean 0, sd 1) on the training rows only; row weight 1 for a session and
  1/k for each of that session's k controls. B0 is the weighted training base
  rate. The M_int interaction is exercise * insulin_total_4h on the raw scale,
  standardised afterwards. Odds ratios are converted back to the raw scale
  (coef / training sd) so they read per 1 mmol/L, per 1 U, or 0 -> 1 for an
  indicator.

Evaluation: forward chaining by fold of the SESSION start (controls travel
with their session). fold = calendar month index (settings fold_by = "month")
or 7-day block index ("week"); the first fold is training-only. Out-of-fold
predictions are pooled; paired Brier differences use a cluster bootstrap over
set_id.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
from config import PROCESSED_DIR, S

EXPLORE_LAST_START = config.ts("explore_last_start")
SEED = 20260921
BOOT_N = 2000
C_FIXED = 1.0
SPORTS = ["gym", "cricket_indoor", "cricket_outdoor", "football"]
BASE_PRED = ["g0", "slope30", "bolus_iob_u", "est_basal_iob_u", "insulin_total_4h", "mins_since_meal"]
POOLED_COLS = ["set_id", "exercise", "cat2", "g0", "slope30", "bolus_iob_u", "est_basal_iob_u",
               "insulin_total_4h", "mins_since_meal", "low_any", "below3_any", "overridden",
               "weight", "fold"]

MODELS = {
    "B0":      [],
    "B1":      ["g0", "slope30"],
    "M_ex":    ["g0", "slope30", "exercise"],
    "M_ins2":  ["g0", "slope30", "bolus_iob_u", "est_basal_iob_u"],
    "M_ins4":  ["g0", "slope30", "insulin_total_4h"],
    "M_full":  ["g0", "slope30", "insulin_total_4h", "exercise"],
    "M_int":   ["g0", "slope30", "insulin_total_4h", "exercise", "ex_x_ins4"],
    "M_sport": ["g0", "slope30", "insulin_total_4h"] + [f"sp_{s}" for s in SPORTS],
}
ALL_MODELS = list(MODELS)
UNIT = {
    "g0": "per 1 mmol/L", "slope30": "per 1 mmol/L per 30 min", "insulin_total_4h": "per 1 U",
    "bolus_iob_u": "per 1 U", "est_basal_iob_u": "per 1 U", "exercise": "exercise vs rest",
    "ex_x_ins4": "exercise x insulin_total_4h, per 1 U", "sp_gym": "gym vs rest",
    "sp_cricket_indoor": "cricket_indoor vs rest", "sp_cricket_outdoor": "cricket_outdoor vs rest",
    "sp_football": "football vs rest",
}


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Part 1: pooled table
# --------------------------------------------------------------------------
def fold_of(start):
    """Fold index of a session start: months since the first data month, or
    7-day blocks since the first window start."""
    t0 = config.data_start()
    if S["fold_by"] == "month":
        return (start.dt.year - t0.year) * 12 + (start.dt.month - t0.month)
    return ((start - t0).dt.days // 7)


def normalise(Sess, C):
    """Accept the private tables (session_id + start) or the public ones
    (session_uid + fold). Returns (sessions, controls) with session_id and fold."""
    Sess, C = Sess.copy(), C.copy()
    for D in (Sess, C):
        if "session_uid" in D.columns and "session_id" not in D.columns:
            D.rename(columns={"session_uid": "session_id"}, inplace=True)
        if "overridden" not in D.columns:
            D["overridden"] = 0
    if "session_split" in Sess.columns:
        Sess = Sess[Sess["session_split"] == "explore"]
    if "usable" in Sess.columns:
        Sess = Sess[Sess["usable"] == 1]
    if "start" in Sess.columns:
        Sess["start"] = pd.to_datetime(Sess["start"])
        assert (Sess["start"] <= EXPLORE_LAST_START).all(), "prospective session in the explore set"
        if "fold" not in Sess.columns:
            Sess["fold"] = fold_of(Sess["start"]).astype(int)
    assert "fold" in Sess.columns, "sessions need a start time or a fold column"
    if "start" in C.columns:
        C["start"] = pd.to_datetime(C["start"])
        assert (C["start"] <= EXPLORE_LAST_START).all(), "control window after the explore cut-off"
    return Sess, C


def build_pooled(Sess, C):
    U = Sess.copy()
    C = C[C["session_id"].isin(U["session_id"])].copy()
    need = ["est_basal_iob_u", "slope30", "overridden"]
    missing = [c for c in need if c not in C.columns]
    assert not missing, f"control table lacks {missing}"

    feat = ["g0", "slope30", "bolus_iob_u", "est_basal_iob_u", "insulin_total_4h",
            "mins_since_meal", "low_any", "below3_any", "overridden"]
    sess = U[["session_id", "cat2"] + feat].rename(columns={"session_id": "set_id"})
    sess.insert(1, "exercise", 1)
    ctrl = C[["session_id"] + feat].rename(columns={"session_id": "set_id"})
    ctrl.insert(1, "exercise", 0)
    ctrl.insert(2, "cat2", "rest")
    # keep each set's controls in their file order (nearest date first in the
    # private table); a stable sort below preserves it
    ctrl["_ord"] = np.arange(len(ctrl))
    sess["_ord"] = -1
    P = pd.concat([sess, ctrl], ignore_index=True)
    P["fold"] = P["set_id"].map(U.set_index("session_id")["fold"]).astype(int)

    miss = P[BASE_PRED].isna().any(axis=1)
    dropped = P[miss].copy()
    P = P[~miss].copy()

    k = P[P["exercise"] == 0].groupby("set_id").size()
    kk = P["set_id"].map(k).astype(float)
    P["weight"] = np.where(P["exercise"] == 1, 1.0, 1.0 / kk)
    for c in ["low_any", "below3_any", "overridden", "exercise"]:
        P[c] = P[c].astype(int)
    P["ex_x_ins4"] = P["exercise"] * P["insulin_total_4h"]
    for s in SPORTS:
        P[f"sp_{s}"] = (P["cat2"] == s).astype(int)
    P = (P.sort_values(["fold", "set_id", "exercise", "_ord"], ascending=[True, True, False, True], kind="stable")
         .drop(columns="_ord").reset_index(drop=True))
    return P, dropped


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Fit:
    """One weighted L2 logistic fit. cols == [] -> weighted base rate (B0)."""

    def __init__(self, cols, X, y, w):
        self.cols = cols
        self.base = float(np.average(y, weights=w))
        self.clf = None
        self.degenerate = None
        if not cols:
            return
        if len(np.unique(y)) < 2:
            self.degenerate = "single outcome class in training rows -> base rate used"
            return
        X = np.asarray(X, float)
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.const = sd == 0
        self.sd = np.where(self.const, 1.0, sd)
        self.clf = LogisticRegression(C=C_FIXED, solver="lbfgs", max_iter=5000)
        self.clf.fit((X - self.mu) / self.sd, y, sample_weight=w)

    def predict(self, X):
        if self.clf is None:
            return np.full(len(X), self.base)
        X = np.asarray(X, float)
        return self.clf.predict_proba((X - self.mu) / self.sd)[:, 1]

    def raw_coef(self):
        if self.clf is None:
            return np.full(len(self.cols), np.nan)
        b = self.clf.coef_[0] / self.sd
        return np.where(self.const, np.nan, b)


def xmat(D, cols):
    return D[cols].values if cols else np.zeros((len(D), 0))


def wbrier(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def wlogloss(y, p, w):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def wauc(y, p, w):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p, sample_weight=w))


# --------------------------------------------------------------------------
# Part 3: forward chaining and pooled out-of-fold predictions
# --------------------------------------------------------------------------
def forward_chain(P, outcome, models):
    oof = P[["set_id", "exercise", "weight", "fold", outcome]].copy()
    for m in models:
        oof[m] = np.nan
    fold_rows, notes = [], []
    test_folds = sorted(P["fold"].unique())[1:]        # the first fold is training-only
    for tf in test_folds:
        tr = P[P["fold"] < tf]
        te = P[P["fold"] == tf]
        if len(te) == 0 or len(tr) == 0:
            notes.append(f"test fold {tf}: no test rows or no training rows; skipped")
            continue
        for m in models:
            cols = MODELS[m]
            f = Fit(cols, xmat(tr, cols), tr[outcome].values, tr["weight"].values)
            oof.loc[te.index, m] = f.predict(xmat(te, cols))
            if f.degenerate:
                notes.append(f"test fold {tf}, {m}: {f.degenerate}")
        fold_rows.append(dict(test_fold=tf, train_sets=tr["set_id"].nunique(), train_rows=len(tr),
                              train_rate_w=np.average(tr[outcome], weights=tr["weight"]),
                              test_sets=te["set_id"].nunique(), test_rows=len(te),
                              test_pos=int(te[outcome].sum()),
                              test_rate_w=np.average(te[outcome], weights=te["weight"])))
    oof = oof.dropna(subset=models)
    return oof, pd.DataFrame(fold_rows), notes


def headline(oof, outcome, models):
    y = oof[outcome].values
    w = oof["weight"].values
    rows = []
    for m in models:
        p = oof[m].values
        rows.append(dict(model=m, brier_w=wbrier(y, p, w), logloss_w=wlogloss(y, p, w), auc_w=wauc(y, p, w),
                         n_rows=len(oof), n_sets=oof["set_id"].nunique(), sum_w=float(w.sum())))
    return pd.DataFrame(rows).set_index("model")


def paired_table(oof, outcome, pairs, seed=SEED, n_boot=BOOT_N):
    y = oof[outcome].values
    w = oof["weight"].values
    codes, uniq = pd.factorize(oof["set_id"])
    n = len(uniq)
    den = np.bincount(codes, weights=w, minlength=n)
    models = sorted({m for pr in pairs for m in pr})
    num = {m: np.bincount(codes, weights=w * (oof[m].values - y) ** 2, minlength=n) for m in models}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    den_b = den[idx].sum(axis=1)
    rows = []
    for a, b in pairs:
        point = num[a].sum() / den.sum() - num[b].sum() / den.sum()
        d = (num[a][idx].sum(axis=1) - num[b][idx].sum(axis=1)) / den_b
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(comparison=f"{a} - {b}", diff_brier=point, ci_lo=lo, ci_hi=hi,
                         frac_draws_negative=float((d < 0).mean())))
    return pd.DataFrame(rows).set_index("comparison")


def calibration(oof, outcome, models):
    out = {}
    for m in models:
        p = oof[m]
        bins = pd.qcut(p, 3, labels=False, duplicates="drop")
        rows = []
        for b in sorted(pd.unique(bins)):
            d = oof[bins == b]
            rows.append(dict(bin=["low", "mid", "high"][int(b)], n=len(d), n_sessions=int(d["exercise"].sum()),
                             p_min=float(d[m].min()), p_max=float(d[m].max()),
                             mean_pred_w=float(np.average(d[m], weights=d["weight"])),
                             obs_rate_w=float(np.average(d[outcome], weights=d["weight"])),
                             obs_rate_unw=float(d[outcome].mean())))
        out[m] = pd.DataFrame(rows).set_index("bin")
    return out


def run_evaluation(P, outcome, models, pairs, label):
    hdr(f"Forward-chaining evaluation: outcome = {outcome}, {label}")
    oof, folds, notes = forward_chain(P, outcome, models)
    print("folds (train = all sets whose session started in an earlier fold):")
    print(folds.to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(folds) else "  none")
    for nt in notes:
        print("  note:", nt)
    if len(oof) == 0:
        print("no out-of-fold rows: nothing to evaluate")
        return oof
    print(f"\npooled out-of-fold rows: {len(oof)}  sets: {oof['set_id'].nunique()}  "
          f"(first-fold sets are training-only and never scored)")
    print("\nheadline (weighted Brier / log loss / AUC; lower Brier and log loss are better):")
    print(headline(oof, outcome, models).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\npaired Brier differences, cluster bootstrap over set_id ({BOOT_N} draws, seed {SEED}); "
          f"negative = first model better:")
    print(paired_table(oof, outcome, pairs).to_string(float_format=lambda v: f"{v:.4f}"))
    return oof


# --------------------------------------------------------------------------
# Part 4: odds ratios with cluster bootstrap
# --------------------------------------------------------------------------
def fit_raw(P, model, outcome):
    cols = MODELS[model]
    f = Fit(cols, xmat(P, cols), P[outcome].values, P["weight"].values)
    return f.raw_coef()


def boot_raw(P, model, outcome, seed=SEED, n_boot=BOOT_N):
    cols = MODELS[model]
    codes, uniq = pd.factorize(P["set_id"])
    n = len(uniq)
    groups = [np.flatnonzero(codes == i) for i in range(n)]
    X = xmat(P, cols)
    y = P[outcome].values
    w = P["weight"].values
    rng = np.random.default_rng(seed)
    coefs = np.full((n_boot, len(cols)), np.nan)
    n_skip = 0
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        ii = np.concatenate([groups[i] for i in pick])
        yb = y[ii]
        if yb.min() == yb.max():
            n_skip += 1
            continue
        f = Fit(cols, X[ii], yb, w[ii])
        coefs[b] = f.raw_coef()
    return coefs, n_skip


def or_table(P, model, outcome, extra_int=True):
    cols = MODELS[model]
    b = fit_raw(P, model, outcome)
    coefs, n_skip = boot_raw(P, model, outcome)
    rows = []
    for j, c in enumerate(cols):
        ok = ~np.isnan(coefs[:, j])
        lo, hi = (np.percentile(coefs[ok, j], [2.5, 97.5]) if ok.sum() else (np.nan, np.nan))
        rows.append(dict(term=c, unit=UNIT[c], OR=np.exp(b[j]), ci_lo=np.exp(lo), ci_hi=np.exp(hi),
                         boot_valid=int(ok.sum())))
    if extra_int and model == "M_int":
        med = float(P["insulin_total_4h"].median())
        je, ji = cols.index("exercise"), cols.index("ex_x_ins4")
        comb = coefs[:, je] + coefs[:, ji] * med
        ok = ~np.isnan(comb)
        lo, hi = (np.percentile(comb[ok], [2.5, 97.5]) if ok.sum() else (np.nan, np.nan))
        rows.append(dict(term="exercise @ median insulin_total_4h",
                         unit=f"exercise vs rest at insulin_total_4h = {med:.2f} U",
                         OR=np.exp(b[je] + b[ji] * med), ci_lo=np.exp(lo), ci_hi=np.exp(hi), boot_valid=int(ok.sum())))
    T = pd.DataFrame(rows).set_index("term")
    return T, n_skip


def print_or(P, model, outcome):
    T, n_skip = or_table(P, model, outcome)
    print(f"\n{model}  (fit on all {len(P)} pooled rows, {P['set_id'].nunique()} sets; "
          f"cluster bootstrap {BOOT_N} draws, seed {SEED}, {n_skip} single-class draws skipped)")
    print(T.to_string(float_format=lambda v: f"{v:.3f}"))
    return T


def sentence(T, term, what, outcome_txt):
    r = T.loc[term]
    return (f"After accounting for starting glucose, the 30-min slope and insulin over the "
            f"previous 4 h, the odds of {outcome_txt} in {what} are {r['OR']:.2f} times "
            f"those of a matched rest window (95% cluster-bootstrap interval "
            f"{r['ci_lo']:.2f} to {r['ci_hi']:.2f}).")


# --------------------------------------------------------------------------
def load_tables(args):
    sess_path = args.sessions or (PROCESSED_DIR / "sessions.csv")
    ctrl_path = args.controls or (PROCESSED_DIR / "control_windows.csv")
    Sess = pd.read_csv(sess_path)
    C = pd.read_csv(ctrl_path)
    print(f"sessions: {sess_path}\ncontrols: {ctrl_path}")
    return normalise(Sess, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", help="session table (default PROCESSED_DIR/sessions.csv)")
    ap.add_argument("--controls", help="control table (default PROCESSED_DIR/control_windows.csv)")
    ap.add_argument("--pooled-out", help="where to write the pooled table")
    ap.add_argument("--or-csv", help="also write the M_full / M_int / M_sport odds-ratio tables here")
    args = ap.parse_args()
    out_path = args.pooled_out or (PROCESSED_DIR / "pooled_windows.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 300)
    Sess, C = load_tables(args)

    hdr("Part 1. Pooled table (usable explore sessions + their control windows)")
    P, dropped = build_pooled(Sess, C)
    P[POOLED_COLS].to_csv(out_path, index=False)
    n_sess = int((P["exercise"] == 1).sum())
    n_ctrl = int((P["exercise"] == 0).sum())
    print(f"Wrote {out_path}: {len(P)} rows = {n_sess} sessions + {n_ctrl} controls, {P['set_id'].nunique()} sets")
    print(f"rows dropped for a missing predictor ({', '.join(BASE_PRED)}): {len(dropped)}")
    if len(dropped):
        print(dropped[["set_id", "exercise", "cat2"] + BASE_PRED].to_string(index=False))
    print(f"controls per set after the drop: "
          f"{P[P['exercise'] == 0].groupby('set_id').size().value_counts().sort_index().to_dict()}")
    print(f"sum of weights: sessions {P.loc[P['exercise'] == 1, 'weight'].sum():.1f}, "
          f"controls {P.loc[P['exercise'] == 0, 'weight'].sum():.1f}")
    print("\nrows by cat2:")
    print(P["cat2"].value_counts().to_string())
    fold_word = "calendar month" if S["fold_by"] == "month" else "7-day block"
    print(f"\nsets by fold ({fold_word} of the session start):")
    print(P[P["exercise"] == 1].groupby("fold").size().rename("sessions").to_string())
    print("\noutcome rates, exercise vs rest (unweighted over rows / weighted):")
    for oc in ["low_any", "below3_any"]:
        for ex in (1, 0):
            d = P[P["exercise"] == ex]
            print(f"  {oc:11s} exercise={ex}: {int(d[oc].sum()):3d}/{len(d):3d} = {d[oc].mean():.3f}  "
                  f"weighted {np.average(d[oc], weights=d['weight']):.3f}")
    print("\npredictor summary (pooled rows):")
    print(P[BASE_PRED].describe().T[["count", "mean", "std", "min", "50%", "max"]].to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"overridden sets: {sorted(P.loc[P['overridden'] == 1, 'set_id'].unique())}")

    hdr("Part 2. Models (fixed; no tuning)")
    for m, cols in MODELS.items():
        print(f"  {m:8s}: {'constant (weighted training base rate)' if not cols else ' + '.join(cols)}")
    print(f"  all: weighted L2 logistic regression, C = {C_FIXED}, standardised on training rows")

    pairs = [(m, "B1") for m in ALL_MODELS if m != "B1"] + [("M_ins4", "M_ins2"), ("M_full", "M_ins4")]
    oof = run_evaluation(P, "low_any", ALL_MODELS, pairs, "all pooled rows")
    if len(oof):
        hdr("Part 3 (cont.). Calibration, tertiles of out-of-fold predicted probability")
        for m, T in calibration(oof, "low_any", ["B1", "M_ins4", "M_full"]).items():
            print(f"\n{m}")
            print(T.to_string(float_format=lambda v: f"{v:.3f}"))

    hdr("Part 4. Odds ratios, fit once on all pooled explore rows (low_any)")
    T_full = print_or(P, "M_full", "low_any")
    T_int = print_or(P, "M_int", "low_any")
    T_sport = print_or(P, "M_sport", "low_any")
    print("\n(a) " + sentence(T_full, "exercise", "an exercise session (M_full)",
                              "a low (< 4.0 mmol/L during the window or in the 2 h after it)"))
    print("(b) " + sentence(T_sport, "sp_football", "a football session (M_sport)",
                            "a low (< 4.0 mmol/L during the window or in the 2 h after it)"))
    if args.or_csv:
        rows = []
        for name, T in (("M_full", T_full), ("M_int", T_int), ("M_sport", T_sport)):
            for term, r in T.iterrows():
                rows.append(dict(model=name, term=term, unit=r["unit"], OR=r["OR"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
        pd.DataFrame(rows).to_csv(args.or_csv, index=False)
        print(f"\nodds-ratio tables written to {args.or_csv}")

    hdr("Part 5a. Sensitivity: overridden sessions and their controls removed")
    ov = sorted(P.loc[P["overridden"] == 1, "set_id"].unique())
    P_no = P[~P["set_id"].isin(ov)].copy()
    print(f"removed sets: {ov}  -> {len(P_no)} rows, {P_no['set_id'].nunique()} sets")
    pairs_b1 = [(m, "B1") for m in ALL_MODELS if m != "B1"]
    run_evaluation(P_no, "low_any", ALL_MODELS, pairs_b1, "overridden sets removed")
    print("\nPart 4(a) repeated without the overridden sets:")
    T_full_no = print_or(P_no, "M_full", "low_any")
    print("\n(a) " + sentence(T_full_no, "exercise", "an exercise session (M_full, overridden sets removed)",
                              "a low (< 4.0 mmol/L)"))

    hdr("Part 5b. Sensitivity: outcome = below3_any (B1, M_ins4, M_full only)")
    sub = ["B1", "M_ins4", "M_full"]
    pairs_sub = [("M_ins4", "B1"), ("M_full", "B1"), ("M_full", "M_ins4")]
    run_evaluation(P, "below3_any", sub, pairs_sub, "all pooled rows")
    print("\nPart 4(a) repeated with below3_any:")
    T_full_b3 = print_or(P, "M_full", "below3_any")
    print("\n(a) " + sentence(T_full_b3, "exercise", "an exercise session (M_full)",
                              "a reading below 3.0 mmol/L during the window or in the 2 h after it"))

    hdr("Part 5 summary. Exercise odds ratio (M_full) across the three settings")
    summ = pd.DataFrame({
        "low_any, all rows": T_full.loc["exercise", ["OR", "ci_lo", "ci_hi"]],
        "low_any, overridden removed": T_full_no.loc["exercise", ["OR", "ci_lo", "ci_hi"]],
        "below3_any, all rows": T_full_b3.loc["exercise", ["OR", "ci_lo", "ci_hi"]],
    }).T
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))

    hdr("Method notes")
    print("""
 1. Pooled table = usable explore sessions plus every control window that
    belongs to one of them. Controls are explore-only by construction.
 2. Rows with any missing value among the base predictors are dropped
    individually; control weights use k = controls REMAINING in the pooled
    table, so each set's controls still sum to weight 1.
 3. pooled_windows.csv carries weight and fold beyond the requested columns.
 4. Fold = calendar month (or 7-day block) of the SESSION start; controls take
    their session's fold. Training = every set with a strictly earlier fold;
    the first fold is training-only.
 5. Standardisation: unweighted mean/sd of the training rows; a predictor that
    is constant in the training rows is left at 0 after centring.
 6. Weighted logistic regression is scikit-learn's: 0.5 * ||beta||^2 +
    C * sum_i w_i * logloss_i, intercept unpenalised, lbfgs.
 7. Weighted Brier / log loss = sum(w * loss) / sum(w); AUC uses sample_weight.
 8. Paired Brier difference: cluster bootstrap over set_id on the pooled
    out-of-fold predictions (no refitting), percentile interval; the same
    resampled sets are used for every comparison in a table. The set order in
    the pooled table is (fold, set_id), so the draws depend on the identifiers.
 9. Calibration tertiles are unweighted terciles of the out-of-fold
    predicted probability.
10. Odds ratios are on the raw scale: standardised coefficient divided by the
    training sd of that column, exponentiated. For M_int the 'exercise' row is
    the exercise effect at insulin_total_4h = 0; an extra row gives the
    exercise OR at the pooled median insulin_total_4h.
11. OR bootstrap: resample set_id with replacement, refit the same model,
    percentile interval of the raw-scale coefficient; single-class draws are
    skipped and counted.
""")


if __name__ == "__main__":
    main()
