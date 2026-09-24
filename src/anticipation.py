"""
anticipation.py - how far ahead of a session's start could a system have known
a session was coming, without the person telling it, and at what false-alarm
cost?

Personal glucose-analysis project. All glucose in mmol/L. Analysis only:
nothing here informs a dosing decision. No model is fitted: three fixed rules
with fixed thresholds.

Reads  PROCESSED_DIR/aligned_5min.csv (never modified) and sessions.csv
Prints the report to stdout and writes anticipation_summary.csv (the rule x
horizon table; no timestamps).

Grid of candidate decision times: every 5-min row of the explore period (or
the anticipation_grid_start / _end settings) with clock time in the
grid_clock range, not inside a session [start, end] and not within 2 h after
one. Every explore session in sessions.csv (usable or not, any category)
defines the exclusion, the labels and the routine history; "sessions caught"
counts usable explore sessions only. Prospective rows are dropped first.

Label(t, H) = 1 if any session starts in (t, t + H], H in {30, 60, 90} min.

Rules at row t:
  R1 routine   for horizon H: a 30-min clock slot that overlaps (t, t + H] had
               a session start (any category) on the same weekday in at least
               2 of the previous 4 weeks (weeks with no data count 0).
  R2 movement  steps over the last 15 min (rows t - 10, t - 5, t) >= 300, or
               mean hr_mean over the last 10 min (rows t - 5, t) >= day
               baseline + 20 bpm. Baseline = median hr_mean 03:00-05:55 that
               day; if none, the median of the previous 7 days' baselines; if
               none, HR cannot fire that day.
  R3           R1 or R2.
  gate         insulin_total_4h >= 5 U: bolus_u + est_basal_u over [t - 4 h, t).
Each rule is reported with and without the gate (rule AND gate).

The "usual training times" used only in the cost framing come from the
private settings (usual_training_times); with none set (null or an empty
list, as in the published example) no usual-day split is reported.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import config
from config import PROCESSED_DIR, S
from build_sessions import load_aligned, hdr, EXPLORE_LAST_START, is_indoor_season     # noqa: E402

SESS_PATH = PROCESSED_DIR / "sessions.csv"
SUMMARY_PATH = PROCESSED_DIR / "anticipation_summary.csv"

GRID_START = config.ts("anticipation_grid_start") or (config.data_start() + pd.Timedelta(days=1))
GRID_END = config.ts("anticipation_grid_end") or EXPLORE_LAST_START
CLOCK_LO, CLOCK_HI = [int(v) for v in S["grid_clock"]]
POST_EXCL = pd.Timedelta(hours=2)
HORIZONS = [30, 60, 90]
STEP_NS = np.timedelta64(5, "m")
ROUTINE_WEEKS = 4
ROUTINE_MIN_WEEKS = 2
SLOT_MIN = 30
N_SLOTS = 24 * 60 // SLOT_MIN
STEPS_15 = 300.0
HR_RISE = 20.0
GATE_U = 5.0
CAT2_ORDER = ["gym", "cricket_indoor", "cricket_outdoor", "football", "cycling"]
BASE_DATE = (GRID_START - pd.Timedelta(days=7 * ROUTINE_WEEKS + 7)).date()
HOUR_WORDS = {30: "half an hour", 60: "an hour", 90: "an hour and a half"}
USUAL = S.get("usual_training_times") or []


def mins(td):
    return td / np.timedelta64(1, "m")


def usual_training_time(ts):
    ts = pd.Timestamp(ts)
    winter = is_indoor_season(ts)
    for u in USUAL:
        if ts.weekday() != int(u["weekday"]) or ts.hour < int(u.get("from_hour", 0)):
            continue
        season = u.get("season", "any")
        if season == "any" or (season == "winter" and winter) or (season == "summer" and not winter):
            return True
    return False


def routine_history(Sess):
    n_days = (GRID_END.date() - BASE_DATE).days + 1
    hist = np.zeros((n_days, N_SLOTS), bool)
    for st in Sess["start"]:
        d = (st.date() - BASE_DATE).days
        if d >= 0:
            hist[d, (st.hour * 60 + st.minute) // SLOT_MIN] = True
    return hist


def r1_fire(G, hist, H):
    D = np.array([(d - BASE_DATE).days for d in G.date])
    M = np.asarray(G.hour) * 60 + np.asarray(G.minute)
    s_lo = M // SLOT_MIN
    s_hi = np.minimum((M + H) // SLOT_MIN, N_SLOTS - 1)
    fire = np.zeros(len(G), bool)
    for j in range(N_SLOTS):
        s = s_lo + j
        valid = s <= s_hi
        if not valid.any():
            break
        s_idx = np.minimum(s, N_SLOTS - 1)
        cnt = np.zeros(len(G), int)
        for k in range(1, ROUTINE_WEEKS + 1):
            dk = D - 7 * k
            ok = dk >= 0
            c = np.zeros(len(G), bool)
            c[ok] = hist[dk[ok], s_idx[ok]]
            cnt += c
        fire |= valid & (cnt >= ROUTINE_MIN_WEEKS)
    return fire


def r2_components(F, full_idx):
    steps15 = F["steps"].fillna(0).rolling(3, min_periods=1).sum()
    hr10 = F["hr_mean"].rolling(2, min_periods=1).mean()
    dates = pd.Series(full_idx.date, index=full_idx)
    night = (full_idx.hour >= 3) & (full_idx.hour < 6) & F["hr_mean"].notna().values
    base_day = F["hr_mean"][night].groupby(dates[night]).median()
    baseline, base_src = {}, {}
    for d in sorted(set(dates)):
        if d in base_day.index and pd.notna(base_day[d]):
            baseline[d], base_src[d] = float(base_day[d]), "own"
        else:
            prev = [base_day[d - timedelta(days=k)] for k in range(1, 8) if (d - timedelta(days=k)) in base_day.index]
            if prev:
                baseline[d], base_src[d] = float(np.median(prev)), "7day"
            else:
                baseline[d], base_src[d] = np.nan, "none"
    base_arr = dates.map(baseline).astype(float).values
    with np.errstate(invalid="ignore"):
        hr_fire = hr10.values >= base_arr + HR_RISE
    steps_fire = steps15.values >= STEPS_15
    return steps_fire, hr_fire, base_src, base_arr


def evaluate(fire, label, Gv, U, H):
    st = U["start"].values.astype("datetime64[ns]")
    Hn = np.timedelta64(H, "m")
    idx = np.where(fire)[0]
    fired = Gv[idx]
    if len(idx):
        brk = np.diff(fired) != STEP_NS
        run = np.concatenate([[0], np.cumsum(brk)])
        run_first = fired[np.concatenate([[True], brk])]
        i0 = np.searchsorted(fired, st - Hn, "left")
        i1 = np.searchsorted(fired, st, "left")
        caught = i1 > i0
        j = np.minimum(i0, len(fired) - 1)
        lead = np.where(caught, mins(st - fired[j]), np.nan)
        lead_run = np.where(caught, mins(st - run_first[run[j]]), np.nan)
    else:
        caught = np.zeros(len(U), bool)
        lead = np.full(len(U), np.nan)
        lead_run = np.full(len(U), np.nan)
    fa_rows = int((fire & ~label).sum())
    if len(idx):
        ep = pd.DataFrame({"run": run, "label": label[idx].astype(int), "i": idx})
        agg = ep.groupby("run").agg(any_label=("label", "max"), first_i=("i", "min"), n=("i", "size"))
        fa = agg[agg["any_label"] == 0]
        n_ep, n_fa_ep = len(agg), len(fa)
        fa_first = Gv[fa["first_i"].values]
        fa_len = fa["n"].values * 5
    else:
        n_ep, n_fa_ep, fa_first, fa_len = 0, 0, np.array([], "datetime64[ns]"), np.array([])
    return dict(caught=caught, lead=lead, lead_run=lead_run, fire_rows=int(fire.sum()), fa_rows=fa_rows,
                n_ep=n_ep, n_fa_ep=n_fa_ep, fa_first=fa_first, fa_len=fa_len)


def caught_row(res, U, n_weeks, mask=None):
    m = np.ones(len(U), bool) if mask is None else np.asarray(mask)
    c = res["caught"][m]
    Um = U[m]
    row = {"n": int(m.sum()), "caught": int(c.sum()), "share": (c.mean() if m.sum() else np.nan)}
    for a in (0, 1):
        ma = Um["aerobic"].values == a
        row[f"aer{a}"] = f"{int(c[ma].sum())}/{int(ma.sum())}"
    for cat in CAT2_ORDER:
        mc = Um["cat2"].values == cat
        if (U["cat2"] == cat).any():
            row[cat] = f"{int(c[mc].sum())}/{int(mc.sum())}"
    lead = res["lead"][m]
    row["lead_med"] = np.nanmedian(lead) if c.any() else np.nan
    row["lead_q1"] = np.nanpercentile(lead, 25) if c.any() else np.nan
    row["lead_q3"] = np.nanpercentile(lead, 75) if c.any() else np.nan
    lr = res["lead_run"][m]
    row["leadrun_med"] = np.nanmedian(lr) if c.any() else np.nan
    row["fire_rows/wk"] = res["fire_rows"] / n_weeks
    row["FA_rows/wk"] = res["fa_rows"] / n_weeks
    row["FA_episodes/wk"] = res["n_fa_ep"] / n_weeks
    row["FA_episodes"] = res["n_fa_ep"]
    return row


def fmt(v):
    return f"{v:.2f}"


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 300)
    df = load_aligned()
    df = df[(df["t"] >= GRID_START - pd.Timedelta(hours=4)) & (df["t"] <= GRID_END)].copy()
    assert df["t"].max() <= EXPLORE_LAST_START, "prospective row leaked"
    Sess = pd.read_csv(SESS_PATH, parse_dates=["start", "end"])
    Sess = Sess[Sess["session_split"] == "explore"].copy()
    assert (Sess["start"] <= EXPLORE_LAST_START).all()
    U = Sess[(Sess["usable"] == 1) & (Sess["start"] >= GRID_START)].reset_index(drop=True)
    if len(U) == 0:
        raise SystemExit("no usable sessions inside the anticipation grid")

    full_idx = pd.date_range(GRID_START - pd.Timedelta(hours=4), GRID_END, freq="5min")
    F = df.set_index("t").reindex(full_idx)
    present = F["window"].notna().values
    excl = np.zeros(len(full_idx), bool)
    for st, en in zip(Sess["start"], Sess["end"]):
        i0 = full_idx.searchsorted(st, "left")
        i1 = full_idx.searchsorted(en + POST_EXCL, "right")
        excl[i0:i1] = True
    clock_ok = (full_idx.hour >= CLOCK_LO) & (full_idx.hour < CLOCK_HI)
    grid_mask = present & (full_idx >= GRID_START) & clock_ok & ~excl
    G = full_idx[grid_mask]
    Gv = G.values.astype("datetime64[ns]")
    n_dates = len(set(G.date))
    n_weeks = n_dates / 7

    starts = np.sort(Sess["start"].values.astype("datetime64[ns]"))
    labels = {}
    nxt_i = np.searchsorted(starts, Gv, "right")
    has_next = nxt_i < len(starts)
    nxt = starts[np.minimum(nxt_i, len(starts) - 1)]
    for H in HORIZONS:
        labels[H] = has_next & ((nxt - Gv) <= np.timedelta64(H, "m"))

    hist = routine_history(Sess)
    steps_fire_full, hr_fire_full, base_src, base_arr = r2_components(F, full_idx)
    steps_fire = steps_fire_full[grid_mask]
    hr_fire = hr_fire_full[grid_mask]
    r2 = steps_fire | hr_fire
    ins = F["bolus_u"].fillna(0) + F["est_basal_u"].fillna(0)
    ins4 = ins.rolling(48, min_periods=1).sum().shift(1)
    gate = (ins4.values >= GATE_U)[grid_mask]
    r1 = {H: r1_fire(G, hist, H) for H in HORIZONS}

    def rules_for(H):
        return {"R1 routine": r1[H], "R2 movement": r2, "R3 routine or movement": r1[H] | r2,
                "gate alone (ref)": gate, "R1 and gate": r1[H] & gate, "R2 and gate": r2 & gate,
                "R3 and gate": (r1[H] | r2) & gate}

    print("Analysis only: nothing here informs a dosing decision. Glucose in mmol/L.")
    print("Rules only, thresholds fixed; no model is fitted; no prospective row is read.")
    hdr("0. Universe")
    print(f"grid: {len(G)} candidate decision rows on {n_dates} dates ({n_weeks:.1f} weeks), "
          f"clock {CLOCK_LO:02d}:00-{CLOCK_HI:02d}:00, outside sessions and their 2 h after")
    print(f"explore sessions: {len(Sess)} (all categories, any usable); usable inside the grid: {len(U)} = positives")
    print("\nusable sessions (positives) by cat2 and aerobic:")
    print(pd.crosstab(U["cat2"], U["aerobic"], margins=True).to_string())
    print("\nlabel prevalence (share of grid rows with a session start within H):")
    for H in HORIZONS:
        print(f"  H = {H:2d}: {labels[H].mean():.4f}  ({int(labels[H].sum())} rows)")
    src = pd.Series(base_src)
    src = src[[d >= GRID_START.date() for d in src.index]]
    print("\nR2 HR baseline source by date: " + ", ".join(f"{k} {v}" for k, v in src.value_counts().items()))
    print(f"R2 firing share of grid rows: steps {steps_fire.mean():.3f}, HR {hr_fire.mean():.3f}, either {r2.mean():.3f}")
    print(f"gate (insulin_total_4h >= {GATE_U:.0f} U) true on {gate.mean():.3f} of grid rows")
    for H in HORIZONS:
        print(f"R1 firing share of grid rows at H = {H}: {r1[H].mean():.3f}")

    results = {}
    summary_rows = []
    low_mask = (U["low_any"] == 1).values
    for H in HORIZONS:
        hdr(f"1. H = {H} min: all usable explore sessions (n = {len(U)})")
        rows = {}
        for name, fire in rules_for(H).items():
            res = evaluate(fire, labels[H], Gv, U, H)
            results[(H, name)] = res
            rows[name] = caught_row(res, U, n_weeks)
            summary_rows.append(dict(H=H, rule=name, subset="all", **rows[name]))
        print(pd.DataFrame(rows).T.to_string(float_format=fmt))
        print("\n  caught: rule fired at any grid row in the H min before start; aerN and cat2 columns = caught/n;")
        print("  lead = start minus first firing row in that window; leadrun_med = from the first row of the firing run;")
        print("  FA = fired, no session within H; FA episodes = runs of consecutive firing rows with no label-1 row")

        hdr(f"2. H = {H} min: sessions that included a low (low_any == 1, n = {int(low_mask.sum())})")
        rows_l = {}
        for name in rules_for(H):
            rows_l[name] = caught_row(results[(H, name)], U, n_weeks, low_mask)
            summary_rows.append(dict(H=H, rule=name, subset="low_any", **rows_l[name]))
        Tl = pd.DataFrame(rows_l).T.drop(columns=["fire_rows/wk", "FA_rows/wk", "FA_episodes/wk", "FA_episodes"])
        print(Tl.to_string(float_format=fmt))

    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)

    hdr("3. Cost framing: R1 routine at H = 60")
    for name in ["R1 routine", "R1 and gate"]:
        res = results[(60, name)]
        lead = res["lead"][res["caught"]]
        lr = res["lead_run"][res["caught"]]
        if len(lead):
            print(f"\n{name}: caught {int(res['caught'].sum())} of {len(U)}; lead before the true start "
                  f"median {np.median(lead):.0f} min, Q1 {np.percentile(lead, 25):.0f}, Q3 {np.percentile(lead, 75):.0f}")
            print(f"  lead counted from the first row of the firing run: median {np.median(lr):.0f} min")
        else:
            print(f"\n{name}: no session caught")
        fa_first = pd.DatetimeIndex(res["fa_first"])
        line = f"  false-alarm episodes: {len(fa_first)} in {n_weeks:.1f} weeks ({len(fa_first) / n_weeks:.2f}/wk)"
        if USUAL:
            usual = np.array([usual_training_time(t) for t in fa_first], bool)
            line += f"; on usual training times: {int(usual.sum())}, other times: {int((~usual).sum())}"
        else:
            line += "; no usual training times set, so no usual-day split"
        print(line)
        if len(fa_first):
            print(f"  episode length (min): median {np.median(res['fa_len']):.0f}, max {res['fa_len'].max():.0f}")

    hdr("4. Lead before the true start by rule and horizon (caught sessions; median [Q1, Q3] min)")
    rows = {}
    for H in HORIZONS:
        for name in ["R1 routine", "R2 movement", "R3 routine or movement"]:
            res = results[(H, name)]
            lead = res["lead"][res["caught"]]
            lr = res["lead_run"][res["caught"]]
            rows[(H, name)] = dict(caught=int(res["caught"].sum()),
                                   lead_med=np.median(lead) if len(lead) else np.nan,
                                   lead_q1=np.percentile(lead, 25) if len(lead) else np.nan,
                                   lead_q3=np.percentile(lead, 75) if len(lead) else np.nan,
                                   leadrun_med=np.median(lr) if len(lr) else np.nan,
                                   run_ge30=int((lr >= 30).sum()), run_ge60=int((lr >= 60).sum()))
    print(pd.DataFrame(rows).T.to_string(float_format=fmt))

    hdr("5. Routine history of the usable sessions")
    def routine_count(ts, tol_slots=0):
        d = (ts.date() - BASE_DATE).days
        s = (ts.hour * 60 + ts.minute) // SLOT_MIN
        lo, hi = max(0, s - tol_slots), min(N_SLOTS - 1, s + tol_slots)
        return int(sum(hist[d - 7 * k, lo:hi + 1].any() for k in range(1, ROUTINE_WEEKS + 1) if d - 7 * k >= 0))
    same = pd.Series([routine_count(t) for t in U["start"]], name="same_slot_wks")
    print("usable sessions by same_slot_wks (previous 4 same-weekday dates with a start in the same 30-min slot) x cat2:")
    print(pd.crosstab(same, U["cat2"], margins=True).to_string())

    hdr("Plain sentences (R1 routine, no gate)")
    for H in HORIZONS:
        res = results[(H, "R1 routine")]
        print(f"A routine-only rule would have flagged {100 * res['caught'].mean():.0f}% of sessions "
              f"{HOUR_WORDS[H]} ahead, at {res['n_fa_ep'] / n_weeks:.1f} false-alarm episodes per week.")
    print("\nsame, sessions that included a low:")
    for H in HORIZONS:
        res = results[(H, "R1 routine")]
        c = res["caught"][low_mask]
        if low_mask.any():
            print(f"A routine-only rule would have flagged {100 * c.mean():.0f}% of the sessions that included a low "
                  f"{HOUR_WORDS[H]} ahead, at {res['n_fa_ep'] / n_weeks:.1f} false-alarm episodes per week.")


if __name__ == "__main__":
    main()
