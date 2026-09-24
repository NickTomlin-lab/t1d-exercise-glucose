"""
make_figures.py - the five published figures (matplotlib, PNG, no timestamps).

  (a) low_rate_by_meal_band.png   low rate by meal band, sessions vs matched
                                  controls, Wilson 95% intervals
  (b) odds_ratios.png             odds ratios with cluster-bootstrap intervals
                                  for M_full and M_int
  (c) lead_time_timeline.png      median (Q1-Q3) time of each signal relative
                                  to the session start, sessions with a low
  (d) forecaster_calibration.png  reliability of the 30-min low alert and band
                                  coverage by glucose level, saved test outputs
  (e) alert_lead.png              alert lead before each low episode

Inputs (defaults point at the published results/ tables; every one can be
swapped for the private pipeline's output with the options below):
  --sessions / --controls   results/sessions_public.csv, results/controls_public.csv
  --odds                    results/odds_ratios_public.csv (fit_session_models.py --or-csv)
  --lead                    results/lead_time_summary.csv  (lead_time.py)
  --alert                   results/alert_lead_episodes.csv (alert_lead.py)
  --test-outputs            PROCESSED_DIR/final_test_outputs.csv (+ aligned_5min.csv
                            for the glucose column); figure (d) is skipped if absent
Output folder: FIGURES_DIR (config; --figures-dir).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

import config                      # noqa: E402
from config import PROCESSED_DIR, FIGURES_DIR, REPO_ROOT   # noqa: E402
from build_controls import wilson  # noqa: E402

RESULTS = REPO_ROOT / "results"
BANDS = ["0-2h", "2-4h", "4-6h", "6h+"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fig_meal_band(sess, ctrl, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(BANDS))
    for k, (name, D, off, col) in enumerate((("exercise sessions", sess, -0.18, "#c0392b"),
                                             ("matched rest windows", ctrl, 0.18, "#2c3e50"))):
        rates, lo, hi, ns = [], [], [], []
        for b in BANDS:
            d = D[D["meal_band"] == b]
            n, kk = len(d), int(d["low_any"].sum())
            r = kk / n if n else np.nan
            w = wilson(kk, n)
            rates.append(r); lo.append(w[0]); hi.append(w[1]); ns.append(n)
        rates, lo, hi = np.array(rates), np.array(lo), np.array(hi)
        ax.errorbar(x + off, rates, yerr=[rates - lo, hi - rates], fmt="o", color=col, capsize=4, label=name)
        for xi, r, n in zip(x + off, rates, ns):
            if n:
                ax.annotate(f"n={n}", (xi, r), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} since last meal" for b in BANDS])
    ax.set_ylabel("share with a low (< 4.0 mmol/L)\nduring the window or the 2 h after")
    ax.set_ylim(0, 1)
    ax.set_title("Low rate by time since the last meal (Wilson 95% intervals)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_odds(odds, out):
    models = ["M_full", "M_int"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=False)
    for ax, m in zip(axes, models):
        T = odds[odds["model"] == m].reset_index(drop=True)
        y = np.arange(len(T))[::-1]
        ax.errorbar(T["OR"], y, xerr=[T["OR"] - T["ci_lo"], T["ci_hi"] - T["OR"]], fmt="o", color="#2c3e50", capsize=4)
        ax.axvline(1, color="grey", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yticks(y)
        labels = [f"{t}\n({u})" if len(u) < 34 else t for t, u in zip(T["term"], T["unit"])]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("odds ratio (log scale), 95% cluster-bootstrap interval")
        ax.set_title(f"{m}: odds of a low", loc="left")
        for yi, r in zip(y, T.itertuples()):
            ax.annotate(f"{r.OR:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", (r.ci_hi, yi), textcoords="offset points",
                        xytext=(6, -3), fontsize=7.5)
        ax.set_xlim(left=min(0.08, T["ci_lo"].min() * 0.8), right=max(30, T["ci_hi"].max() * 4))
    fig.suptitle("Sessions against matched rest windows: odds ratios from the pooled logistic models", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_lead_timeline(lead, out):
    L = lead[(lead["group"] == "with_low") & (lead["kind"] == "signal")].set_index("name")
    order = ["t_steps_rise", "t_hr_rise", "t_glucose_fall", "t_pump_cut", "t_first_low"]
    names = {"t_steps_rise": "steps rise (>= 200 in 5 min)", "t_hr_rise": "heart rate up 30 bpm",
             "t_glucose_fall": "glucose falling (<= -0.5 per 15 min)", "t_pump_cut": "pump cuts delivery",
             "t_first_low": "first reading < 4.0 mmol/L"}
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(order))[::-1]
    for yi, sig in zip(y, order):
        if sig not in L.index:
            continue
        r = L.loc[sig]
        ax.plot([r["q1"], r["q3"]], [yi, yi], color="#7f8c8d", lw=6, solid_capstyle="butt", alpha=0.5)
        ax.plot(r["median"], yi, "o", color="#c0392b", ms=8)
        # text above the bar, starting at Q1, on a white patch so the zero line
        # never runs through it
        ax.annotate(f"median {r['median']:.0f} min (Q1 {r['q1']:.0f}, Q3 {r['q3']:.0f}; n={int(r['n'])})",
                    (r["q1"], yi), textcoords="offset points", xytext=(0, 9), ha="left", va="bottom",
                    fontsize=8, zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([names[s] for s in order])
    ax.set_xlabel("minutes relative to the session start (negative = before)")
    ax.set_title("When each signal first appeared, sessions that included a low\n(median dot, Q1-Q3 bar)", loc="left")
    ax.set_xlim(-45, max(200, float(L["q3"].max()) + 110))
    ax.set_ylim(-0.6, len(order) - 0.2)   # headroom for the annotation above the top row
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_forecaster(F, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    p, y = F["p_low30"].to_numpy(float), F["ylow"].to_numpy()
    bins = [0, .05, .1, .2, .4, .6, .8, 1.01]
    xs, ys, ns = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.plot(xs, ys, "o-", color="#2c3e50")
    for x_, y_, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x_, y_), textcoords="offset points", xytext=(4, -12), fontsize=7.5)
    ax.set_xlabel("predicted probability of a reading < 4.0 in the next 30 min")
    ax.set_ylabel("observed share")
    ax.set_title("Low alert reliability, retired test set")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    yt = F["actual"].to_numpy(float)
    inside = (yt >= F["band_lo"].to_numpy(float)) & (yt <= F["band_hi"].to_numpy(float))
    edges = [2, 4, 6, 8, 10, 14, 23]
    labels, cov, lo, hi, ns = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (yt >= a) & (yt < b)
        if m.sum():
            k = int(inside[m].sum()); n = int(m.sum())
            w = wilson(k, n)
            labels.append(f"{a}-{b}"); cov.append(k / n); lo.append(w[0]); hi.append(w[1]); ns.append(n)
    cov, lo, hi = np.array(cov), np.array(lo), np.array(hi)
    x = np.arange(len(labels))
    ax.bar(x, cov, color="#95a5a6")
    ax.errorbar(x, cov, yerr=[cov - lo, hi - cov], fmt="none", ecolor="#2c3e50", capsize=4)
    ax.axhline(0.8, color="#c0392b", ls="--", lw=1, label="declared 80%")
    for xi, c, n in zip(x, cov, ns):
        ax.annotate(f"n={n}", (xi, 0.02), ha="center", fontsize=7.5, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("actual glucose 30 min ahead (mmol/L)")
    ax.set_ylabel("share of readings inside the band")
    ax.set_title(f"Band coverage by glucose level (overall {inside.mean() * 100:.1f}%)")
    ax.set_ylim(0, 1); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_alert_lead(E, out):
    leads = E["lead_min"].to_numpy(float)
    got = leads[~np.isnan(leads)]
    n = len(leads)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    top = max(65, (np.nanmax(leads) + 10) if len(got) else 65)
    edges = np.arange(0, top + 1, 5)
    ax.hist(got, bins=edges, color="#2c3e50", edgecolor="white")
    ticks = list(range(0, int(top) + 1, 10))
    labels = [str(t) for t in ticks]
    if n - len(got):
        xm = top + 10
        ax.bar([xm], [n - len(got)], width=4, color="#c0392b")
        ticks.append(xm); labels.append("missed")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    if len(got):
        med = np.median(got)
        ax.axvline(med, color="#c0392b", ls="--", lw=1)
        ax.annotate(f"median {med:.0f} min", (med, ax.get_ylim()[1] * 0.9), xytext=(5, 0), textcoords="offset points",
                    color="#c0392b", fontsize=9)
    ax.set_xlabel("minutes between the first alert and the first reading < 4.0 mmol/L (60-min look-back)")
    ax.set_ylabel("low episodes")
    ax.set_title(f"Alert lead before each low, retired test set\n(n = {n} episodes; "
                 f"{int((got >= 20).sum())} with >= 20 min, {int((got >= 30).sum())} with >= 30 min, "
                 f"{n - len(got)} missed)", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=str(RESULTS / "sessions_public.csv"))
    ap.add_argument("--controls", default=str(RESULTS / "controls_public.csv"))
    ap.add_argument("--odds", default=str(RESULTS / "odds_ratios_public.csv"))
    ap.add_argument("--lead", default=str(RESULTS / "lead_time_summary.csv"))
    ap.add_argument("--alert", default=str(RESULTS / "alert_lead_episodes.csv"))
    ap.add_argument("--test-outputs", default=str(PROCESSED_DIR / "final_test_outputs.csv"))
    args = ap.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    P = __import__("pathlib").Path

    if P(args.sessions).exists() and P(args.controls).exists():
        sess = pd.read_csv(args.sessions)
        ctrl = pd.read_csv(args.controls)
        if "usable" in sess.columns:
            sess = sess[sess["usable"] == 1]
        fig_meal_band(sess, ctrl, FIGURES_DIR / "low_rate_by_meal_band.png")
        print("wrote", FIGURES_DIR / "low_rate_by_meal_band.png")
    if P(args.odds).exists():
        fig_odds(pd.read_csv(args.odds), FIGURES_DIR / "odds_ratios.png")
        print("wrote", FIGURES_DIR / "odds_ratios.png")
    if P(args.lead).exists():
        fig_lead_timeline(pd.read_csv(args.lead), FIGURES_DIR / "lead_time_timeline.png")
        print("wrote", FIGURES_DIR / "lead_time_timeline.png")
    if P(args.test_outputs).exists():
        F = pd.read_csv(args.test_outputs, parse_dates=["t"]).sort_values("t")
        A = pd.read_csv(PROCESSED_DIR / "aligned_5min.csv", parse_dates=["t"], low_memory=False,
                        usecols=["t", "glucose"]).set_index("t")["glucose"]
        fut = pd.concat({k: A.shift(-k) for k in range(1, 7)}, axis=1).min(axis=1)
        F["ylow"] = (fut.reindex(F["t"]).to_numpy() < 4).astype(int)
        fig_forecaster(F, FIGURES_DIR / "forecaster_calibration.png")
        print("wrote", FIGURES_DIR / "forecaster_calibration.png")
    else:
        print("no final_test_outputs.csv: figure (d) skipped")
    if P(args.alert).exists():
        fig_alert_lead(pd.read_csv(args.alert), FIGURES_DIR / "alert_lead.png")
        print("wrote", FIGURES_DIR / "alert_lead.png")


if __name__ == "__main__":
    main()
