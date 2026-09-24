# Exercise-related hypoglycaemia on a hybrid closed loop

The author's own type 1 diabetes data (Omnipod 5 pump, Dexcom G6 then G7 sensor, Apple Watch), December 2025 to July 2026, used to ask three questions: how much an exercise session raises the odds of a low compared with the same clock time on a rest day, how long before the pump reacts the body has already signalled exercise, and how much warning a 30-minute low alert built on the same data can give. The data are the author's, the code was written and run with AI tools from prompts the author wrote, and every number here is checked against the printed reports in `results/reports/`.

## Safety statement

This is analysis only. Nothing in this repository informs a dosing decision, nothing here is a medical device, and nothing here is advice. The pump's own forecast sits inside a regulated safety layer; this one does not. Do not act on any output of this code.

## The problem

On a hybrid closed loop the pump adjusts basal insulin every five minutes from the sensor trace, so it can only react once glucose is already moving. Exercise pulls glucose down faster than the pump can withdraw insulin, and insulin already delivered keeps acting for hours. The pump accounts for that insulin with a straight two-hour decay, so a meal bolus given three hours before a session counts as zero on the pump's screen while, physiologically, some of it is still working. The question underneath all three analyses is whether the person, the watch or the recent insulin total knew something the pump did not, and how early.

## Data

Four streams, aligned on one 5-minute grid: sensor glucose (the union of two exports of the same sensor), pump boluses and announced carbohydrates, the pump's delivery-state and mode timelines (paused, automated, maximum; Activity mode), and watch workouts, heart rate and steps. Glucose is in mmol/L throughout. Details and the export routes are in `docs/data_sources.md`.

Published here: the session table and its matched control windows with random identifiers and no dates or times (`results/sessions_public.csv`, `results/controls_public.csv`, `results/pooled_windows_public.csv`), sanitised copies of the printed reports (`results/reports/`), five figures (`results/figures/`), the rule-by-horizon and lead-time summary tables, and a synthetic 14-day sample (`data/synthetic/`).

Not published: the raw exports, the 5-minute table, the event log, any date or clock time, anything recorded after July 2026 (a quarantined block that no script has read for a result), and the private settings file that holds the real window boundaries and the author's own workout-relabel rule. Exact reproduction of the 5-minute forecaster numbers needs the private data; the code and the published session table reproduce the session-level results exactly (`python src/fit_session_models.py --sessions results/sessions_public.csv --controls results/controls_public.csv`).

**The synthetic sample is synthetic.** `data/synthetic/aligned_5min_synthetic.csv` is drawn from aggregate statistics of the real training rows (`data/synthetic/summary_stats.json`) with workout start times drawn at random between 07:00 and 21:00, placeholder dates in the year 2000 and `synthetic = 1` on every row. It exists so the pipeline can be run; the published results come from the real data, not from it.

## Method

- **Time-based split, fixed once.** The 5-minute forecaster's test set is the last three weeks of the second data window; it was scored once and then retired. Session-level work uses everything up to the end of that window as an exploration set and treats the later block as untouched.
- **Baselines first.** For the forecaster: last value (glucose in 30 minutes = glucose now) and a straight-line trend through the last hour. For the session models: a constant rate (B0) and starting glucose plus its 30-minute slope (B1). Nothing is called good until it beats these on the same rows.
- **Coverage, not just a point.** Any model that outputs a band is judged by how often the truth fell inside it, overall and separately below 4 mmol/L.
- **Predictions logged before each result.** The author wrote down what he expected before each script ran (for example "the insulin curve will improve the classifier not at all"), and the result is reported against that.
- **Every comparison is paired.** Model differences are paired differences on the same rows with a bootstrap interval that resamples sessions (with their controls) or time blocks, never rows.
- **Fixed models, no tuning.** The session models are a fixed list of weighted L2 logistic regressions with C = 1, forward-chained by calendar month so that every prediction comes from an earlier month. The forecaster is LightGBM with fixed hyperparameters and early stopping on a validation window that precedes the test set.
- **Where the forecaster comes from.** The forecaster follows the approach of Cuya (2025), the winning entry of the BrisT1D blood glucose prediction competition: a single LightGBM model on lagged tabular features rather than a neural sequence model. It is adapted here to one person's data, a 30-minute horizon in mmol/L, and a band whose coverage is checked, following the argument of Martinsson et al. (2020) that a glucose forecast should report its uncertainty. No other architecture was tried; the choice was the simplest approach that had won on the closest public task.

## Results

All numbers are from `results/reports/`; n is given with each.

**The 30-minute forecaster** (`forecaster_verdict.txt`, 5,872 clean test rows over 21 days). Point RMSE 1.207 mmol/L against 1.534 for the last-value baseline and 2.259 for the linear trend. The 80% band covered 84.1% of readings overall but 65.2% of the 155 readings below 4 mmol/L and 60.6% of the 132 readings within four hours of exercise; it under-covers exactly where a low is coming. The low alert (any reading below 4 in the next 30 minutes, threshold 0.60) caught all 24 low episodes of the test period at 2.2 alert episodes a day, Brier 0.0228 against 0.0449 for a constant rate, with 75.6% of alert rows followed by a low.

**How much warning the alert gave** (`alert_lead_report.txt`, figure `alert_lead.png`). For the 24 low episodes, the first alert came a median 5 minutes before the first reading below 4 (quartiles 2 and 10); 2 of 24 episodes had 20 minutes or more of warning and the same 2 had 30 minutes or more; in 6 the alert came on the same row as the low, and 1 had no alert in the preceding hour. Context: the sensor's own predictive alert fires when it expects a reading below 3.1 mmol/L within 20 minutes, and its low alert fires at the user's threshold; this comparison uses a 4.0 mmol/L threshold and the model's 30-minute horizon, so it is not a like-for-like head-to-head, and the CGM's own alerts are not in the data.

**Exercise sessions against matched rest windows** (`controls_report_110sessions.txt`, figure `low_rate_by_meal_band.png`). On the 110 usable sessions, 40.0% included a low during the session or in the two hours after, against 32.7% in exercise-free windows at the same clock time on nearby days: ratio 1.22 (bootstrap 95% interval 0.92 to 1.59). The earlier two-window run on 89 sessions gave 1.10 (0.82 to 1.43) (`sessions_v2_controls_report.txt`). Football stands out (60% against 13%, ratio 4.50, 2.00 to 15.00, ten sessions); gym does not (ratio 0.99, 70 sessions). Sessions started two to four hours after a meal were the riskiest band (61%, 28 sessions), and that band was also the riskiest without exercise (41%, 39 control windows).

**After accounting for glucose and insulin** (`pooled_models_report.txt`, figure `odds_ratios.png`; 110 sessions and 329 controls, cluster-bootstrap intervals). With starting glucose, its 30-minute slope and the insulin delivered in the previous four hours in the model, an exercise session carried 2.32 times the odds of a low of a matched rest window (1.42 to 4.01). Each extra mmol/L of starting glucose lowered the odds (0.88, 0.77 to 0.98) and each extra unit of insulin in the previous four hours raised them (1.11, 1.06 to 1.18). The interaction is the clearest term: each unit of insulin in the previous four hours multiplied the exercise effect by 1.33 (1.14 to 1.61), so exercise on an empty insulin slate showed no excess (0.43 at zero units, 0.17 to 1.01) and 2.86 (1.65 to 5.96) at the median 6.6 units. The earlier 109-session run gave 2.12 (1.25 to 3.60) for exercise and 1.33 (1.15 to 1.61) for the interaction (`pooled_models_report_prompt05_109sessions.txt`); the shift comes from one admitted session and re-matched December controls, both in the training-only first fold.

**Four-hour insulin total against the pump's two-hour insulin on board.** In the earlier run the model with the four-hour total had a lower forward-chained Brier score than the model with the pump's two-hour pair (difference -0.009, interval -0.021 to +0.002). In the re-run on the published table the two are indistinguishable (+0.001, -0.010 to +0.013). Out of fold, no session model beat B1 by an interval clear of zero in either run; the odds ratios above are fitted once on all rows and are the stronger statement.

**Pump lead time** (`lead_time_report.txt`, figure `lead_time_timeline.png`; 44 sessions that included a low, minutes relative to the session start). Steps rose a median 20 minutes before the start, heart rate crossed baseline plus 30 bpm at the start, glucose began falling 5 minutes after it, the pump first cut delivery 27.5 minutes after it (42 sessions with a cut), and the first low came 97.5 minutes after it. The watch's movement signal led the pump's first cut by a median 50 minutes and its heart-rate signal by 25; the first low followed the pump's cut by a median 45 minutes, and in 9 of the 42 sessions the low came before the pump had cut at all. Sessions without a low show the same ordering with a quicker cut (median 20 minutes).

**Could a system have known a session was coming?** (`anticipation_report.txt`, 110 sessions, 29.6 weeks of candidate decision times). A routine rule (a session in the same weekday and half-hour slot in two of the previous four weeks) flagged 12% of sessions an hour ahead at 1.3 false-alarm episodes a week; it is essentially a detector of two fixed weekly fixtures. A movement rule (steps or heart rate) flagged 96% of sessions at any horizon, but only 25 to 40 minutes before the start, at about 50 false-alarm episodes a week, because it fires on any walk.

**The insulin curve, a null result** (`cv_classifier_v2_report.txt`; 38,694 out-of-fold rows, 261 low episodes). Replacing the pump's two-hour straight-line insulin on board with a five-hour exponential curve changed the low classifier's Brier score by +0.0001 (interval -0.0004 to +0.0006; baseline Brier 0.0342), and in the two-to-five-hour post-meal window by +0.0003 (-0.0007 to +0.0013). The logged prediction was "not at all". The one regime where a difference cleared zero was the four hours after exercise (-0.0012, -0.0027 to -0.0001), one of 24 cells, kept as a lead for a pre-registered check rather than a result.

## Limitations

- One person. Nothing here generalises beyond the author without being repeated on someone else's data.
- Football rests on ten sessions; its interval runs from 2 to 15.
- Controls are matched on clock time and nearness in date, not on what the author did. Activity mode, carbohydrate before a session and the decision to train at all are the author's own management, so the exercise effect is the effect of an exercise session as actually managed, confounded by that management.
- The 30-minute alert and the sensor's own alerts are not compared head to head: different threshold, different horizon, and the sensor's alerts are not in the data.
- No counterfactual. Nothing here shows that acting on an earlier signal would have prevented a low, only when each signal appeared.
- The pump's automated delivery rate is not exported; the basal insulin-on-board column is an estimate calibrated to daily totals, exact only during pauses and maximum-rate stretches.

## What is next

Two frozen models: v1 with the inputs used here and v2 with the logged inputs added, both trained on the same data up to 31 December 2026 and scored once on January to March 2027 against the same baselines, with the predictions written down before the scoring run.

Under the freeze plan the block recorded after July 2026 stops being quarantined on 31 December 2026, when it becomes training data for v1 and v2. This repository will be updated once after the scoring run in April 2027.

## How AI tools were used

The design and the prompts were written in a Claude chat. Each numbered prompt was pasted into a fresh Claude Code session, which wrote and ran the scripts and printed a report; every result quoted here was checked against those printed reports. The author set the questions, the rules (the fixed split, baselines first, nothing prospective printed), the predictions logged before each run, and every data decision (which watch entries were gym, which sessions were mis-timed, which gaps were real). This repository was assembled the same way from a prompt that specified the privacy scrub, the published columns and the checks.

## Academic context

This is a personal project alongside the MA in AI at the University of Southampton. The Module 1 proposal set the question (link placeholder: `docs/module1_proposal.pdf`, to be added by the author or not). The Module 4 report reviewed Martinsson et al. (2020), a glucose forecaster with a variance estimate, and Cuya (2025), the winning tabular model of the BrisT1D challenge; the forecaster here follows the second and checks its band the way the first argued for.

## Running the code

```
pip install -r requirements.txt
python src/run_all.py                 # synthetic sample: sessions -> controls -> models -> lead time -> anticipation
python src/run_all.py --forecaster    # also the 5-minute forecaster scripts
python src/make_figures.py            # figures from the published tables (figure d needs the private test outputs)
```

To run on your own data, point `T1D_DATA_DIR` (or `--data-dir`) at a private folder holding `raw/`, a `settings.json` (copy `data/settings_example.json`) and an `overrides.csv` (copy `data/overrides_example.csv`), run `src/build_dataset.py`, then `src/run_all.py --skip-synthetic`. `tools/scrub_check.py` greps a tree for anything that must not be published; `tools/sanitise_reports.py` strips dates, times and identifiers from a printed report.

## References

- Cuya, C. (2025). 1st place solution: single LightGBM model. BrisT1D Blood Glucose Prediction Competition, Kaggle.
- James, S. et al. (2025). The BrisT1D dataset. [complete from the author's reference index]
- Martinsson, J., Schliep, A., Eliasson, B. and Mogren, O. (2020). Blood glucose prediction with variance estimation using recurrent neural networks. Journal of Healthcare Informatics Research, 4, 1-18.

## Licence

MIT, copyright 2026 Nick Tomlin. See `LICENSE`.
