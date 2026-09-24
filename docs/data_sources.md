# Data sources

Everything below describes the author's own data. None of it is in this repository: the raw exports, the 5-minute table built from them and the event log stay private. The code in `src/` expects the raw files in the layout at the end of this page; the synthetic sample in `data/synthetic/` stands in for them so the pipeline can be run without any of it.

## The four streams

**1. Continuous glucose monitor (Dexcom G6, then G7).** Two exports carry the same sensor's readings:

- the *Glooko export*, a zip of CSV files produced by the pump vendor's web platform for a chosen date range. Its CGM file has one row per reading at minute precision (local wall time) in mmol/L. Readings below the sensor floor appear as a sentinel value and are mapped to 2.2 mmol/L with a `below_range` flag;
- the *Clarity export*, a single CSV from Dexcom's own platform, with readings to the second, the rate of change, and the device's alert settings. Below-range readings are written as the word "Low" and are mapped the same way.

The two disagree on well under 1 in 20 minutes and almost always by 0.1 mmol/L or less. The one systematic disagreement fell on the UK clock-change day, which is flagged `dst_day` and excluded from the clean set. Each pipeline has gaps the other fills, so the glucose backbone is the union of the two (Glooko preferred where both exist). Stretches where both are empty are listed in the private overrides file as `real_gap` rows.

**2. Pump insulin and carbohydrates (Omnipod 5, automated mode).** The same Glooko export holds one row per bolus (units delivered, the carbohydrate entry attached to it, the ratio used), daily totals of basal and bolus insulin, the scheduled basal profile and pump alarms. Announced carbohydrates are the `Carbs input` on bolus rows; unannounced food is an unobserved input.

The export does **not** carry the automated basal rate the pump chooses every five minutes. What can be pulled from the vendor's web platform, through the author's own signed-in browser session, is the pump's *delivery-state timeline* at one-second resolution: whether delivery was paused by the algorithm, running in automated mode, or at the maximum rate; and the pump's *mode events* (Activity mode, which raises the target and reduces delivery; Limited mode, running without CGM input; Manual mode). Those two pulls are the `basal_states_*.json` and `modes_*.json` files. `src/pull_pump_timeline.py` documents what they contain; the endpoint and the patient identifier used for the pull are not published. `build_dataset.py` turns the timelines into the fraction of every 5-minute interval spent in each state or mode. The automated delivery rate is then estimated per day so that it reproduces the day's basal total from the daily-totals file; suspends (0 U/h) and maximum-rate stretches are exact, the modulation while delivering is smoothed to the day's average.

**3. Apple Watch through Apple Health.** The *Health export* (a large XML file) holds workouts (type, start, end), heart-rate samples (sparse in the background, dense during workouts) and step counts. Every timestamp in the export carries the device's offset at export time, so it is converted to UTC and then to Europe/London wall time rather than read as printed. Workout labels are what the watch was told: the football type was also used for cricket, and a fixed weekday-and-clock rule held in the private settings tells the two apart. A few short "Other" entries and one "Cycling" entry were corrected from memory through the overrides file.

**4. The event log.** A separate hand-kept log of day-to-day context exists but is not used by any script here and is not published in any form.

## How the exports were obtained

- Glooko CSV export: the platform's own export function, one zip per date range.
- Clarity CSV: the platform's export function, at most 90 days per file.
- Pump timelines: pulled from the vendor's web platform through the author's signed-in browser tab, one JSON per data window, without storing or scripting any credential. The endpoint, its parameters and the patient identifier are deliberately absent from this repository.
- Apple Health: the Health app's "Export All Health Data".

## The 5-minute grid

`build_dataset.py` aligns all four streams on one row per 5-minute interval per data window. Columns: glucose (union, with source and below-range flag), bolus units and carbohydrate grams landing in the interval, linear-decay bolus insulin on board, the pump state fractions, estimated basal delivery and its insulin on board, mode fractions, mean heart rate and sample count, steps, minutes of workout and its category, hours since the last workout ended, hour and weekday, sensor-generation flag, the quality flags (`dst_day`, `real_gap`, `pod_change`, `missing_glucose`), the 30-minute-ahead target, `past_hour_complete`, `clean`, the fixed `split` and the exercise-day category. The synthetic sample carries the same columns plus `synthetic = 1`.

Data windows are contiguous blocks defined by the exports; nothing is interpolated across the gap between blocks, and a later window never sees insulin, workouts or past-hour rows from before a fence. The last window is quarantined: written to the table, never modelled, never printed.

## Raw layout expected by `build_dataset.py`

```
<RAW_DIR>/
  glooko/*.zip                 Glooko CSV exports, one per window
  clarity/*.csv                Clarity exports
  glooko_api/basal_states_*.json
  glooko_api/modes_*.json      pump timeline pulls
  health/*.zip                 Health export (export.xml inside)
```

Set `T1D_DATA_DIR` (or `--data-dir`) to a private folder holding `raw/`, `processed/`, a `settings.json` (copy `data/settings_example.json` and fill in the real window boundaries, split dates and relabel rule) and an `overrides.csv` (same schema as `data/overrides_example.csv`).
