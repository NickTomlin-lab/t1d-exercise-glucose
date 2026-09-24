"""
pull_pump_timeline.py - documented STUB for the pump-timeline pulls.

The two JSON inputs that build_dataset.py reads from RAW_DIR/glooko_api/
(basal_states_*.json and modes_*.json) were fetched from the pump vendor's
web platform through a signed-in browser session, one file per data window.
The original pull was done interactively (browser JavaScript in the person's
own logged-in tab; no credentials were ever stored or scripted) and carried
the platform hostname, an API path and the person's patient identifier, none
of which belong in a public repository. This module keeps the function
signatures and describes what each pull returned; the bodies raise
NotImplementedError. See docs/data_sources.md for the general description.
"""


def pull_basal_states(start_date, end_date):
    """Fetch the pump's delivery-state timeline for [start_date, end_date].

    What it returned: a JSON object {"series": {...}, "devices": [...]} whose
    series hold three step functions at one-second resolution, each a list of
    {"x": <unix seconds, local wall time>, "y": 0 or 1}:
      basalBarAutomatedSuspend   delivery paused by the algorithm (0 U/h)
      basalBarAutomated          automated delivery (rate not exposed)
      basalBarAutomatedMax       delivery at the maximum rate
    build_dataset.py turns the 0/1 edges into segments and computes the
    fraction of every 5-minute interval spent in each state.
    """
    raise NotImplementedError("see docs/data_sources.md")


def pull_mode_events(start_date, end_date):
    """Fetch the pump's mode events for [start_date, end_date].

    What it returned: the same envelope, with series of events carrying
    "timestamp" and "endTimestamp" (ISO strings ending in Z but in local
    wall time) for the modes Activity (the vendor's internal name is a
    "hypoprotect" mode: raised target, reduced delivery), Limited (running
    without CGM input) and Manual. build_dataset.py computes frac_activity,
    frac_limited and frac_manual per 5-minute interval from them.
    """
    raise NotImplementedError("see docs/data_sources.md")


if __name__ == "__main__":
    print(__doc__)
