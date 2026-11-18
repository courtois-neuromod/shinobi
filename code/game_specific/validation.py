"""RAM invariants specific to Shinobi III, run on every repetition after the shared ones.

Like the shared checks, each one recomputes its quantity straight from
``_variables.json`` rather than calling the generator.
"""

from __future__ import annotations

import collections

from .generator import DEATH_CAUSES


def _diff_decreases(series) -> int:
    return sum(1 for i in range(1, len(series)) if series[i] < series[i - 1])


def check_repetition(events, repvars, summary, path, report) -> None:
    """Deaths against the death states and the lives counter, and post-hit recovery
    against the hits."""
    tt = events["trial_type"].astype(str)
    counts = collections.Counter(tt)
    deaths = sum(n for t, n in counts.items()
                 if t == "PlayerDied" or t.startswith("PlayerDied/"))
    if "status" in repvars:
        # Every life loss is preceded by a death state; a death state may have no life
        # loss when the recording stops during the death animation.
        status = repvars["status"]
        expected = sum(1 for i, v in enumerate(status)
                       if v in DEATH_CAUSES and (i == 0 or status[i - 1] != v))
        if expected != deaths:
            report.add("V1", "death count disagrees with the status death states", path,
                       f"status says {expected}, events say {deaths}")
        if "lives" in repvars and _diff_decreases(repvars["lives"]) > deaths:
            report.add("V1", "more lives lost than deaths", path,
                       f"lives drops {_diff_decreases(repvars['lives'])} times, "
                       f"{deaths} PlayerDied events")

    recovery = events[tt == "PlayerState/HitRecovery"]
    hits = events[tt == "PlayerDamaged"]["frame_start"].astype(int).tolist()
    if len(recovery) > len(hits):
        report.add("V1", "more HitRecovery rows than PlayerDamaged", path,
                   f"{len(recovery)} recoveries, {len(hits)} hits")
    stray = sum(1 for f in recovery["frame_start"].astype(int)
                if not any(abs(f - h) <= 3 for h in hits))
    if stray:
        report.add("V1", "HitRecovery not anchored on a PlayerDamaged", path,
                   f"{stray} row(s) start more than 3 frames from any hit")
