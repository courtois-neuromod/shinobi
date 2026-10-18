"""RAM invariants specific to Shinobi III, run on every repetition after the shared ones.

Like the shared checks, each one recomputes its quantity straight from
``_variables.json`` rather than calling the generator.
"""

from __future__ import annotations

import collections


def _diff_decreases(series) -> int:
    return sum(1 for i in range(1, len(series)) if series[i] < series[i - 1])


def check_repetition(events, repvars, summary, path, report) -> None:
    """Deaths against the lives counter, and post-hit recovery against the hits."""
    tt = events["trial_type"].astype(str)
    counts = collections.Counter(tt)
    deaths = sum(n for t, n in counts.items()
                 if t == "PlayerDied" or t.startswith("PlayerDied/"))
    if "lives" in repvars:
        expected = _diff_decreases(repvars["lives"])
        if expected != deaths:
            report.add("V1", "death count disagrees with the lives counter", path,
                       f"counter says {expected}, events say {deaths}")

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
