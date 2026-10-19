"""The Shinobi III parts of the replay pipeline (:mod:`videogames_utils.replays`): the
game's own ``_summary.json`` fields, the scenario of each level and the variables' correction.

The counting functions are those of the former ``code/replays/generate_replays.py``.
"""

from __future__ import annotations

import numpy as np


def fix_position_resets(X_player):
    """Sometimes X_player resets to 0 but the player's position should keep
    increasing.
    This fixes it and makes sure that X_player is continuous. If not, the
    values after the jump are corrected.

    Parameters
    ----------
    X_player : list
        List of raw positions at each timeframe from one repetition.

    Returns
    -------
    list
        List of lists of fixed (continuous) positions. One per repetition.

    """

    fixed_X_player = []
    raw_X_player = X_player
    fix = 0  # keeps trace of the shift
    fixed_X_player.append(raw_X_player[0])  # add first frame
    for i in range(1, len(raw_X_player) - 1):  # ignore first and last frames
        if raw_X_player[i - 1] - raw_X_player[i] > 100:
            fix += raw_X_player[i - 1] - raw_X_player[i]
        fixed_X_player.append(raw_X_player[i] + fix)
    fixed_X_player.append(fixed_X_player[-1]) # re-add the last frame for consistency
    return fixed_X_player


# Shinobi has no scenario.json: each level was played with its own done condition.
# Keys are the `level` column of the events files.
SCENARIOS = {
    "1-0": "scenario_Level1.json",
    "4-1": "scenario_Level4-1.json",
    "5-0": "scenario_Level5-0.json",
}


def scenario(level: str):
    """The scenario file a level is replayed with."""
    return SCENARIOS[level]


def process_variables(repvars: dict) -> dict:
    """Fix position resets"""
    repvars["X_player"] = fix_position_resets(repvars["X_player"])
    return repvars


def _unit_decreases(series) -> int:
    """Frames on which a counter drops by exactly 1."""
    diff = np.diff(series, n=1)
    return int(np.sum(diff == -1))


def summary(repvars: dict, events) -> dict:
    """The game's fields of a repetition's ``_summary.json``."""
    return {
        "ScoreGained": repvars["score"][-1] - repvars["score"][0],
        "HealthLost": _unit_decreases(repvars["health"]),
        "ShurikensThrown": _unit_decreases(repvars["shurikens"]),
        "EnemiesDefeated": int((events["trial_type"] == "EnemyDefeated").sum())
        if len(events) else 0,
    }
