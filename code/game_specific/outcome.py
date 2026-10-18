"""How a repetition ended.

The labels, shared by the CNeuroMod videogame datasets:

    cleared
    failed/killed, failed/fall, failed/timeout
    incomplete/warp, incomplete/interrupted
    unknown

**shinobi.** Unchanged in substance: with no RAM map there is still no signal beyond
"did the player lose a life", which is what the shipped code used.
"""

from __future__ import annotations

def shinobi_outcome(repvars: dict) -> str:
    """Outcome for Shinobi III.

    Unchanged from the shipped behaviour: with no RAM map for this game the only
    available signal is whether a life was lost. Note this makes ``cleared`` a statement
    about survival, not about reaching the end of the stage.
    """
    try:
        lives = repvars["lives"]
        lost = any(lives[i] < lives[i - 1] for i in range(1, len(lives)))
        return "failed" if lost else "cleared"
    except (KeyError, IndexError, TypeError):
        return "unknown"




outcome = shinobi_outcome
