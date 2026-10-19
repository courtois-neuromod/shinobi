"""How a repetition ended.

The labels, shared by the CNeuroMod videogame datasets:

    cleared
    failed/killed, failed/fall, failed/timeout
    incomplete/warp, incomplete/interrupted
    unknown

**shinobi.** The shipped logic only asked "did the player lose a life" and wrote
``failed`` with no cause, and it missed deaths whose recording stopped before ``lives``
dropped. The player's animation state (``status``, $FF415A) enters a dedicated value on
the frame of every death, 41 for health drained by enemies and 43 for a fall, so the
cause of the final death is now stated as for the Mario games.
"""

from __future__ import annotations

from .generator import DEATH_CAUSES


def shinobi_outcome(repvars: dict) -> str:
    """Outcome for Shinobi III.

    ``failed/killed`` or ``failed/fall`` from the death state ``status`` entered last
    (see :data:`.generator.DEATH_CAUSES`), ``cleared`` if it never entered one. As
    before, ``cleared`` is a statement about survival, not about reaching the end of the
    stage.
    """
    try:
        status = repvars["status"]
        causes = [DEATH_CAUSES[v] for i, v in enumerate(status)
                  if v in DEATH_CAUSES and (i == 0 or status[i - 1] != v)]
        if not causes:
            return "cleared"
        return "failed/killed" if causes[-1] == "Enemy" else "failed/fall"
    except (KeyError, IndexError, TypeError):
        return "unknown"


outcome = shinobi_outcome
