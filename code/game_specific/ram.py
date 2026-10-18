"""RAM variables of Shinobi III (Genesis) verified before they were added to
``stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis/data.json``.
"""

from __future__ import annotations

from typing import List

from videogames_utils.events.ram import Candidate


# --- Shinobi III (Genesis) ---------------------------------------------------------
# There is no public RAM map. This byte was found by recording all 64 KB of work RAM
# over a replay and keeping bytes that are zero before every health loss and start a
# countdown on the frame of it. It is set to 80 (or 64, depending on the attack) when
# health drops and counts down by one per frame, which is the period during which the
# player flashes. The same byte also runs from 48 after events that cost no health
# (knock-backs) and sits at 1 for ~25 frames at other moments, so the generator only
# uses runs that begin on a health loss.
#
# The Genesis core exposes 68000 work RAM word-swapped: the counter observed at raw
# offset $FF4164 of the memory block reads correctly when declared at $FF4165, which is
# how the emulator's variable reader was verified to return it (80 -> 1).

SHINOBI_CANDIDATES: List[Candidate] = [
    Candidate("hit_timer", 0xFF4165, "|u1", "(no public symbol)",
              "post-hit recovery countdown, 80 or 64 -> 0 over 81-103 frames"),
]
