"""Hand-written documentation of the shinobi events, rendered into the dataset README.

The reference itself is ``reference.md``, next to this file.
"""

import os.path as op

#: Why a vocabulary entry produces no row in the dataset. Entries absent from here that
#: have zero rows are reported as "never occurred". Keep this honest: "not implemented"
#: means the pipeline has no detector for it.
NOT_EMITTED = {
    "PlayerState/HitRecovery": "requires the `hit_timer` variable, present only in "
                               "sidecars replayed with the current `data.json`",
    "WeaponPowerupExpired/{powerup_type}": "not implemented (no RAM signal for the "
                                           "weapon upgrade)",
    "Action/Other": "never pressed",
    "Action/Start": "never pressed",
    "Action/Select": "never pressed",
}


#: How each event is read from the RAM and how it was validated, in Markdown. Embedded in
#: the README and in the common event reference.
with open(op.join(op.dirname(__file__), "reference.md")) as _handle:
    REFERENCE = _handle.read()
