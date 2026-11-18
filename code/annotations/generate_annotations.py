#!/usr/bin/env python
"""Generate annotated BIDS event files for the shinobi dataset.

The pipeline is ``videogames_utils.events``, shared by the CNeuroMod videogame datasets;
what is specific to this game lives in ``code/game_specific/``.

Usage, from the dataset root:
    python code/annotations/generate_annotations.py -d . --overwrite --validate
    python code/annotations/generate_annotations.py -d . -sub sub-01 -o /tmp/out

Requires the replay sidecars (``gamelogs/*_variables.json``); run
``code/replays/generate_replays.py`` first if they are missing.
"""

import os.path as op
import sys

from videogames_utils.events import run
from videogames_utils.events.game import load_game

DATASET = op.join(op.dirname(op.abspath(__file__)), "..", "..")

if __name__ == "__main__":
    sys.exit(run.main(game=load_game(DATASET)))
