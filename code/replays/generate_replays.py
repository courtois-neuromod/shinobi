#!/usr/bin/env python
"""Generate the replay outputs of the shinobi dataset from its .bk2 files.

The pipeline is ``videogames_utils.replays``, shared by the CNeuroMod videogame
datasets; what is specific to this game lives in ``code/game_specific/``.

Usage, from the dataset root:
    python code/replays/generate_replays.py -nj -1
    python code/replays/generate_replays.py -sub sub-01 -ses ses-001
"""

import os.path as op
import sys

from videogames_utils import replays
from videogames_utils.events.game import load_game

DATASET = op.join(op.dirname(op.abspath(__file__)), "..", "..")

if __name__ == "__main__":
    sys.exit(replays.main(game=load_game(DATASET)))
