"""What :mod:`videogames_utils.events` needs to know about
Shinobi III: Return of the Ninja Master (Genesis).

See :class:`videogames_utils.events.game.Game`.
"""

from videogames_utils.events.game import Game

from . import docs, generator, outcome, ram, validation
from .vocabulary import VOCABULARY


def level_from_events(level: str) -> str:
    """The events.tsv level in the annotated form: ``4-1`` -> ``4``."""
    return level.split("-")[0]


GAME = Game(
    task="shinobi",
    name="Shinobi III: Return of the Ninja Master (Genesis)",
    game_id="ShinobiIIIReturnOfTheNinjaMaster-Genesis",
    frame_rate=generator.FRAME_RATE,
    vocabulary=VOCABULARY,
    generate=generator.generate,
    outcome=outcome.outcome,
    level_from_events=level_from_events,
    form_states=frozenset({"PlayerState/Normal"}),
    form_var=None,
    state_timers=(),
    coin_var=None,
    check_repetition=validation.check_repetition,
    # Completion is read from the end-of-level fade, which the recording sometimes stops
    # short of (17 of 536 cleared repetitions), so a miss is only a warning.
    completion_miss=("warning", "recording stopped before the end-of-level fade"),
    notes=docs.NOTES,
    not_emitted=docs.NOT_EMITTED,
    golden_sessions=[("sub-01", "ses-002")],
    ram_candidates=ram.SHINOBI_CANDIDATES,
)
