"""The ``trial_type`` vocabulary of Shinobi III: Return of the Ninja Master (Genesis).

Every event this game may emit.
"""

from videogames_utils.events.vocabulary import EventType

VOCABULARY = [
    EventType("PlayerDamaged",
              "The player is hit and loses health.",
              former=("Hit/powerup_lost", "HealthLoss", "Player_damaged")),
    EventType("PlayerDied/Enemy",
              "The player dies after their health is drained by enemy hits: the hit that "
              "empties the health bar.",
              former=("PlayerDied", "Player_died")),
    EventType("PlayerDied/Fall",
              "The player falls into a pit or into water and dies. `onset` is the frame "
              "the player reaches the bottom of the screen and `duration` runs to the "
              "frame the life is actually lost, 2-6 s later, once the death animation "
              "and the fade have played (or to the end of the recording when it stops "
              "first).",
              former=("PlayerDied", "Player_died"),
              durational=True),
    EventType("LifeGained",
              "The player collects or earns an extra life.",
              former=("Life_gained",)),
    EventType("HealthGained",
              "The player collects a health item and the health bar increases.",
              former=("HealthGain", "Health_gained")),
    EventType("PlayerState/Normal",
              "The player is in the game's single form. Joe Musashi has no power-up "
              "forms, so this row covers every frame on which the player is alive in "
              "the level (the temporary weapon upgrade is not tracked; see the "
              "README). Present so that every frame of play has a form row.",
              former=("Player_state/Normal",),
              durational=True),
    EventType("PlayerState/HitRecovery",
              "Post-hit recovery: the game's post-hit counter, during which the player "
              "flashes.",
              former=("Player_state/Hit_recovery",),
              durational=True),
    EventType("ItemCollected/{item_type}",
              "The player collects a health item, weapon upgrade, extra life or other "
              "collectible.",
              former=("Item_collected/{item_type}",)),
    EventType("EnemyDefeated",
              "The player defeats an enemy. There is no known RAM map for enemy types, "
              "so this event is untyped and is inferred from score increments.",
              former=("Kill", "Enemy_defeated")),
    EventType("ProjectileAppeared/Shuriken",
              "The player throws a shuriken. A point event: there is no known RAM map "
              "for object positions, so the projectile cannot be tracked on screen.",
              former=("Projectile_appeared/Shuriken",)),
    EventType("WeaponPowerupStarted/{powerup_type}",
              "The player receives a temporary or persistent weapon upgrade.",
              former=("Weapon_powerup_started/{powerup_type}",),
              durational=True),
    EventType("WeaponPowerupExpired/{powerup_type}",
              "A temporary weapon upgrade ends or is lost.",
              former=("Weapon_powerup_expired/{powerup_type}",)),
    EventType("LevelStarted",
              "A new level or gameplay attempt begins.",
              former=("Level_started",)),
    EventType("LevelCompleted",
              "The player successfully finishes the level: the start of the "
              "end-of-level fade that closes the recording. It is missed in the few "
              "cleared repetitions whose recording stopped before the fade.",
              former=("Level_complete", "Level_completed")),
    EventType("Action/Left",
              "The player holds the left direction.",
              former=("LEFT",),
              durational=True),
    EventType("Action/Right",
              "The player holds the right direction.",
              former=("RIGHT",),
              durational=True),
    EventType("Action/Up",
              "The player holds the up direction.",
              former=("UP",),
              durational=True),
    EventType("Action/Down",
              "The player holds the down direction (duck / crouch).",
              former=("DOWN",),
              durational=True),
    EventType("Action/Jump",
              "The player presses the jump button.",
              former=("JUMP",),
              durational=True),
    EventType("Action/Attack",
              "The player presses the attack button.",
              former=("HIT",),
              durational=True),
    EventType("Action/Ninjutsu",
              "The player presses the ninjutsu button.",
              former=("NINJUTSU",),
              durational=True),
    EventType("Action/Other",
              "The player presses a button with no documented function in this game "
              "(X, Y or Z on the six-button pad). The raw button is in the `button` "
              "column.",
              durational=True),
    EventType("Action/Start",
              "The player presses START (pauses the game).",
              former=("START",),
              durational=True),
    EventType("Action/Select",
              "The player presses MODE.",
              former=("SELECT", "MODE"),
              durational=True),
    EventType("Screen/Gameplay",
              "The player controls the character in the level. The Screen/* rows of a "
              "repetition partition it: exactly one is active at any frame, and this "
              "is the complement of all the others.",
              durational=True),
    EventType("Screen/Death",
              "The death sequence: from the frame the player dies, or drops off the "
              "bottom of the screen, to the end of the death animation and the freeze "
              "that follows it. The player has no control.",
              durational=True),
    EventType("Screen/LevelEnd",
              "The end-of-level sequence after the level is completed: the closing "
              "fade, with the ROUND CLEAR bonus tally on level 1. The player has no "
              "control.",
              former=("Screen/Level_end",),
              durational=True),
    EventType("Screen/Transition",
              "A transition inside a level with no player control: the section fades, "
              "and the fade back into play after a death.",
              durational=True),
    EventType("Screen/GameOver",
              "The GAME OVER / CONTINUE screen after the last life is lost. Recordings "
              "were meant to stop at the death, so this only appears where one ran on.",
              former=("Screen/Game_over",),
              durational=True),
    EventType("Repetition",
              "One repetition of gameplay (one .bk2 file). This is the container row "
              "that carries `stim_file`; all other events fall inside its window.",
              former=("gym-retro_game",),
              durational=True),
]
