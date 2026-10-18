"""The ``trial_type`` vocabulary of Shinobi III: Return of the Ninja Master (Genesis).

Every event this game may emit, in the order of the README and the BIDS sidecar.
"""

from videogames_utils.events.vocabulary import EventType

VOCABULARY = [
    EventType("PlayerDamaged",
              "The player is hit and loses the current power-up state (Mario games) or "
              "health (Shinobi).",
              former=("Hit/powerup_lost", "HealthLoss", "Player_damaged")),
    EventType("PlayerDied",
              "The player loses all health or dies from an environmental hazard. "
              "`onset` is the frame health reaches zero and `duration` runs to the "
              "frame the life is actually lost, 2-6 s later, once the death animation "
              "and the fade have played.",
              former=("Player_died",),
              durational=True),
    EventType("LifeGained",
              "The player collects or earns an extra life.",
              former=("Life_gained",)),
    EventType("HealthGained",
              "The player collects a health item and the health bar increases.",
              former=("HealthGain", "Health_gained")),
    EventType("PlayerState/Normal",
              "The player is in Shinobi's single form. Joe Musashi has no power-up "
              "forms, so this row simply covers every frame on which the player is "
              "alive in the level (the temporary weapon upgrade is not tracked; see "
              "the README). Present for uniformity with the Mario datasets.",
              former=("Player_state/Normal",),
              durational=True),
    EventType("PlayerState/HitRecovery",
              "Post-hit recovery: the player has just been damaged and blinks. In the "
              "Mario games nothing can hurt the player until it ends; in Shinobi it is "
              "the game's post-hit counter.",
              former=("Player_state/Hit_recovery",),
              durational=True),
    EventType("ItemCollected/{item_type}",
              "The player collects a health item, weapon upgrade, extra life or other "
              "collectible.",
              former=("Item_collected/{item_type}",)),
    EventType("EnemyDefeated",
              "The player defeats an enemy. Shinobi has no RAM map for enemy types, so "
              "this event is untyped and is inferred from score increments.",
              former=("Kill", "Enemy_defeated")),
    EventType("ProjectileAppeared/Shuriken",
              "The player throws a shuriken. A point event: Shinobi has no RAM map for "
              "object positions, so the projectile cannot be tracked on screen.",
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
              "The player successfully finishes the level. In Shinobi this is the "
              "start of the end-of-level fade that closes the recording; it is missed "
              "in the few cleared repetitions whose recording stopped before the fade.",
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
              "The player presses the attack button (Shinobi).",
              former=("HIT",),
              durational=True),
    EventType("Action/Ninjutsu",
              "The player presses the ninjutsu button (Shinobi).",
              former=("NINJUTSU",),
              durational=True),
    EventType("Action/Other",
              "The player presses a button with no documented function in this game "
              "(e.g. L/R on the SNES pad in Super Mario All-Stars). The raw button is "
              "in the `button` column.",
              durational=True),
    EventType("Action/Start",
              "The player presses START (pauses the game).",
              former=("START",),
              durational=True),
    EventType("Action/Select",
              "The player presses SELECT / MODE.",
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
              "The end-of-level sequence after the level is completed: the flagpole "
              "slide, walk into the castle and time bonus (SMB1); the goal card and "
              "COURSE CLEAR screen (mario3); the ROUND CLEAR bonus tally (shinobi). "
              "The player has no control.",
              former=("Screen/Level_end",),
              durational=True),
    EventType("Screen/Transition",
              "A transition inside a level with no player control: pipe, vine or door "
              "travel and the black or fading screen while the next area loads (SMB1 "
              "pipes, vines and the exit from a bonus area; mario3 pipes and doors; "
              "shinobi section fades), and the fade back into play after a shinobi "
              "death.",
              durational=True),
    EventType("Screen/GameOver",
              "The GAME OVER screen after the last life is lost: mario3's world map "
              "with the GAME OVER dialog, shinobi's GAME OVER / CONTINUE screen. "
              "Recordings were meant to stop at the death, so this only appears where "
              "one ran on.",
              former=("Screen/Game_over",),
              durational=True),
    EventType("gym-retro_game",
              "One repetition of gameplay (one .bk2 file). This is the container row "
              "that carries `stim_file`; all other events fall inside its window.",
              durational=True),
]
