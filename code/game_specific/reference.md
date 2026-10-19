**This dataset's events are materially less well characterised than those of a game with
a public RAM map, and should be read with that in mind.** There is no public RAM map for Shinobi III,
and roughly half the entries in its `data.json` are background/palette scratch
(`backTree`, `backbrush`, …) with no gameplay meaning. Everything below is derived from a
counter whose meaning is unambiguous.

### On-screen visibility

The game has no object visibility at all: nothing in the available RAM map locates enemy
objects, so `EnemyDefeated` is inferred from score deltas.

### Screens

| Row | Meaning | Detection |
|---|---|---|
| `Screen/Gameplay` | The player controls the character in the level. | `blackScreen` 0 or 40 (40 is the scroll lock of an enemy wave, not a black screen) |
| `Screen/Death` | The death sequence, no control. | from the frame `health` reaches 0 to the life loss: 127-344 frames of animation and fade |
| `Screen/LevelEnd` | The end-of-level sequence, no control. | the end-of-level fade running to the end of the recording (`blackScreen` 22 or 62): ~380 frames of ROUND CLEAR tally on level 1, 15-16 frames on levels 4 and 5 |
| `Screen/Transition` | Between areas, no control. | section fades (22 or 62 then 20, ~80 frames) and the fade back in after a death (21 then 20, ~85 frames) |
| `Screen/GameOver` | The GAME OVER screen. | `blackScreen` 21 with `lives` < 0, 3 recordings |

### Events

| Event | Description | Detection |
|---|---|---|
| `PlayerDamaged` | Hit; the health bar decreased. | `health` decreases, except for the drop to 0 that starts a death, which is the death rather than a hit. |
| `HealthGained` | Collected a health item. | `health` increases. |
| `PlayerDied/Enemy` | Health drained by enemy hits. | The player's animation state (`status`, $FF415A) enters **41**, on the frame `health` reaches 0: the player collapses in the play area, 1-3 health left before the last hit. A point event; the sequence that follows is the `Screen/Death` row. 18 deaths. |
| `PlayerDied/Fall` | Fell into a pit or into water. Durational. | `status` enters **43**, on the frame `health` reaches 0: the player is at the bottom edge of the screen and the health bar is wiped in one frame, usually from full. `duration` runs to the life loss -- `lives` only drops once the death animation and fade have played, 2-6 s later -- or to the end of the recording when it stops first (4 deaths). 126 deaths. |
| `LifeGained` | Gained a life. | `lives` increases. |
| `EnemyDefeated` | Defeated an enemy. **Untyped.** | `instantScore` increases by one of the documented enemy values: **200** (basic enemies), **300** (mortars, machineguns), **400** (cauldron-heads), **500** (anti-riot cop, hovering ninja). |
| `ProjectileAppeared/Shuriken` | Threw a shuriken. | `shurikens` decreases by exactly 1. |
| `ItemCollected/Shurikens` | Picked up shurikens. | `shurikens` increases. The amount is not carried on the row; it is recoverable from `shurikens` in the repetition's `_variables.json`. |
| `WeaponPowerupStarted/Ninjutsu{n}` | Used a ninjutsu. | `ninjitsu` decreases; `typeOfNinjitsu` names it. Happens in only 8 of 666 repetitions, and `typeOfNinjitsu` is always 0, so ninjutsu is not tracked as a state. |
| `PlayerState/HitRecovery` | The game's post-hit counter, during which the player flashes. | Non-zero run of `hit_timer` (**new**, $FF4165 -- the byte at raw offset $FF4164 of the word-swapped work RAM) that begins within 3 frames of a `PlayerDamaged`: 80 or 64 → 0, one per frame. Runs from 48 (knock-backs costing no health) and the 25-frame idles at 1 are ignored. Cut where `health` reaches 0. |
| `PlayerState/Normal` | Shinobi's single form; present so that every dataset has a form for every frame of play. | One row per stretch of frames on which the player is alive (outside `Screen/Death` and `Screen/GameOver`). The temporary weapon upgrade is not tracked. |
| `LevelStarted` | A gameplay attempt began. | Frame 0. |
| `LevelCompleted` | Finished the level. | The start of the end-of-level fade that closes the recording: the last `blackScreen` stretch of the repetition holds **22** or **62** (see Screens above). Over the corpus this is present in 519 of 532 cleared repetitions and in none of the 134 failed ones; the 13 misses are recordings that stopped inside the last enemy wave, before the fade. |

### Known limitations

- **No `EnemyOnScreen` / `Enemy_counter`.** Nothing in the
  available RAM map locates enemy objects.
- **`EnemyDefeated` is untyped**, and only the four documented score values are counted.
  Other increments occur — 1000 is common, and 250/350/700/3000/5000 appear — but nothing
  available attributes them, so they are deliberately not counted rather than guessed at.
  Roughly a third of scoring events are therefore not represented. (The previous pipeline
  counted only 200 and 300, silently dropping the 400- and 500-point kills its own
  docstring listed.)
- **`LevelCompleted` depends on the recording reaching the fade.** The previous pipeline
  fabricated one five seconds before the end of any repetition in which no life was lost
  — a property of the whole repetition rather than a moment in time. An earlier release
  of this module then dropped the event, having found `blackScreen` set in every
  repetition; it is, but that value (40) is the scroll lock of an enemy wave, and the
  *trailing* fade value discriminates perfectly (see the `LevelCompleted` row above).
  13 cleared repetitions whose recording stopped inside the last enemy wave get no event;
  their `_summary.json` `Outcome` still says `cleared`.
- `PlayerDied/*` and `LifeGained` are **new**: the previous pipeline had no player-death
  event at all, despite `lives` recording it unambiguously. The cause comes from the
  death state of `status`, which takes no other value at any other time; checked on
  rendered frames for all 144 deaths. The game has no level timer, so there is no
  `PlayerDied/Timeout`, and no other death occurs in levels 1, 4 and 5.

### Controller actions

| Event | Genesis |
|---|---|
| `Action/Left`, `Action/Right`, `Action/Up`, `Action/Down` | D-pad |
| `Action/Jump` | `C` |
| `Action/Attack` | `B` |
| `Action/Ninjutsu` | `A` |
| `Action/Start` | `START` |
| `Action/Select` | `MODE` |
| `Action/Other` | `X`, `Y`, `Z` |

### Repetition outcomes

The outcome is "was a life lost", the only signal the RAM map supports.

### Provenance

No public RAM map exists for Shinobi III; everything here is derived from counters whose
meaning is unambiguous.
