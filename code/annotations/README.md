# Annotated event files - shinobi

`generate_annotations.py` turns the per-frame RAM dumps written by
`code/replays/generate_replays.py` (`gamelogs/*_variables.json`) into
BIDS event files at `sub-*/ses-*/func/*_desc-annotated_events.tsv`.

The event logic is shared by all four CNeuroMod videogame datasets and lives in
[`videogames_utils.events`](https://github.com/courtois-neuromod/videogames_utils);
this directory only holds the command-line front end. The machine-readable
vocabulary is published as `task-shinobi_events.json` at the dataset root.

## Usage

```bash
python code/annotations/generate_annotations.py -d . --overwrite --validate
```

## Columns

| Column | Description |
|---|---|
| `trial_type` | Event type, from the controlled vocabulary below. |
| `level` | Level of the repetition the event belongs to. |
| `onset` | Seconds from the start of the run. |
| `duration` | Seconds. 0 for point events. |
| `frame_start` | First emulator frame of the event, relative to its repetition. |
| `frame_stop` | Last emulator frame of the event, relative to its repetition. |
| `button` | Raw controller button behind an `Action/*` event. |
| `stim_file` | Path to the repetition's `.bk2` replay. |

Onsets are computed at the console's true frame rate (59.922743 Hz, read from the
emulator core), not the 60.0 Hz the pipeline previously assumed. That correction
shifts onsets by up to ~0.6 s in the longest repetitions.

## Event types

`{...}` is replaced at generation time with a decoded name.

| Event | Description |
|---|---|
| `Player_damaged` | The player is hit and loses the current power-up state (Mario games) or health (Shinobi). |
| `Player_died` | The player loses all health or dies from an environmental hazard. |
| `Life_gained` | The player collects or earns an extra life. |
| `Health_gained` | The player collects a health item and the health bar increases. |
| `Player_state/Hit_recovery` | Post-hit recovery: the player has just been damaged and blinks. In the Mario games nothing can hurt the player until it ends; in Shinobi it is the game's post-hit counter. |
| `Item_collected/{item_type}` | The player collects a health item, weapon upgrade, extra life or other collectible. |
| `Enemy_defeated` | The player defeats an enemy. Shinobi has no RAM map for enemy types, so this event is untyped and is inferred from score increments. |
| `Projectile_appeared/Shuriken` | The player throws a shuriken. A point event: Shinobi has no RAM map for object positions, so the projectile cannot be tracked on screen. |
| `Weapon_powerup_started/{powerup_type}` | The player receives a temporary or persistent weapon upgrade. |
| `Weapon_powerup_expired/{powerup_type}` | A temporary weapon upgrade ends or is lost. |
| `Level_started` | A new level or gameplay attempt begins. |
| `Level_completed` | The player successfully finishes the level. |
| `Action/Left` | The player holds the left direction. |
| `Action/Right` | The player holds the right direction. |
| `Action/Up` | The player holds the up direction. |
| `Action/Down` | The player holds the down direction (duck / crouch). |
| `Action/Jump` | The player presses the jump button. |
| `Action/Attack` | The player presses the attack button (Shinobi). |
| `Action/Ninjutsu` | The player presses the ninjutsu button (Shinobi). |
| `Action/Other` | The player presses a button with no documented function in this game (e.g. L/R on the SNES pad in Super Mario All-Stars). The raw button is in the `button` column. |
| `Action/Start` | The player presses START (pauses the game). |
| `Action/Select` | The player presses SELECT / MODE. |
| `gym-retro_game` | One repetition of gameplay (one .bk2 file). This is the container row that carries `stim_file`; all other events fall inside its window. |

## Renamed from the previous vocabulary

This release renames every event type. Old analyses filtering on the former
names need updating; the mapping is:

| Former | Now |
|---|---|
| `DOWN` | `Action/Down` |
| `HIT` | `Action/Attack` |
| `HealthGain` | `Health_gained` |
| `HealthLoss` | `Player_damaged` |
| `Hit/powerup_lost` | `Player_damaged` |
| `JUMP` | `Action/Jump` |
| `Kill` | `Enemy_defeated` |
| `LEFT` | `Action/Left` |
| `Level_complete` | `Level_completed` |
| `MODE` | `Action/Select` |
| `NINJUTSU` | `Action/Ninjutsu` |
| `RIGHT` | `Action/Right` |
| `SELECT` | `Action/Select` |
| `START` | `Action/Start` |
| `UP` | `Action/Up` |

### Accuracy notes and known limitations

Shinobi is the least well characterised of the four datasets. There is **no public RAM
map** for Shinobi III, and roughly half the entries in its `data.json` are
background/palette scratch with no gameplay meaning.

- `Enemy_defeated` is inferred from score increments and is **untyped**. Only the
  documented enemy values (200, 300, 400, 500) are counted. The previous pipeline counted
  only 200 and 300, silently dropping the 400- and 500-point kills its own docstring
  listed. Other increments occur (1000 is common, as are 250/350/700/3000/5000) but
  nothing available attributes them, so they are deliberately not counted rather than
  guessed at -- roughly a third of scoring events are therefore not represented.
- There are **no** `Enemy_on_screen` / `Enemy_counter` events:
  nothing in the current RAM map locates enemy objects.
- `Level_completed` is **not emitted**. The previous pipeline fabricated it, placing an
  event five seconds before the end of any repetition in which no life was lost -- which
  is a property of the whole repetition, not a moment in time, and is the same rule that
  defines the summary's `Outcome`. Candidate real signals were checked over 14
  repetitions and none discriminates: `blackScreen` is set in every repetition, cleared
  or not; `status` ends at 0 in nearly all; `section` reaches 2 in only 8 of 11 cleared
  repetitions and never exceeds 1 on level 5. Whether a repetition was cleared remains
  available in its `_summary.json` `Outcome`.
- `Player_state/Hit_recovery` uses `hit_timer`, **a RAM variable added in this release**
  ($FF4165; the Genesis core exposes work RAM word-swapped, so the byte seen at raw
  offset $FF4164 is declared one higher). It is set to 80 or 64 on the frame health drops
  and counts down once per frame while the player flashes. The same byte also runs from
  48 after knock-backs that cost no health and idles at 1 for ~25 frames at other
  moments, so only stretches that begin on a `Player_damaged` are emitted, and a stretch
  is cut where health reaches 0. Whether the game actually ignores damage during this
  window has not been verified; the row states the game's own post-hit counter.
- Ninjutsu is **not** tracked as a state: `ninjitsu` decrements in only 8 of the 666
  repetitions and `typeOfNinjitsu` is 0 throughout, so there is nothing to build on.
- `Player_died` and `Life_gained` are new. The previous pipeline had **no player-death
  event at all**, despite `lives` recording it unambiguously.

Recovering enemy types and a genuine end-of-level event needs a RAM reverse-engineering
workstream, which is out of scope for this pass.

## Validation

```bash
python -m videogames_utils.events.validate_cli check . shinobi
python -m videogames_utils.events.validate_cli cross-port ../mario ../mariostars
```

`check` runs two layers: the schema and controlled-vocabulary checks (V0), and
invariants recomputed straight from `_variables.json` by a different route than
the generator used (V1) -- coin counts against the coin counter, deaths against
the lives counter, level completion against the summary outcome, enemy track
bookkeeping, and frame-range bounds.

A human video review measures precision and recall per event type:

```python
from videogames_utils.events import review
review.build_review_set('.', 'shinobi', out_dir='/tmp/review', per_type=50)
# rate the clips in /tmp/review/review.html, then:
review.score_reviews('/tmp/review/ratings.json', '/tmp/review/manifest.json')
```
