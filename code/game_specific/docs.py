"""Hand-written documentation of the shinobi events, rendered into the dataset README."""

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

#: Accuracy notes, in Markdown. Reviewed by hand when the generator changes.
NOTES = """\
Shinobi is the least well characterised of the four datasets. There is **no public RAM
map** for Shinobi III, and roughly half the entries in its `data.json` are
background/palette scratch with no gameplay meaning.

- `EnemyDefeated` is inferred from score increments and is **untyped**. Only the
  documented enemy values (200, 300, 400, 500) are counted. Other increments occur (1000
  is common, as are 250/350/700/3000/5000) but nothing available attributes them, so they
  are deliberately not counted rather than guessed at -- roughly a third of scoring events
  are therefore not represented.
- There are **no** `EnemyOnScreen` events: nothing in the current RAM map locates
  enemy objects.
- `Screen/*` and `LevelCompleted` come from the (misnamed) `blackScreen` byte at
  $FF0024, a screen-mode value: 40 is the scroll lock of an enemy wave (ordinary play,
  not a black screen); 22 or 62 then 20 is a section fade; 20 from the frame health
  reaches 0 is the death animation, 21 the life loss and the fade back in; 21 with lives
  negative is the GAME OVER screen; and a fade of 22 or 62 running to the end of the
  recording is the end of a completed level (the ROUND CLEAR tally on level 1). That
  trailing fade is present in 519 of 536 cleared repetitions and in none of the 130
  failed ones; the 17 misses are recordings that stopped inside the last enemy wave. All
  checked on rendered frames.
- `PlayerDied` is durational: `lives` only drops 2-6 s after the fact, once the death
  animation and fade have played, so `onset` is the frame health reached 0 and
  `duration` runs to the life loss.
- `PlayerState/Normal` exists so that every dataset has a form for every frame of play;
  Shinobi has no forms. The temporary weapon upgrade is not tracked, and
  `WeaponPowerupStarted/Ninjutsu` marks a ninjutsu cast (8 repetitions), not a weapon
  upgrade.
- `PlayerState/HitRecovery` uses `hit_timer`, a RAM variable added to `data.json`
  ($FF4165; the Genesis core exposes work RAM word-swapped, so the byte seen at raw
  offset $FF4164 is declared one higher). It is set to 80 or 64 on the frame health drops
  and counts down once per frame while the player flashes; only stretches that begin on
  a `PlayerDamaged` are emitted. Rows appear once the sidecars have been replayed with
  that variable.
"""
