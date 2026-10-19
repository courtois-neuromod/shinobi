# shinobi replay processing

Replays every `.bk2` of the dataset in the emulator and writes, next to it in
`sub-*/ses-*/gamelogs/`, its frame-by-frame game variables, a video, low-level features
and a summary. The pipeline is `videogames_utils.replays`, shared by the CNeuroMod
videogame datasets; what is specific to Shinobi III: Return of the Ninja Master (Genesis) lives in `code/game_specific/`.

Each level is replayed with its own scenario (`scenario_Level*.json`, see `code/game_specific/replays.py`), and `X_player` is corrected for the resets of the position counter before the variables are saved.

## Installation

```bash
python -m venv env
source env/bin/activate
git submodule update --init code/videogames_utils
pip install -r code/replays/requirements.txt
```

The ROM and the integration (`data.json`, states, scenarios) are in
`stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis/`.

## Usage

```bash
python code/replays/generate_replays.py -nj -1
python code/replays/generate_replays.py -sub sub-01 -ses ses-001 --skip_videos
```

- `-d`, `--datapath`: dataset root (default: the current directory).
- `-o`, `--output`: where to write the outputs (default: the dataset root).
- `-s`, `--stimuli`: where the integration is (default: `<datapath>/stimuli`).
- `-nj`, `--n_jobs`: parallel jobs, `-1` for all cores (default: 1).
- `--subjects`, `--sessions`: process only these, e.g. `-sub sub-01 -ses ses-001`.
- `--skip_videos`, `--skip_variables`, `--skip_lowlevel`: skip an output.
- `--force`: regenerate outputs that already exist (needed when `data.json` has gained
  variables).
- `-v`, `--verbose`: detailed logging.

At the end, every replay whose `done` flag is not set on its last frame only is listed:
it ends on the scenario's done condition otherwise, so this points at a problem with the
recording.

## Generated files

For each `sub-*_ses-*_task-shinobi_*.bk2`:

- `*_variables.json`: the game variables of `data.json`, one value per frame, and the
  buttons pressed.
- `*_recording.mp4`: the replay with its audio, at the emulator's frame rate.
- `*_lowlevel.npy`: luminance, optical flow and audio envelope, per frame.
- `*_summary.json`: one record per repetition, below.

## Summary fields

Common to the CNeuroMod videogame datasets:

| Field | Meaning |
|---|---|
| `Subject`, `Session`, `Run` | BIDS labels, e.g. `sub-01`, `ses-002`, `run-01`. |
| `Level` | The `level-` entity of the `.bk2`. |
| `Bk2File` | The `.bk2`, relative to the dataset root. |
| `IndexInRun` | Position among the run's repetitions that have a `.bk2`, from 0. |
| `IndexGlobal`, `IndexLevel` | Position among the subject's repetitions, overall and at this level, in the order they were played, from 0. |
| `Phase` | The protocol phase: null: the protocol has no phases. |
| `Outcome` | How the repetition ended: `cleared`, `failed/killed`, `failed/fall`, `failed/timeout`, `incomplete/warp`, `incomplete/interrupted`. |
| `Duration`, `FrameCount` | Length of the replay, in seconds and in frames. |

Specific to Shinobi III: Return of the Ninja Master (Genesis):

| Field | Meaning |
|---|---|
| `ScoreGained` | `score` at the last frame minus the first. |
| `HealthLost` | Frames on which `health` drops by 1. |
| `ShurikensThrown` | Frames on which `shurikens` drops by 1. |
| `EnemiesDefeated` | `EnemyDefeated` events: score increments of 200, 300, 400 or 500 points. |
