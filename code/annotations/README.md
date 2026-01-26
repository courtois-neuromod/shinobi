# Shinobi Annotations Generator

Generates BIDS-compatible `*_desc-annotated_events.tsv` files from pre-processed game variables.

## Prerequisites & Installation

1.  **Environment**: Python 3.8+, Shinobi dataset (with `.bk2` replays).
2.  **Replays must be processed first** using `code/replays/generate_replays.py` to generate `*_variables.json` files.
3.  **Setup**:
    ```bash
    python -m venv env
    source env/bin/activate
    pip install -r code/annotations/requirements.txt
    ```

## Usage

```bash
python code/annotations/generate_annotations.py
```

### Arguments
-   `--datapath`: Root directory of the dataset.

## Generated Annotations

The script produces BIDS-compatible `*_desc-annotated_events.tsv` files with the following structure:

### Column Order

| Column | Description |
|--------|-------------|
| trial_type | Type of event (see below) |
| rep_index | Repetition index within the run (integer) |
| level | Level identifier (e.g., "level-1", "level-4", "level-5") |
| onset | Time in seconds from the start of the run (3 decimal places) |
| duration | Duration of the event in seconds (3 decimal places) |
| frame_start | Frame index where event starts (integer) |
| frame_stop | Frame index where event ends (integer) |

### Event Types

#### Repetition Events
- `gym-retro_game` - Base repetition events from the original events file

#### Button Press Events
Continuous events with onset and duration:
- `RIGHT`, `LEFT`, `UP`, `DOWN` - D-pad directions
- `HIT` (B button) - Attack
- `JUMP` (C button) - Jump
- Other button presses as recorded

#### Combat Events
Instantaneous events (duration=0):
- `Kill` - Enemy killed (detected from score increases of 200-300 points)

#### Health Events
Instantaneous events (duration=0):
- `HealthLoss` - Player took damage (health decreased)
- `HealthGain` - Player gained health (health increased)

#### Level Completion Events
Instantaneous events (duration=0):
- `Level_complete` - Level completed (detected via instantScore increment of 5000+ points)