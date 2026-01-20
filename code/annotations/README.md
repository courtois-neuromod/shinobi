# Shinobi Annotations Generator

This script generates BIDS-compatible annotated event files (`*_desc-annotated_events.tsv`) for the Shinobi dataset. It reads pre-processed game variables and computes detailed annotations for gameplay events including button presses, kills, and health changes.

## Prerequisites

- Python 3.8 or higher
- The Shinobi dataset with `.bk2` replay files
- **Replays must be processed first** using `code/replays/create_replays.py` to generate `*_variables.json` files

## Installation

### 1. Create a Python virtual environment

From the root directory of the shinobi repository:

```bash
python -m venv env
```

### 2. Activate the environment

```bash
source env/bin/activate  # On Linux/Mac
# OR
env\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r code/annotations/requirements.txt
```

This will install:
- numpy
- pandas

## Usage

### Basic Usage

From the root directory of the shinobi repository:

```bash
python code/annotations/generate_annotations.py
```

This will:
- Scan all `*_events.tsv` files in the dataset (recursively)
- Load corresponding replay variables from `gamelogs/*_variables.json`
- Generate `*_desc-annotated_events.tsv` files with detailed event annotations

### Options

```bash
# Specify a custom data path (default is current directory '.')
python code/annotations/generate_annotations.py --datapath /path/to/shinobi
```

## Generated Annotations

The script produces `*_desc-annotated_events.tsv` files with the following structure:

### Column Order

| Column | Description |
|--------|-------------|
| trial_type | Type of event (see below) |
| rep_index | Repetition index within the run |
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

## Dependencies

This script requires that replays have been processed first:

```bash
# First, process replays to generate variables
python code/replays/create_replays.py --datapath .

# Then run annotations
python code/annotations/generate_annotations.py --datapath .
```

## Troubleshooting

### "Variables file not found" errors
- Ensure you've run `code/replays/create_replays.py` first
- Check that `gamelogs/*_variables.json` files exist for each .bk2 file

### "No data path specified"
- The script defaults to the current directory
- Run from the dataset root or use `--datapath /path/to/shinobi`