# Shinobi Annotations Generator

This script generates BIDS-compatible event files (`*_desc-annotated_events.tsv`) and metadata sidecars (`.json`) for the Shinobi dataset. It extracts game state variables from `.bk2` replay files and computes handcrafted annotations such as keypresses, kills, and health changes.

## Prerequisites

- Python 3.8 or higher
- The Shinobi dataset with `.bk2` replay files
- ROM files in the `stimuli/` directory

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

This will install all required packages including:
- numpy, pandas
- stable-retro
- torch
- Pillow
- videogames_utils (from courtois-neuromod)
- bids_loader (from courtois-neuromod)

## Usage

### Basic Usage

From the root directory of the shinobi repository:

```bash
python code/annotations/generate_annotations.py
```

This will:
- Scan all `*_events.tsv` files in the dataset (recursively)
- For each events file, identify the corresponding `.bk2` replay files
- Generate a new `*_desc-annotated_events.tsv` file containing frame-by-frame annotations
- Create/Update `.json` sidecar files for the `.bk2` replays with metadata (score, duration, etc.)

### Options

```bash
# Specify a custom data path (default is current directory '.')
python code/annotations/generate_annotations.py --datapath /path/to/shinobi
```

## Generated Annotations

The script produces `*_desc-annotated_events.tsv` files. These files contain standard BIDS columns (`onset`, `duration`, `trial_type`) and additional columns for game state variables.

### State Variables

Sampled at 60Hz (once per frame).

- **frame_idx**: Frame index (0-indexed).
- **X_player**: Player horizontal position (corrected for resets).
- **Y_player**: Player vertical position (inverted axis in some contexts, lower number = higher).
- **shurikens**: Number of shurikens available.
- **lives**: Number of remaining lives.
- **health**: Player health (starts at 16).
- **score**: Player score.
- **level**: Current level identifier.

### Handcrafted Events

These are derived events with specific durations.

- **Keypresses**: `RIGHT`, `LEFT`, `UP`, `DOWN`, `HIT` (B), `JUMP` (C).
  - Onset: Time of press.
  - Duration: Time until release.
- **Kill**: Event marked when an enemy is killed (inferred from score increases).
- **HealthLoss**: Event marked when the player loses health.
- **HealthGain**: Event marked when the player gains health.

## Troubleshooting

### "File not found" or Missing Files
- Ensure you are running the script from the dataset root or specified the correct `--datapath`.
- Verify `stimuli/` contains the necessary ROM files for `stable-retro`.

### Dependencies Issues
- If `videogames_utils` or `bids_loader` fail to install, ensure you have git installed and access to the repositories.

### "No data path specified"
- The script defaults to the current directory. If you are running it from `code/annotations/`, it might not find the data. Run from the dataset root or use `--datapath ..`.