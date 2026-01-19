# Shinobi Replay Processing

This script processes `.bk2` replay files from the Shinobi dataset and generates various outputs including videos, metadata, game variables, RAM dumps, and low-level psychophysical features.

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
pip install -r code/replays/requirements.txt
```

This will install all required packages including:
- numpy, pandas
- stable-retro (for replay processing)
- joblib, tqdm (for parallel processing)
- videogames_utils (from local ../videogames_utils - includes moviepy for video generation)

**Note**: The videogames_utils package is installed from the local repository at `../../videogames_utils` relative to the shinobi repo. Make sure this directory exists and is up to date.

## Usage

### Basic Usage

From the root directory of the shinobi repository:

```bash
python code/replays/create_replays.py
```

This will:
- Scan all `*_events.tsv` files in the dataset
- Process all `.bk2` replay files referenced in those events
- Generate all output files by default in `sub-XX/ses-XXX/gamelogs/` directories

### Output Files

For each `.bk2` replay file, the following files are generated (all BIDS-compliant naming):

1. **`*_recording.mp4`** - Video playback of the replay with audio
2. **`.json`** - Metadata sidecar (duration, score, enemies killed, etc.)
3. **`*_variables.json`** - Frame-by-frame game variables (position, health, score, etc.)
4. **`*_ramdump.npz`** - Full RAM state at each frame
5. **`*_lowlevel.npy`** - Low-level psychophysical features:
   - Luminance
   - Optical flow
   - Audio envelope per frame

### Skipping Specific Outputs

If you want to skip certain outputs (e.g., to save time/space), use the `--skip_*` flags:

```bash
# Skip video generation (fastest, saves most space)
python code/replays/create_replays.py --skip_videos

# Skip multiple outputs
python code/replays/create_replays.py --skip_videos --skip_ramdumps

# Only generate JSON metadata and variables
python code/replays/create_replays.py --skip_videos --skip_ramdumps --skip_lowlevel
```

Available skip flags:
- `--skip_videos` - Skip video generation
- `--skip_variables` - Skip game variables extraction
- `--skip_ramdumps` - Skip RAM dump generation
- `--skip_lowlevel` - Skip low-level features computation

### Advanced Options

```bash
# Use parallel processing with multiple jobs (default is single-threaded)
python code/replays/create_replays.py --n_jobs 4

# Use all available CPU cores
python code/replays/create_replays.py --n_jobs -1

# Verbose output
python code/replays/create_replays.py --verbose

# Custom stimuli path (if ROMs are in a different location)
python code/replays/create_replays.py --stimuli /path/to/stimuli

# Custom data path (if running from different directory)
python code/replays/create_replays.py --datapath /path/to/shinobi
```

## How It Works

1. **Discovery**: The script walks through the dataset directory and finds all `*_events.tsv` files
2. **Extraction**: For each events file, it extracts the list of `.bk2` replay files
3. **Ordering**: Replays are sorted and assigned global and level-specific indices
4. **Smart Processing**: For each replay, the script checks which outputs already exist and only regenerates missing files
5. **Processing**: Replays are processed sequentially by default (use `--n_jobs -1` for parallel processing with all CPU cores)

## File Structure

```
shinobi/
├── sub-01/
│   ├── ses-002/
│   │   ├── func/
│   │   │   └── sub-01_ses-002_task-shinobi_run-01_events.tsv
│   │   └── gamelogs/
│   │       ├── sub-01_ses-002_task-shinobi_run-01_level-1_rep-01.bk2
│   │       ├── sub-01_ses-002_task-shinobi_run-01_level-1_rep-01.json
│   │       ├── sub-01_ses-002_task-shinobi_run-01_level-1_rep-01_recording.mp4
│   │       ├── sub-01_ses-002_task-shinobi_run-01_level-1_rep-01_variables.json
│   │       ├── sub-01_ses-002_task-shinobi_run-01_level-1_rep-01_ramdump.npz
│   │       └── sub-01_ses-002_task-shinobi_run-01_level-1_rep-01_lowlevel.npy
│   └── ...
├── stimuli/
│   └── ShinobiIIIReturnOfTheNinjaMaster-Genesis/
├── code/
│   └── replays/
│       ├── create_replays.py
│       ├── requirements.txt
│       └── README.md
└── env/  # Created by you
```

## Troubleshooting

### "File not found" errors for .bk2 files
- Ensure the `.bk2` files exist in the paths specified in the `*_events.tsv` files
- The paths in events.tsv should be relative to the dataset root

### ROM/stimuli errors
- Make sure the `stimuli/` directory exists in the dataset root
- Verify that `stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis/` contains the ROM and game data files

### Memory issues
- Keep single-threaded processing (default) or use fewer parallel jobs: `--n_jobs 2`
- Skip RAM dumps and videos: `--skip_ramdumps --skip_videos`

### Already processed files
- The script automatically detects existing outputs and skips them
- To force regeneration, delete the existing output files

## Performance Tips

- **Fastest**: `--skip_videos --skip_lowlevel --skip_ramdumps` (only JSON + variables)
- **Balanced**: `--skip_ramdumps` (all except RAM dumps)
- **Full processing**: No skip flags (default - generates everything)

Processing time per replay (approximate):
- JSON only: ~1-2 seconds
- With video: ~10-30 seconds
- With all outputs: ~30-60 seconds

## Questions or Issues?

If you encounter any problems or have questions about the script, please check:
1. That all dependencies are installed correctly
2. That the virtual environment is activated
3. That you're running from the dataset root directory
4. The verbose output for detailed error messages: `--verbose`
