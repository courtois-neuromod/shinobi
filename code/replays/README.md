# Shinobi Replay Processing

Processes `.bk2` replay files to generate video, metadata, game variables, and low-level features.

## Prerequisites & Installation

1.  **Environment**: Python 3.8+, Shinobi dataset (with `.bk2` replays), and ROMs in `stimuli/`.
2.  **Setup**:
    ```bash
    python -m venv env
    source env/bin/activate
    pip install -r code/replays/requirements.txt
    ```

## Usage

```bash
python code/replays/generate_replays.py
```

### Arguments
-   `--datapath`: Root directory of the dataset.
-   `--skip_videos`, `--skip_variables`, `--skip_lowlevel`: Skip specific outputs.
-   `--n_jobs`: Number of parallel jobs (default: 1).
-   `--stimuli`: Custom path for ROMs.
-   `--verbose`: Enable detailed logging.

## Generated Files

For each replay (e.g., `sub-{subject}_ses-{session}_task-shinobi_run-{run}_rep-{replay}.bk2`):
1.  `*_recording.mp4`: Video recording.
2.  `*_variables.json`: Frame-by-frame RAM variables.
3.  `*_lowlevel.npy`: Luminance, optical flow, and audio features.
4.  `*_summary.json`: Summary metadata (BIDS sidecar).

## Summary Variables (in sidecar JSON)

All variables rely on RAM addresses defined in `stimuli/ShinobiIIIReturnOfTheNinjaMaster-Genesis/data.json`.

| Variable | Source / Logic |
| :--- | :--- |
| **Duration** | Total replay duration in seconds. |
| **Cleared** | Whether the level was cleared without losing a life. |
| **End_score** | Final score at end of replay. |
| **Total_health_lost** | Count of health decrements. |
| **Shurikens_used** | Count of shuriken decrements. |
| **Enemies_killed** | Count of score increments of 200-300 points. |
