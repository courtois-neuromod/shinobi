"""
This script is used to generate replay outputs for the Shinobi dataset.

By default, all files are generated:
  - JSON sidecar file with metadata
  - MP4 video file
  - Variables JSON file with game variables
  - Low-level features NPY file (luminance, optical flow, audio envelope)

Use the flags below to skip specific outputs:
  --skip_videos      : Skip generating video files (.mp4).
  --skip_variables   : Skip generating variables files (_variables.json).
  --skip_lowlevel    : Skip generating low-level features (_lowlevel.npy).

Use the -v/--verbose flag to display verbose output.
"""

import argparse
import gc
import os
import os.path as op
import stable_retro as retro
import pandas as pd
import json
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import logging
import multiprocessing
from videogames_utils.replay import get_variables_from_replay
from videogames_utils.video import make_mp4
from videogames_utils.psychophysics import (
    compute_luminance,
    compute_optical_flow,
    audio_envelope_per_frame,
)

def fix_position_resets(X_player):
    """Sometimes X_player resets to 0 but the player's position should keep
    increasing.
    This fixes it and makes sure that X_player is continuous. If not, the
    values after the jump are corrected.

    Parameters
    ----------
    X_player : list
        List of raw positions at each timeframe from one repetition.

    Returns
    -------
    list
        List of lists of fixed (continuous) positions. One per repetition.

    """

    fixed_X_player = []
    raw_X_player = X_player
    fix = 0  # keeps trace of the shift
    fixed_X_player.append(raw_X_player[0])  # add first frame
    for i in range(1, len(raw_X_player) - 1):  # ignore first and last frames
        if raw_X_player[i - 1] - raw_X_player[i] > 100:
            fix += raw_X_player[i - 1] - raw_X_player[i]
        fixed_X_player.append(raw_X_player[i] + fix)
    fixed_X_player.append(fixed_X_player[-1]) # re-add the last frame for consistency
    return fixed_X_player

def generate_kill_events(repvars, FS=60, dur=0.1):
    """Create a BIDS compatible events dataframe containing kill events,
    based on a sudden increase of score.
    + 200 pts : basic enemies (all levels)
    + 300 pts : mortars and machineguns (lvl 5)
    + 400 pts : cauldron-heads (level 1)
    + 500 pts : anti-riot cop (lvl 5) ; hovering ninja (lvl 4)


    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition.
    FS : int
        The sampling rate of the .bk2 file
    min_dur : float
        Minimal duration of a kill segment, defaults to 1 (sec)

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format containing the
        kill events.
    """
    instant_score = repvars['instantScore']
    diff_score = np.diff(instant_score, n=1)

    onset = []
    duration = []
    trial_type = []
    level = []
    for idx, x in enumerate(diff_score):
        if x in [200,300]:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('Kill')
            level.append(repvars["level"])

    #build df
    events_df = pd.DataFrame(data={'onset':onset,
                               'duration':duration,
                               'trial_type':trial_type,
                               'level':level})
    return events_df

def get_passage_order(bk2_df):
    """
    Sorts the DataFrame and assigns cumulative indices for global and
    level-specific order.

    Parameters:
    bk2_df (pd.DataFrame): DataFrame containing replay data with a 'bk2_file' column.

    Returns:
    pd.DataFrame: The processed DataFrame with additional columns for subject,
                  session, level, global_idx, and level_idx.
    """
    bk2_df["subject"] = [
        x.split("/")[-1].split("_")[0] for x in bk2_df["bk2_file"].values
    ]
    bk2_df["session"] = [
        x.split("/")[-1].split("_")[1] for x in bk2_df["bk2_file"].values
    ]
    bk2_df = bk2_df.sort_values(["subject", "session", "run", "idx_in_run"]).assign(
        global_idx=lambda x: x.groupby("subject").cumcount()
    )
    bk2_df = bk2_df.sort_values(
        ["subject", "level", "session", "run", "idx_in_run"]
    ).assign(level_idx=lambda x: x.groupby(["subject", "level"]).cumcount())
    bk2_df = bk2_df.sort_values(["subject", "global_idx"])
    return bk2_df

def create_sidecar_dict(repvars):
    info_dict = {}

    info_dict["duration"] = len(repvars["X_player"])/60

    lives_lost = sum([x for x in np.diff(repvars["lives"], n=1) if x < 0])
    if lives_lost == 0:
        cleared = True
    else:
        cleared = False
    info_dict["cleared"] = cleared

    info_dict["end_score"] = repvars["score"][-1]

    diff_health = np.diff(repvars["health"], n=1)
    try:
        index_health_loss = list(np.unique(diff_health, return_counts=True)[0]).index(-1)
        total_health_loss = np.unique(diff_health, return_counts=True)[1][index_health_loss]
    except Exception as e:
        print(e)
        total_health_loss = 0
    info_dict["total health lost"] = int(total_health_loss)

    diff_shurikens = np.diff(repvars["shurikens"], n=1)
    try:
        index_shurikens_loss = list(np.unique(diff_shurikens, return_counts=True)[0]).index(-1)
        total_shurikens_loss = np.unique(diff_shurikens, return_counts=True)[1][index_shurikens_loss]
    except Exception as e:
        total_shurikens_loss = 0
    info_dict["shurikens used"] = int(total_shurikens_loss)

    info_dict["enemies killed"] = len(generate_kill_events(repvars, FS=60, dur=0.1))
    return info_dict

def _check_outputs_exist(json_fname, mp4_fname, variables_fname, lowlevel_fname, args):
    """
    Check which output files already exist.

    Returns:
        tuple: (all_exist, missing_outputs) where all_exist is bool and
               missing_outputs is list of output types that need to be generated
    """
    missing = []

    # JSON is always required
    if not op.exists(json_fname):
        missing.append("json")

    # Check optional outputs (if not skipped)
    if not args.skip_videos and not op.exists(mp4_fname):
        missing.append("video")
    if not args.skip_variables and not op.exists(variables_fname):
        missing.append("variables")
    if not args.skip_lowlevel and not op.exists(lowlevel_fname):
        missing.append("lowlevel")

    return len(missing) == 0, missing

def process_bk2_file(task, args):
    """
    Process one .bk2 file.

    Parameters:
      task: a tuple (bk2_file, bk2_idx, stimuli_path, run, save_videos, save_variables, save_states, total_idx)
    """

    # Get datapath
    DATA_PATH = op.abspath(args.datapath)

    # If user provides --simple, use the simplified ROM
    # and change pipeline folder name accordingly.

    args.game_name = "ShinobiIIIReturnOfTheNinjaMaster-Genesis"

    # Setup derivatives folder
    OUTPUT_FOLDER = DATA_PATH
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Make suere the game is integrated to stable-retro
    if args.stimuli is None:
        STIMULI_PATH = op.abspath(op.join(DATA_PATH, "stimuli"))
    else:
        STIMULI_PATH = op.abspath(args.stimuli)
    logging.debug(f"Adding stimuli path: {STIMULI_PATH}")
    retro.data.Integrations.add_custom_path(STIMULI_PATH)

    bk2_file, run, idx_in_run, level, subject, session, global_idx, level_idx = (
        task
    )
    bk2_path = op.abspath(op.join(DATA_PATH, bk2_file))
    if bk2_file == "Missing file" or isinstance(bk2_path, float):
        return
    if not op.exists(bk2_path):
        logging.error(f"File not found: {bk2_path}")
        return

    # Set the output file names using BIDS-like naming.
    json_fname = op.join(OUTPUT_FOLDER, bk2_file.replace(".bk2", "_summary.json"))
    mp4_fname = json_fname.replace("_summary.json", "_recording.mp4")
    variables_fname = json_fname.replace("_summary.json", "_variables.json")
    lowlevel_fname = json_fname.replace("_summary.json", "_lowlevel.npy")

    # Check if all required outputs already exist - skip if so
    all_exist, missing_outputs = _check_outputs_exist(
        json_fname, mp4_fname, variables_fname, lowlevel_fname, args
    )
    if all_exist:
        entities = bk2_file.split("/")[-1].replace(".bk2", "")
        logging.info(f"Skipping (all outputs exist): {entities}")
        return
    else:
        entities = bk2_file.split("/")[-1].replace(".bk2", "")
        logging.info(f"Processing {entities} (missing: {', '.join(missing_outputs)})")
        os.makedirs(os.path.dirname(json_fname), exist_ok=True)

    logging.debug(f"Processing: {bk2_path}")
    
    skip_first_step = idx_in_run == 0
    
    repetition_variables, replay_info, replay_frames, audio_track, audio_rate = (
        get_variables_from_replay(
            op.join(DATA_PATH, bk2_file),
            skip_first_step=skip_first_step,
            game=args.game_name,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
    )
    # Fix position resets
    repetition_variables["X_player"] = fix_position_resets(repetition_variables["X_player"])

    if not args.skip_videos:
        os.makedirs(os.path.dirname(mp4_fname), exist_ok=True)
        make_mp4(replay_frames, mp4_fname, audio=audio_track, sample_rate=audio_rate, fps=60)
        logging.info(f"Video saved to: {mp4_fname}")
    if not args.skip_variables:
        os.makedirs(os.path.dirname(variables_fname), exist_ok=True)
        with open(variables_fname, "w") as f:
            json.dump(repetition_variables, f)
        logging.info(f"Variables saved to: {variables_fname}")
    if not args.skip_lowlevel:
        os.makedirs(os.path.dirname(lowlevel_fname), exist_ok=True)
        # Compute low-level features (luminance, optical flow, audio envelope)
        luminance = compute_luminance(replay_frames)
        optical_flow = compute_optical_flow(replay_frames)
        audio_envelope = audio_envelope_per_frame(
            audio_track,
            sample_rate=audio_rate,
            frame_rate=60.0,
            frame_count=len(replay_frames),
        )

        lowlevel_dict = {
            "luminance": luminance,
            "optical_flow": optical_flow,
            "audio_envelope": audio_envelope,
        }
        np.save(lowlevel_fname, lowlevel_dict)
        logging.info(f"Low-level features saved to: {lowlevel_fname}")

    info_dict = create_sidecar_dict(repetition_variables)
    os.makedirs(os.path.dirname(json_fname), exist_ok=True)
    with open(json_fname, "w") as f:
        json.dump(info_dict, f)
    logging.info(f"JSON saved for: {json_fname}")

    # Explicitly free large objects to prevent memory accumulation
    del replay_frames, audio_track, repetition_variables, replay_info
    gc.collect()

def _worker_wrapper(task_args_tuple):
    """Wrapper function for multiprocessing - must be at module level to be picklable."""
    task, args_dict = task_args_tuple
    # Reconstruct args namespace in worker process
    args_ns = argparse.Namespace(**args_dict)
    process_bk2_file(task, args_ns)

def main(args):
    # Set logging level based on --verbose flag.
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    DATA_PATH = op.abspath(args.datapath)

    bk2_list = []
    for root, folders, files in sorted(os.walk(DATA_PATH)):
        for file in files:
            if "events.tsv" in file and "annotated" not in file:
                run_events_file = op.join(root, file)
                run = file.split("_")[-2]
                logging.info(f"Processing events file: {run_events_file}")
                try:
                    events_dataframe = pd.read_table(run_events_file)
                except Exception as e:
                    logging.error(f"Cannot read {run_events_file}: {e}")
                    continue
                bk2_files = events_dataframe["stim_file"].values.tolist()

                for idx_in_run, bk2_file in enumerate(bk2_files):
                    if isinstance(bk2_file, str):
                        if bk2_file != "Missing file":
                            level = events_dataframe["level"][idx_in_run]
                            if ".bk2" in bk2_file:
                                bk2_info = {
                                    "bk2_file": bk2_file,
                                    "run": run.split("-")[-1],
                                    "idx_in_run": idx_in_run,
                                    "level": level,
                                }
                                bk2_list.append(bk2_info)
    bk2_df = pd.DataFrame(bk2_list)
    bk2_df = get_passage_order(bk2_df)

    # Process tasks
    tasks = [tuple(row) for row in bk2_df.values]
    logging.info(f"Found {len(tasks)} bk2 files to process.")
    n_jobs = os.cpu_count() if args.n_jobs == -1 else args.n_jobs
    
    # Use multiprocessing.Pool with maxtasksperchild=1 to guarantee memory cleanup
    # This kills each worker after processing ONE file, forcing the OS to reclaim
    # all memory (including stable-retro's internal leaks)
    
    # Convert args to dict for pickling
    args_dict = vars(args)
    task_arg_pairs = [(task, args_dict) for task in tasks]
    
    if n_jobs != 1:
        with multiprocessing.Pool(processes=n_jobs, maxtasksperchild=1) as pool:
            list(tqdm(pool.imap(_worker_wrapper, task_arg_pairs), 
                     total=len(tasks), desc="Processing files"))
    else:
        # Single process mode: still use subprocess isolation
        with multiprocessing.Pool(processes=1, maxtasksperchild=1) as pool:
            list(tqdm(pool.imap(_worker_wrapper, task_arg_pairs), 
                     total=len(tasks), desc="Processing files"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datapath",
        default=".",
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the shinobi dataset.",
    )
    parser.add_argument(
        "-s",
        "--stimuli",
        default=None,
        type=str,
        help="Data path to look for the stimuli files (rom, state files, data.json etc...).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Path to the output folder (deprecated - outputs now go to gamelogs/ within datapath).",
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        default=1,
        type=int,
        help="Number of parallel jobs to run. Default is 1 (single-threaded). Use -1 to use all available cores.",
    )
    parser.add_argument(
        "--skip_videos",
        action="store_true",
        help="Skip generating the playback video file (_recording.mp4).",
    )
    parser.add_argument(
        "--skip_variables",
        action="store_true",
        help="Skip generating the variables file (_variables.json).",
    )
    parser.add_argument(
        "--skip_lowlevel",
        action="store_true",
        help="Skip generating low-level features (_lowlevel.npy).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display verbose output.",
    )

    args = parser.parse_args()

    # Main loop
    main(args)
