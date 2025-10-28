#!/usr/bin/env python
"""
This script is used to generate JSON sidecar files (and optionally additional files)
and playback videos for the mario dataset.
By default, only the JSON file is kept.
Use the flags below to have the script generate and save extra files:
  --save_videos      : Save the playback video file (.mp4).
  --save_variables  : Save the variables file (.npz) that contains game variables.
  --save_states     : Save the full RAM state at each frame into a *_states.npy file.

Use the -v/--verbose flag to display verbose output.
"""

import argparse
import os
import os.path as op
import retro
import pandas as pd
import json
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import logging
from cneuromod_vg_utils.replay import get_variables_from_replay
from cneuromod_vg_utils.video import make_mp4
from skvideo import io
import gzip
from cneuromod_vg_utils.psychophysics import (
    compute_luminance,
    compute_optical_flow,
    audio_envelope_per_frame,
)


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

def create_sidecar_dict(repetition_variables):
    sidecar_dict = {}
    sidecar_dict["Subject"] = repetition_variables["subject"]
    sidecar_dict["World"] = repetition_variables["level"][1]
    sidecar_dict["Level"] = repetition_variables["level"][-1]
    sidecar_dict["Duration"] = len(repetition_variables["score"]) / 60
    # sidecar_dict["Terminated"] = repetition_variables["terminate"][-1] == True
    if repetition_variables["player_y_screen"][-1] > 1:
        cleared = False
    elif repetition_variables["lives"][-1] == -1:
        cleared = False
    elif repetition_variables["player_state"][-1] == 6:
        cleared = False
    elif repetition_variables["player_state"][-1] == 11:
        cleared = False
    else:
        cleared = True
    return sidecar_dict


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
    json_fname = op.join(OUTPUT_FOLDER, bk2_file.replace(".bk2", ".json"))
    # Check if already processed.
    if op.exists(json_fname):
        logging.info(f"Already processed: {json_fname}")
        return
    else:
        os.makedirs(os.path.dirname(json_fname), exist_ok=True)

    logging.info(f"Processing: {bk2_path}")

    mp4_fname = json_fname.replace(".json", ".mp4")
    ramdump_fname = json_fname.replace(".json", ".npz")
    variables_fname = json_fname.replace(".json", "_variables.json")
    confs_fname = json_fname.replace(".json", "_confs.npy")
    
    skip_first_step = idx_in_run == 0
    
    repetition_variables, replay_info, replay_frames, replay_states, audio_track, audio_rate = (
        get_variables_from_replay(
            op.join(DATA_PATH, bk2_file),
            skip_first_step=skip_first_step,
            game=args.game_name,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
    )
    
    if args.save_videos:
        os.makedirs(os.path.dirname(mp4_fname), exist_ok=True)
        make_mp4(replay_frames, mp4_fname, audio=audio_track, sample_rate=audio_rate, fps=60)
        logging.info(f"Video saved to: {mp4_fname}")
    if args.save_ramdumps:
        os.makedirs(os.path.dirname(ramdump_fname), exist_ok=True)
        np.savez(ramdump_fname, np.array(replay_states))
        logging.info(f"States saved to: {ramdump_fname}")
    if args.save_variables:
        os.makedirs(os.path.dirname(variables_fname), exist_ok=True)
        with open(variables_fname, "w") as f:  # Changed 'wb' to 'w' for text mode
            json.dump(repetition_variables, f)
    if args.save_confs:
        os.makedirs(os.path.dirname(confs_fname), exist_ok=True)
        # Compute psychophysical confounds (luminance, optical flow, audio envelope)
        luminance = compute_luminance(replay_frames)
        optical_flow = compute_optical_flow(replay_frames)
        audio_envelope = audio_envelope_per_frame(
            audio_track,
            sample_rate=audio_rate,
            frame_rate=60.0,
            frame_count=len(replay_frames),
        )

        confounds_dict = {
            "luminance": luminance,
            "optical_flow": optical_flow,
            "audio_envelope": audio_envelope,
        }
        np.save(confs_fname, confounds_dict)
        logging.info(f"Confounds saved to: {confs_fname}")

    #info_dict = create_sidecar_dict(repetition_variables)
    #info_dict["IndexInRun"] = idx_in_run
    #info_dict["Run"] = run
    #info_dict["IndexGlobal"] = global_idx
    #info_dict["IndexLevel"] = level_idx
    #info_dict["LevelFullName"] = level
    #info_dict["Bk2File"] = entities
    info_dict = {}
    os.makedirs(os.path.dirname(json_fname), exist_ok=True)
    with open(json_fname, "w") as f:
        json.dump(info_dict, f)
    logging.info(f"JSON saved for: {json_fname}")
    breakpoint()

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
    if n_jobs != 1:
        with tqdm_joblib(
            tqdm(desc="Processing files", total=len(tasks))
        ) as progress_bar:
            Parallel(n_jobs=n_jobs)(
                delayed(process_bk2_file)(task, args) for task in tasks
            )
    else:
        for task in tqdm(tasks, desc="Processing files"):
            process_bk2_file(task, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datapath",
        default="sourcedata/shinobi",
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the mario dataset.",
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
        default='outputdata/',
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        default=-1,
        type=int,
        help="Number of parallel jobs to run. Use -1 to use all available cores.",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save the playback video file (.mp4).",
    )
    parser.add_argument(
        "--save_variables",
        action="store_true",
        help="Save the variables file (.json) that contains game variables.",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="Save full RAM state at each frame into a *_states.npy file.",
    )
    parser.add_argument(
        "--save_ramdumps",
        action="store_true",
        help="Save RAM dumps at each frame into a *_ramdumps.npy file.",
    )
    parser.add_argument(
        "--save_confs",
        action="store_true",
        help="Save psychophysical confounds into a *_confs.npy file.",
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
