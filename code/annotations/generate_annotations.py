# Generate annotations of game events and add them to the events_file
# Requires gym-retro installed with Shinobi III : Return of the Ninja Master properly integrated
# NB : currently assumes that the framerate stayed constant at 60fps, which is likely not the case (based on discrepancies in bk2/events durations values)

import retro
import os
import os.path as op
import pandas as pd
import numpy as np
import argparse
from torch import Tensor
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--datapath",
    default='.',
    type=str,
    help="Data path to look for events.tsv and .bk2 files. Should be the root of the shinobi dataset.",
)


def replay_bk2(path, emulator, size=None, reward=None, skip_first_step=True):
    """Replay a bk2 file and return the images as a numpy array
    of shape (n_frames, channels=3, width, height), actions a list of list of bool,
    rewards as a list of floats, done a list of bool, info a list of dict.
    """
    movie = retro.Movie(path)
    emulator.initial_state = movie.get_state()
    emulator.reset()
    images = []
    info = []
    done = []
    rewards = []
    actions = []

    if skip_first_step:
        movie.step()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        actions.append(keys)
        obs, _rew, _done, _info = emulator.step(keys)
        if size is not None:
            obs = resize(obs, size)
        images.append(obs)
        info.append(_info)
        if reward is None:
            rewards.append(_rew)
        else:
            rewards.apend(_info[reward])
        done.append(_done)
    return np.moveaxis(np.array(images), -1, 1), actions, rewards, done, info

def images_from_array(array):
    if isinstance(array, Tensor):
        array = array.numpy()
    mode = "P" if (array.shape[1] == 1 or len(array.shape) == 3) else "RGB"
    if array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    if mode == "RGB":
        array = np.moveaxis(array, 1, 3)
    if array.min() < 0 or array.max() < 1:  # if pixel values in [-0.5, 0.5]
        array = 255 * (array + 0.5)

    images = [Image.fromarray(np.uint8(arr), mode) for arr in array]
    return images
    
def save_GIF(array, path, duration=200, optimize=False):
    """Save a GIF from an array of shape (n_frames, channels, width, height),
    also accepts (n_frames, width, height) for grey levels.
    """
    assert path[-4:] == ".gif"
    images = images_from_array(array[0:-1:4])
    images[0].save(
        path, save_all=True, append_images=images[1:], optimize=optimize, loop=0, duration=duration)

def make_replay(bk2_fpath, skip_first_step, save_gif=True, duration=10):
    # Instantiate emulator
    env = retro.make("ShinobiIIIReturnOfTheNinjaMaster-Genesis")
    frames, actions, rewards, done, info = replay_bk2(bk2_fpath, env, skip_first_step=skip_first_step)
    repetition_variables = reformat_info(info, actions, env, bk2_fpath)
    if save_gif:
        save_GIF(frames, bk2_fpath.replace(".bk2", ".gif"), duration=duration, optimize=False)

    env.close()
    return repetition_variables

def reformat_info(info, actions, env, bk2_fpath):
    """
    Reformats the info structure for a dictionnary structure containing the relevant info.
    """
    repetition_variables = {}
    repetition_variables["filename"] = bk2_fpath
    repetition_variables["level"] = bk2_fpath.split("/")[-1].split("_")[-2]
    repetition_variables["subject"] = bk2_fpath.split("/")[-1].split("_")[0]
    repetition_variables["session"] = bk2_fpath.split("/")[-1].split("_")[1]
    repetition_variables["repetition"] = bk2_fpath.split("/")[-1].split("_")[-1].split(".")[0]
    repetition_variables["actions"] = env.buttons

    for key in info[0].keys():
        repetition_variables[key] = []
    for button in env.buttons:
        repetition_variables[button] = []
    
    for frame_idx, frame_info in enumerate(info):
        for key in frame_info.keys():
            repetition_variables[key].append(frame_info[key])
        for button_idx, button in enumerate(env.buttons):
            repetition_variables[button].append(actions[frame_idx][button_idx])
    
    return repetition_variables

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


def create_runevents(runvars, events_dataframe, FS=60, get_actions=True, get_healthloss=True, get_kills=True, get_frames=True):
    """Create a BIDS compatible events dataframe from game variables and start/duration info of repetitions

    Parameters
    ----------
    runvars : list
        A list of repvars dicts, corresponding to the different repetitions of a run. Each repvar must have it's own duration and onset.
    events_dataframe : pandas.DataFrame
        A BIDS-formatted DataFrame specifying the onset and duration of each repetition.
    FS : int
        The sampling rate of the .bk2 file
    get_actions : boolean
        If True, generates actions events based on key presses
    get_healthloss : boolean
        If True, generates health loss events based on changes on the "lives" variable
    get_kills : boolean
        If True, generates events indicating when an enemy has been killed (based on score increase)

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format.
    """
    all_df = [events_dataframe]
    for idx, repvars in enumerate(runvars):
        
        repvars['rep_onset'] = [events_dataframe['onset'][idx]]
        repvars['rep_duration'] = [events_dataframe['duration'][idx]]

        if "actions" in repvars.keys():
            if get_actions:
                ACTIONS = repvars["actions"]
                for act in ACTIONS:
                    temp_df = generate_key_events(repvars, act, FS=FS)
                    temp_df['onset'] = temp_df['onset'] + repvars['rep_onset']
                    all_df.append(temp_df)

            if get_healthloss:
                temp_df = generate_healthloss_events(repvars, FS=FS, dur=0.1)
                temp_df['onset'] = temp_df['onset'] + repvars['rep_onset']
                all_df.append(temp_df)

            if get_kills:
                temp_df = generate_kill_events(repvars, FS=FS, dur=0.1)
                temp_df['onset'] = temp_df['onset'] + repvars['rep_onset']
                all_df.append(temp_df)
            
            if get_frames:
                temp_df = generate_frame_events(repvars, FS=60)
                temp_df['onset'] = temp_df['onset'] + repvars['rep_onset']
                all_df.append(temp_df)
    try:
        events_df = pd.concat(all_df).sort_values(by='onset').reset_index(drop=True)
    except ValueError:
        print('No bk2 files available for this run. Returning empty df.')
        events_df = pd.DataFrame()
    return events_df

def generate_frame_events(repvars, FS=60):
    """Create a BIDS compatible events dataframe containing frame events and the associated values

    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format containing the
        corresponding events.
    """


    # Get frame events, use any action for this, in this case "START". There is one action value per frame
    number_of_frames = len(repvars["START"])
    onset = []
    duration = []
    trial_type = []
    frame_idx = []
    for idx in range(number_of_frames):
        onset.append(round(idx/FS, 3))
        duration.append(round(1/FS, 3))
        trial_type.append("frame")
        frame_idx.append(idx+1)
    data_dict = {'onset':onset,
                'duration':duration,
                'trial_type':trial_type,
                'frame_idx':frame_idx}
    # Add variables to data_dict
    for key in repvars.keys():
        if len(repvars[key]) == number_of_frames: # Check if that variable has one value per frame, if yes it goes into the event file.
            data_dict[key] = [repvars[key][i] for i in range(number_of_frames)]
        else: # If not the value itself goes in the event file, repeated at each frame. 
            data_dict[key] = [repvars[key] for x in range(number_of_frames)]

    return pd.DataFrame(data=data_dict)


def generate_key_events(repvars, key, FS=60):
    """Create a BIDS compatible events dataframe containing key (actions) events

    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition
    key : string
        Name of the action variable to process
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format containing the
        corresponding action events.
    """
    var = repvars[key]
    # always keep the first and last value as 0 so diff will register the state transition
    var[0] = 0
    var[-1] = 0

    var_bin = [int(val) for val in var]
    diffs = list(np.diff(var_bin, n=1))
    presses = [round(i/FS, 3) for i, x in enumerate(diffs) if x == 1]
    releases = [round(i/FS, 3) for i, x in enumerate(diffs) if x == -1]
    onset = presses
    duration = [round(releases[i] - presses[i], 3) for i in range(len(presses))]
    trial_type = ['{}'.format(key) for i in range(len(presses))]
    events_df = pd.DataFrame(data={'onset':onset,
                                   'duration':duration,
                                   'trial_type':trial_type})
    return events_df

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
    instant_score = repvars['score_Instant']
    diff_score = np.diff(instant_score, n=1)

    onset = []
    duration = []
    trial_type = []
    for idx, x in enumerate(diff_score):
        if x in [200,300]:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('Kill')

    #build df
    events_df = pd.DataFrame(data={'onset':onset,
                               'duration':duration,
                               'trial_type':trial_type})
    return events_df

def generate_healthloss_events(repvars, FS=60, dur=0.1):
    """Create a BIDS compatible events dataframe containing Health Loss events

    Parameters
    ----------
    repvars : dict
        A dict containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file
    dur : float
        Arbitrary duration of the generated event, defaults to 0.1

    Returns
    -------
    events_df : pandas.DataFrame
        An events DataFrame in BIDS-compatible format containing the
        Health Loss and Gain events.
    """
    health = repvars['health']
    diff_health = np.diff(health, n=1)

    onset = []
    duration = []
    trial_type = []
    for idx, x in enumerate(diff_health):
        if x < 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthLoss')
        if x > 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthGain')

    events_df = pd.DataFrame(data={'onset':onset,
                               'duration':duration,
                               'trial_type':trial_type})
    return events_df

def create_info_dict(repvars):
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
    info_dict["total health lost"] = total_health_loss

    diff_shurikens = np.diff(repvars["shurikens"], n=1)
    try:
        index_shurikens_loss = list(np.unique(diff_shurikens, return_counts=True)[0]).index(-1)
        total_shurikens_loss = np.unique(diff_shurikens, return_counts=True)[1][index_shurikens_loss]
    except Exception as e:
        total_shurikens_loss = 0
    info_dict["shurikens used"] = total_shurikens_loss
    
    info_dict["enemies killed"] = len(generate_kill_events(repvars, FS=60, dur=0.1))
    return info_dict

def main():
    # Get datapath
    args = parser.parse_args()
    DATA_PATH = args.datapath
    if DATA_PATH == ".":
        print("No data path specified. Searching files in this folder.")

    # Process each file
    for root, folder, files in os.walk(DATA_PATH):
        if not "sourcedata" in root:
            for file in files:
                if "events.tsv" in file and not "annotated" in file:
                    run_events_file = op.join(root, file)
                    events_annotated_fname = run_events_file.replace("_events.", "_desc-annotated_events.")
                    #if not op.isfile(events_annotated_fname):
                    print(f"Processing : {file}")
                    events_dataframe = pd.read_table(run_events_file)
                    bk2_files = events_dataframe['stim_file'].values.tolist()
                    runvars = []
                    for bk2_idx, bk2_file in enumerate(bk2_files):
                        if bk2_file != "Missing file" and type(bk2_file) != float:
                            print("Adding : " + bk2_file)
                            bk2_fname = op.join(DATA_PATH, bk2_file)
                            if op.exists(bk2_file):
                                #repvars = extract_variables(bk2_fname)
                                repvars = make_replay(bk2_file, skip_first_step=bk2_idx==0)
                                repvars["X_player"] = fix_position_resets(repvars["X_player"])
                                runvars.append(repvars)
                                # create json sidecar
                                info_dict = create_info_dict(repvars)
                                with open(bk2_file.replace(".bk2", ".json"), "w") as outfile:
                                    json.dump(info_dict, outfile, default=str)
                        else:
                            print("Missing file, skipping")
                            runvars.append({})
                    events_df_annotated = create_runevents(runvars, events_dataframe)
                    # Correct a few things
                    env = retro.make("ShinobiIIIReturnOfTheNinjaMaster-Genesis")
                    actions = env.buttons
                    env.close()
                    for action in actions:
                        try:
                            events_df_annotated[action].replace({"0":False,
                                                                "1":True})
                        except Exception as e:
                            print(e)
                    events_df_annotated = events_df_annotated.drop(["filename", "actions", "rep_onset", "rep_duration"], axis=1)
                    events_df_annotated.replace({'level': {'1-0': '1',
                                                           '4-1': '4',
                                                           '5-0': '5'}}, inplace = True)
                    events_df_annotated.replace({'trial_type': {'B':'HIT',
                                                                'C':'JUMP'}}, inplace = True)
                    # Save
                    events_df_annotated.to_csv(events_annotated_fname, sep="\t")
                    print("Done.")
    

if __name__ == "__main__":
    main()
