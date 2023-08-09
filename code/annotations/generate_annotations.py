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
from bids_loader.stimuli.game import get_variables_from_replay
from shinobi_behav.features.features import fix_position_resets

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--datapath",
    default='.',
    type=str,
    help="Data path to look for events.tsv and .bk2 files. Should be the root of the shinobi dataset.",
)

def load_max_pos(level, dataset_info_path):
    """
    Load the maximum X position for a given level from the dataset_info.json file.

    Args:
        level (int): The level for which to load the maximum X position.
        path_to_data (str, optional): The path to the data directory. Defaults to shinobi_behav.DATA_PATH.

    Returns:
        float: The maximum X position for the given level.
    """
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    max_pos = dataset_info["Maximum X position"][level]
    return max_pos

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
    level = [repvars["level"] for x in onset]
    duration = [round(releases[i] - presses[i], 3) for i in range(len(presses))]
    trial_type = ['{}'.format(key) for i in range(len(presses))]
    events_df = pd.DataFrame(data={'onset':onset,
                                   'duration':duration,
                                   'trial_type':trial_type,
                                   'level':level})
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
    level = []
    for idx, x in enumerate(diff_health):
        
        if x < 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthLoss')
            level.append(repvars["level"])
        if x > 0:
            onset.append(idx/FS)
            duration.append(dur)
            trial_type.append('HealthGain')
            level.append(repvars["level"])

    events_df = pd.DataFrame(data={'onset':onset,
                               'duration':duration,
                               'trial_type':trial_type,
                               'level':level})
    return events_df

def create_info_dict(repvars, dataset_info_path):
    """
    Creates a dictionary containing information about a replay.

    Parameters:
    repvars (dict): A dictionary containing replay variables.

    Returns:
    dict: A dictionary containing information about the replay, including the subject ID, session ID, repetition level,
    whether the level was cleared, the duration of the repetition, the final score, the amount of health lost, the percent
    complete, and whether the repetition was fake or not.
    """
    # Init dict
    info_dict = {}

    # Add subject ID
    replayfile = repvars["filename"]
    info_dict["SubjectID"] = replayfile.split("/")[-1].split("_")[0]

    # Add session ID
    info_dict["SessionID"] = replayfile.split("/")[-1].split("_")[1]

    # Add repetition level
    info_dict["Level"] = replayfile.split("/")[-1].split("_")[4]

    # Add cleared
    lives_lost = sum([x for x in np.diff(repvars["lives"], n=1) if x < 0])
    if lives_lost == 0:
        cleared = True
    else:
        cleared = False
    info_dict["Cleared"] = cleared

    # Add repetition duration
    info_dict["Duration"] = len(repvars["X_player"])/60

    # Add final score (performance)
    info_dict["FinalScore"] = repvars["score"][-1]

    # Add amount of health lost (performance)
    diff_health = np.diff(repvars["health"], n=1)
    try:
        index_health_loss = list(np.unique(diff_health, return_counts=True)[0]).index(-1)
        total_health_loss = np.unique(diff_health, return_counts=True)[1][index_health_loss]
    except Exception as e:
        print(e)
        total_health_loss = 0
    info_dict["TotalHealthLost"] = total_health_loss

    # Add percent complete
    max_pos = load_max_pos(info_dict['Level'], dataset_info_path)
    end_of_level = max_pos - 300 # We assume the level is complete is the player reaches max_pos - 300
    x_pos_clean = fix_position_resets([repvars])[0]
    percent_complete = x_pos_clean[-1]/end_of_level*100
    info_dict["PercentComplete"] = percent_complete

    # Add if fakerep
    if info_dict["FinalScore"] < 200:
        fakerep = True
    else:
        fakerep = False

    info_dict["FakeRep"] = fakerep

    return info_dict

def main():
    # Get datapath
    args = parser.parse_args()
    DATA_PATH = args.datapath
    if DATA_PATH == ".":
        print("No data path specified. Searching files in this folder.")
    dataset_info_path = op.join(DATA_PATH, ".." ,"shinobi_training", "code", "dataset_info.json")
    # Process each file
    for root, folder, files in os.walk(DATA_PATH):
        if not "sourcedata" in root:
            for file in files:
                if "events.tsv" in file and not "annotated" in file:
                    run_events_file = op.join(root, file)
                    events_annotated_fname = run_events_file.replace("_events.", "_desc-annotated_events.")
                    if not op.isfile(events_annotated_fname):
                        print(f"Processing : {file}")
                        events_dataframe = pd.read_table(run_events_file)
                        bk2_files = events_dataframe['stim_file'].values.tolist()
                        runvars = []
                        for bk2_idx, bk2_file in enumerate(bk2_files):
                            if bk2_file != "Missing file" and type(bk2_file) != float:
                                print("Adding : " + bk2_file)
                                bk2_fname = op.join(DATA_PATH, bk2_file)
                                if op.exists(bk2_fname):
                                    #repvars = make_replay(bk2_file, skip_first_step=bk2_idx==0)
                                    repvars = get_variables_from_replay(bk2_fname, skip_first_step=bk2_idx==0, save_gif=False, game=None, inttype=retro.data.Integrations.STABLE)
                                    repvars["X_player"] = fix_position_resets([repvars])[0]
                                    runvars.append(repvars)
                                    # create json sidecar
                                    info_dict = create_info_dict(repvars, dataset_info_path)
                                    with open(bk2_fname.replace(".bk2", ".json"), "w") as outfile:
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
                        try:
                            events_df_annotated = events_df_annotated.drop(["filename", "actions", "rep_onset", "rep_duration"], axis=1)
                        except Exception as e:
                            print(e)
                        events_df_annotated.replace({'level': {'1-0': 'level-1',
                                                            '4-1': 'level-4',
                                                            '5-0': 'level-5'}}, inplace = True)
                        events_df_annotated.replace({'trial_type': {'B':'HIT',
                                                                    'C':'JUMP'}}, inplace = True)
                        # Save
                        events_df_annotated.to_csv(events_annotated_fname, sep="\t")
                        print("Done.")


if __name__ == "__main__":
    main()
