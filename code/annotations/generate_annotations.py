# Generate annotations of game events and add them to the events_file
# Requires gym-retro installed with Shinobi III : Return of the Ninja Master properly integrated
# NB : currently assumes that the framerate stayed constant at 60fps, which is likely not the case (based on discrepancies in bk2/events durations values)
# TODO : correct for difference with bk2 timings and make sure they fit within the gym-retro_game event in events.tsv
# TODO : keypress events are sometimes off by 1 frame (investigate ?)

import retro
import os
import os.path as op
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--datapath",
    default='.',
    type=str,
    help="Data path to look for events.tsv and .bk2 files. Should be the root of the shinobi dataset.",
)

def extract_variables(filepath):
    """Runs the logfile to generate a dict that saves all the variables indexed
    in the data.json file.

    Parameters
    ----------
    filepath : str
        The path to the .bk2 file location

    Returns
    -------
    repetition_variables : dict
        A Python dict with one entry per variable. In each entry, a list
        containing the state of a variable at each frame across a whole
        repetition.
    """
    # Grab some info
    split_filepath = filepath.split("/")
    split_filename = split_filepath[-1].split("_")
    level = split_filename[-2][-1]
    
    # Instantiate emulator
    env = retro.make("ShinobiIIIReturnOfTheNinjaMaster-Genesis", state=f"Level{level}")
    key_log = retro.Movie(filepath)
    env.reset()

    # Init variables dict
    repetition_variables = {}
    repetition_variables["filename"] = filepath
    repetition_variables["level"] = level
    repetition_variables["actions"] = env.buttons

    # Run the first step to obtain variable keys
    _, _, _, frame_variables = env.step([key_log.get_key(i, 0) for i in range(env.num_buttons)])

    # Init all entries
    for key in frame_variables:
        repetition_variables[key] = []
    for action in env.buttons:
        repetition_variables[action] = []
    
    # Starts accumulating variables states
    env.reset()
    while key_log.step():
        action_list = [key_log.get_key(i, 0) for i in range(env.num_buttons)]
        _, _, _, frame_variables = env.step(action_list)
        for key, item in frame_variables.items():
            repetition_variables[key].append(item)  
        for idx_action, action in enumerate(env.buttons):
            repetition_variables[action].append(action_list[idx_action])
    env.close()

    # Fix X_player
    repetition_variables["X_player"] = fix_position_resets(repetition_variables["X_player"])
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
        ACTIONS = repvars["actions"]
        if get_actions:
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


def main():
    # Get datapath
    args = parser.parse_args()
    DATA_PATH = args.datapath
    if DATA_PATH == ".":
        print("No data path specified. Searching files in this folder.")

    # Process each file
    for root, folder, files in os.walk(DATA_PATH):
        for file in files:
            if "events.tsv" in file and not "annotated" in file:
                events_annotated_fname = run_events_file.replace("_events.", "_annotated_events.")
                if not op.isfile(events_annotated_fname):
                    run_events_file = op.join(root, file)
                    print(f"Processing : {file}")
                    events_dataframe = pd.read_table(run_events_file)
                    bk2_files = events_dataframe['stim_file'].values.tolist()
                    runvars = []
                    for bk2_file in bk2_files:
                        print("Adding : " + bk2_file)
                        if bk2_file is not np.nan:
                            bk2_fname = op.join(DATA_PATH, bk2_file)
                            if op.exists(bk2_fname):
                                repvars = extract_variables(bk2_fname)
                                runvars.append(repvars)
                    events_df_annotated = create_runevents(runvars, events_dataframe)
                    events_df_annotated = events_df_annotated.drop(["filename", "actions", "rep_onset", "rep_duration"], axis=1)
                    events_df_annotated.to_csv(events_annotated_fname, sep="\t")
                    print("Done.")
    

if __name__ == "__main__":
    main()
