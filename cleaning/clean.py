import os
import os.path as op
import pandas as pd
import numpy as np


def clean_eventfiles():
    """Reads every events.tsv file and (1) replace NaNs by a meaningful string,
    (2) removes Unnamed columns (that could've been created by mistake while
    handling dataframes) and (3) adjust the filepath to add the "behavior" folder.

    """
    datapath = "./"
    eventfiles_list = []
    for root, directory, files in os.walk(datapath):
        for file in files:
            if "events.tsv" in file:
                eventfiles_list.append(op.join(root, file))

    for eventfile in sorted(eventfiles_list):
        dataframe = pd.read_csv(eventfile, sep="\t")
        # Replace NaN by proper string
        dataframe = dataframe.fillna("Missing file")
        # remove junk columns
        for column_name in dataframe.columns:
            if "Unnamed" in column_name:
                dataframe = dataframe.drop(columns=column_name)
        # Adjust the sourcedata filepath
        for idx, stimfile in enumerate(dataframe["stim_file"]):
            if "/" in stimfile and not "behavior" in stimfile:
                split_string = stimfile.split("/")
                split_string.insert(1, "behavior")
                final_string = "/".join(split_string)
                dataframe["stim_file"][idx] = final_string
        dataframe.to_csv(eventfile, sep="\t", index=False)

def bk2_to_bids():
    """Moves .bk2 files from the sourcedata folder to a proper BIDS structure.
    """
    datapath = "./"
    eventfiles_list = []
    # Get TSV list
    for root, directory, files in os.walk(datapath):
        for file in files:
            if "events.tsv" in file:
                eventfiles_list.append(op.join(root, file))
    # Open TSV and obtain bk2 path
    for eventfile in sorted(eventfiles_list):
        dataframe = pd.read_csv(eventfile, sep="\t")

        for idx, stimfile in enumerate(dataframe["stim_file"]):
            if stimfile != "Missing file": # If stimfile is not "Missing file"
                # Read info from namestring
                split_string = stimfile.split("/")
                subject = split_string[-1][4:6]
                session = split_string[-1][19:22]
                run = eventfile[-13:-11]
                level = split_string[-1][-11:-10]
                rep = '0' + str(int(split_string[-1][-5]) + 1)
                fname = f'sub-{subject}_ses-{session}_task-shinobi_run-{run}_level-{level}_rep-{rep}.bk2'
                fpath = op.join(f'sub-{subject}', f'ses-{session}', 'gamelogs')
                # Move the files
                os.makedirs(fpath, exist_ok=True)
                os.system(f'cp {stimfile} {op.join(fpath, fname)}')
                # Modify events.tsv
                dataframe["stim_file"][idx] = op.join(fpath, fname)
        dataframe.to_csv(eventfile, sep="\t", index=False)




def main():
    clean_eventfiles()
    bk2_to_bids()


if __name__ == "__main__":
    main()
