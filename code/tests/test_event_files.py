import os
import pandas as pd
import os.path as op
import glob
import csv
import json


BK2_DURATION_DIFF_THRES = 0.2
BOLD_DURATION_DIFF_THRES = 30


def test_eventfiles():
    """Checks that all the .bk2 files mentioned in the events files are present
    in the sourcedata folder.

    """
    datapath = "./"
    eventfiles_list = sorted(glob.glob("sub-*/ses-*/func/*_events.tsv"))

    bk2files_fromevents = []
    for eventfile in sorted(eventfiles_list):
        event_dataframe = pd.read_csv(eventfile, sep="\t")
        assert ("stim_file" in event_dataframe)
        for filepath in event_dataframe["stim_file"]:
            if filepath != "Missing file" and not pd.isna(filepath):
                bk2files_fromevents.append(op.join(datapath, filepath))

    bk2files_infolder = []
    for root, directory, files in os.walk(datapath):
        for file in files:
            if ".bk2" in file and "ShinobiIII" in file:
                bk2files_infolder.append(op.join(root, file))

    bk2files_infolder.sort()
    bk2files_fromevents.sort()

    # Get a list of bk2 files that are referenced in the events.tsv but not found in sourcedata
    in_events_not_in_source = [
        x for x in bk2files_fromevents if x not in bk2files_infolder
    ]

    # Get a list of bk2 files that aren't referenced in the events.tsv
    in_source_not_in_events = [
        x for x in bk2files_infolder if x not in bk2files_fromevents
    ]

    error_msg = ""
    if in_events_not_in_source:
        error_msg += "\nFollowing files referenced in event files are not in the sourcedata folder:\n"
        error_msg += "\n".join(in_events_not_in_source)

    if in_source_not_in_events:
        error_msg += (
            "\nFollowing in the sourcedata folder are not referenced in event files:\n"
        )
        error_msg += "\n".join(in_source_not_in_events)

    assert not in_events_not_in_source and not in_source_not_in_events, error_msg


def test_event_files_not_empty():
    """Test event files don't contain just the header."""
    empty_files = []
    for event_path in glob.glob("sub-*/ses-*/func/*_events.tsv"):
        with open(event_path, "r") as f:
            events = csv.reader(f, delimiter="\t")
            n_lines = sum([1 for row in events])
        if n_lines < 2:
            empty_files.append(event_path)

    assert not empty_files, "\nEmpty event files :\n" + "\n".join(empty_files)


def test_durations():
    """Check duration and bk2_duration in event file match, and check duration in event file
    corresponds to the duration mentioned in bold.json files."""
    problematic_bk2 = []
    problematic_bold = []
    for event_path in sorted(glob.glob("sub-*/ses-*/func/*_events.tsv")):
        json_path = event_path.replace("_events.tsv", "_bold.json")
        with open(event_path, "r") as f:
            events = csv.reader(f, delimiter="\t")
            rows = [row for row in events]
        if len(rows) > 1:
            for row in rows[1:]:
                onset, duration, duration_bk2, level, bk2_path = row[1:]
                if (
                    bk2_path != "Missing file"
                    and abs(float(duration) - float(duration_bk2))
                    > BK2_DURATION_DIFF_THRES
                ):
                    problematic_bk2.append(
                        f"{bk2_path} : duration={duration}, duration_bk2={duration_bk2}"
                    )
            # the total duration should correpond to the last onset + the last duration
            run_duration = float(onset) + float(duration)

            with open(json_path, "r") as f:
                run_metadata = json.load(f)
            bold_duration = (
                run_metadata["dcmmeta_shape"][-1] * run_metadata["RepetitionTime"]
            )
            if (
                bold_duration < run_duration
                or bold_duration > run_duration + BOLD_DURATION_DIFF_THRES
            ):
                problematic_bold.append(
                    f"{json_path} : bold duration={bold_duration}, duration from events={run_duration}"
                )

    error_msg = ""
    if problematic_bk2:
        error_msg += "\nContradictory durations in bk2:\n"
        error_msg += "\n".join(problematic_bk2)
    if problematic_bold:
        error_msg += "\nContradictory durations in bold:\n"
        error_msg += "\n".join(problematic_bold)

    assert not problematic_bk2 and not problematic_bold, error_msg


# TODO : test to check that runs/sessions are in consecutive order
