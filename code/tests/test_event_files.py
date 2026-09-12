import os
import pandas as pd
import os.path as op
import glob
import csv
import json


BK2_DURATION_DIFF_THRES = 0.2
BOLD_DURATION_DIFF_THRES = 30


def is_missing(stim_file):
    """True for the "missing replay" sentinel.

    Compared case-insensitively: this dataset contains both "Missing file" and
    "Missing File", and the exact-match test silently let the second form through, so
    rows with no replay were compared as if they had one.
    """
    if not isinstance(stim_file, str):
        return True
    value = stim_file.strip().lower()
    # Some rows carry an empty stim_file rather than the sentinel.
    return value in ("", "nan", "missing file")


def plain_event_files():
    """The run-level events files, excluding the generated annotated ones.

    A plain `sub-*/ses-*/func/*_events.tsv` glob also matches
    `*_desc-annotated_events.tsv`, which has a different schema; these tests parse the
    plain layout and fail on the annotated one.
    """
    return sorted(
        path for path in glob.glob("sub-*/ses-*/func/*_events.tsv")
        if "desc-annotated" not in path
    )


def test_eventfiles():
    """Checks that all the .bk2 files mentioned in the events files are present
    in the sourcedata folder.

    """
    datapath = "./"
    eventfiles_list = sorted(plain_event_files())

    bk2files_fromevents = []
    for eventfile in sorted(eventfiles_list):
        event_dataframe = pd.read_csv(eventfile, sep="\t")
        assert ("stim_file" in event_dataframe)
        for filepath in event_dataframe["stim_file"]:
            if not pd.isna(filepath) and not is_missing(filepath):
                bk2files_fromevents.append(op.join(datapath, filepath))

    bk2files_infolder = []
    for root, directory, files in os.walk(datapath):
        # git-annex keeps its object store under .git/annex/objects, where every file is
        # also named *.bk2; walking into it would compare the events against the annex
        # internals rather than against the working tree.
        directory[:] = [d for d in directory if d != ".git"]
        for file in files:
            # Was `"ShinobiIII" in file`, which matches the pre-BIDS naming. The
            # replays were renamed to `*_task-shinobi_*.bk2`, so that filter found 0 of
            # the 666 files actually present and the test compared against an empty set.
            if file.endswith(".bk2"):
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
    for event_path in plain_event_files():
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
    for event_path in sorted(plain_event_files()):
        json_path = event_path.replace("_events.tsv", "_bold.json")
        with open(event_path, "r") as f:
            events = csv.reader(f, delimiter="\t")
            rows = [row for row in events]
        if len(rows) > 1:
            header = rows[0]
            for row in rows[1:]:
                fields = dict(zip(header, row))
                onset = fields["onset"]
                duration = fields["duration"]
                duration_bk2 = fields["duration_bk2"]
                bk2_path = fields["stim_file"]
                if (
                    not is_missing(bk2_path)
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
