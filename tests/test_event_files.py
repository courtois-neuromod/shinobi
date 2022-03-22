import os
import pandas as pd
import os.path as op
import glob
import csv
import json


def test_eventfiles():
    """Checks that all the .bk2 files mentioned in the events files are present
    in the sourcedata folder.

    """
    datapath = "./"
    eventfiles_list = []
    for root, directory, files in os.walk(datapath):
        for file in files:
            if "events.tsv" in file:
                eventfiles_list.append(op.join(root, file))

    bk2files_fromevents = []
    for eventfile in sorted(eventfiles_list):
        event_dataframe = pd.read_csv(eventfile, sep="\t")
        for filepath in event_dataframe["stim_file"]:
            if type(filepath) == str:
                bk2files_fromevents.append(op.join(datapath, filepath))

    bk2files_infolder = []
    for root, directory, files in os.walk(datapath):
        for file in files:
            if ".bk2" in file and "ShinobiIII" in file:
                bk2files_infolder.append(op.join(root, file))

    # TODO : add './' in front of bk2files_fromevents
    bk2files_infolder.sort()
    bk2files_fromevents.sort()

    in_source_in_events = [x for x in bk2files_infolder if x in bk2files_fromevents]

    in_events_not_in_source = [
        x for x in bk2files_fromevents if x not in bk2files_infolder
    ]

    # Get a list of bk2 files that aren't reference in the events.tsv
    in_source_not_in_events = [
        x for x in bk2files_infolder if x not in bk2files_fromevents
    ]
    with open("bk2files_not_in_events.log", "w") as f:
        f.write("\n".join(in_source_not_in_events))

    assert (
        len(in_events_not_in_source) == 0
    ), "Files referenced in the events.tsv are missing from the sourcedata folder."
    # TODO : print la liste des fichiers manquants
    # TODO : ajouter ces fichiers dans un fichier créé pour l'occasion et le supprimer si le test réussi


def test_event_files_not_empty():
    empty_files = []
    for event_path in glob.glob("sub-*/ses-*/func/*_events.tsv"):
        with open(event_path, "r") as f:
            events = csv.reader(f, delimiter="\t")
            n_lines = sum([1 for row in events])
        if n_lines < 2:
            empty_files.append(event_path)

    assert not empty_files, "Empty event files :" + "\n".join(empty_files)


def test_durations():
    problematic_bk2 = []
    problematic_bold = []
    for event_path in glob.glob("sub-*/ses-*/func/*_events.tsv"):
        json_path = event_path.replace("_events.tsv", "_bold.json")
        with open(event_path, "r") as f:
            events = csv.reader(f, delimiter="\t")
            rows = [row for row in events]
        if len(rows) > 1:
            for row in rows[1:]:
                onset, duration, duration_bk2, level, bk2_path = row[2:]
                if (
                    bk2_path != "Missing file"
                    and abs(float(duration) - float(duration_bk2)) > 0.1
                ):
                    problematic_bk2.append(
                        f"{bk2_path} : duration={duration}, duration_bk2={duration_bk2}"
                    )
            # the total duration should correpond to the last onset + the last duration
            run_duration = float(onset) + float(duration)

            with open(json_path, "r") as f:
                run_metadata = json.load(f)
            bold_duration = float(
                run_metadata["time"]["samples"]["AcquisitionTime"][-1]
            ) - float(run_metadata["time"]["samples"]["AcquisitionTime"][0])
            if bold_duration < run_duration or bold_duration > run_duration + 30:
                problematic_bold.append(
                    f"{json_path} : bold duration={bold_duration}, duration from events={run_duration}"
                )

    assert not problematic_bk2 and not problematic_bold, (
        "Contradictory durations in bk2:\n"
        + "\n".join(problematic_bk2)
        + "\nContradictory durations in bold:\n"
        + "\n".join(problematic_bold)
    )
