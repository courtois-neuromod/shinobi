from shinobi_behav.data.data import extract_variables
import os
import pandas as pd
import os.path as op

def test_eventfiles():
    """Checks that all the .bk2 files mentioned in the events files are present
    in the sourcedata folder.

    """
    datapath = './'
    eventfiles_list = []
    for root, directory, files in os.walk(datapath):
        for file in files:
            if 'events.tsv' in file:
                eventfiles_list.append(op.join(root, file))

    bk2files_fromevents = []
    for eventfile in sorted(eventfiles_list):
        event_dataframe = pd.read_csv(eventfile, sep='\t')
        for filepath in event_dataframe['stim_file']:
            if type(filepath) == str:
                bk2files_fromevents.append(op.join(datapath, filepath))

    bk2files_infolder = []
    for root, directory, files in os.walk(datapath):
        for file in files:
            if '.bk2' in file and 'ShinobiIII' in file:
                bk2files_infolder.append(op.join(root, file))

    bk2files_infolder.sort()
    bk2files_fromevents.sort()



    in_source_in_events = [x for x in bk2files_infolder if x in bk2files_fromevents]

    in_events_not_in_source = [x for x in bk2files_fromevents if x not in bk2files_infolder]

    # Get a list of bk2 files that aren't reference in the events.tsv
    in_source_not_in_events = [x for x in bk2files_infolder if x not in bk2files_fromevents]
    with open('bk2files_not_in_events.log', 'w') as f:
        f.write('\n'.join(in_source_not_in_events))

    assert len(in_events_not_in_source) == 0, 'Files referenced in the events.tsv are missing from the sourcedata folder.'
