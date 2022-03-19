import os
import os.path as op
import pandas as pd
import numpy as np

def clean_eventfiles():
    datapath = './'
    eventfiles_list = []
    for root, directory, files in os.walk(datapath):
        for file in files:
            if 'events.tsv' in file:
                eventfiles_list.append(op.join(root, file))

    for eventfile in sorted(eventfiles_list):
        dataframe = pd.read_csv(eventfile, sep='\t')
        dataframe = dataframe.fillna('Missing file')
        for idx, stimfile in enumerate(dataframe['stim_file']):
            if '/' in stimfile and not 'behavior' in stimfile:
                split_string = stimfile.split('/')
                split_string.insert(2, 'behavior')
                final_string = '/'.join(split_string)
                dataframe['stim_file'][idx] = final_string
        dataframe.to_csv(eventfile, sep='\t')

def main():
    clean_eventfiles()

if __name__=="__main__":
    main()
