# -*- coding: utf-8 -*-
"""
Functions to facilitate task-based validation.

Created on Oct 25 2023
@author: Mohammad Torabi
"""

from tracemalloc import start
import numpy as np

################################# Preprocessing Functions ####################################

def events_time_to_labels(events, Fs, num_time, event_types=[], return_0_1=False):
    '''
    event_types is a list of event types to be considered. If None, 0 and 1s will be returned.
    Assigns the longest event in each TR to that TR (in the interval from last TR to current TR).
    It assumes that the first time point is TR0 which corresponds to [0 sec, TR sec] interval.
    '''
    assert events[0, 0]=='onset', 'The first column of the events file should be the onset!'
    assert events[0, 1]=='duration', 'The second column of the events file should be the duration!'
    assert events[0, 2]=='trial_type', 'The third column of the events file should be the trial type!'
    
    event_labels = np.zeros((num_time, 1))
    for i in range(events.shape[0]):
        # skip the first row which is the header
        if i==0:
            continue

        if events[i, 2] in event_types:
            start_time = float(events[i, 0])
            end_time = float(events[i, 0]) + float(events[i, 1])
            start_TR = np.round(start_time*Fs)
            end_TR = np.round(end_time*Fs)
            event_labels[start_TR:end_TR] = event_types.index(events[i, 2])
   
    if return_0_1:
        event_labels = np.multiply(event_labels!=0, 1)
        
    return event_labels