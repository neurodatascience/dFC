# -*- coding: utf-8 -*-
"""
Functions to facilitate task-based validation.

Created on Oct 25 2023
@author: Mohammad Torabi
"""

import numpy as np

################################# Preprocessing Functions ####################################

def events_time_to_labels(events, Fs, num_time, event_types=[], return_0_1=False):
    '''
    event_types is a list of event types to be considered. If None, 0 and 1s will be returned.
    '''
    assert events[0, 0]=='onset', 'The first column of the events file should be the onset!'
    assert events[0, 1]=='duration', 'The second column of the events file should be the duration!'
    assert events[0, 2]=='trial_type', 'The third column of the events file should be the trial type!'
    
    event_labels = np.zeros((num_time, 1))
    for i in range(events.shape[0]):
        if i==0:
            continue
        if events[i, 2] in event_types:
            event_labels[int(float(events[i, 0])*Fs):int(float(events[i, 0])*Fs)+int(float(events[i, 1])*Fs)] = event_types.index(events[i, 2])
   
    if return_0_1:
        event_labels = np.multiply(event_labels!=0, 1)
        
    return event_labels