# -*- coding: utf-8 -*-
"""
Functions to facilitate task-based validation.

Created on Oct 25 2023
@author: Mohammad Torabi
"""

import numpy as np
from nilearn import glm
from scipy import signal

################################# Preprocessing Functions ####################################

def events_time_to_labels(events, TR_mri, num_time_mri, event_types=[], oversampling=50, return_0_1=False):
    '''
    event_types is a list of event types to be considered. If None, 0 and 1s will be returned.
    Assigns the longest event in each TR to that TR (in the interval from last TR to current TR).
    It assumes that the first time point is TR0 which corresponds to [0 sec, TR sec] interval.
    oversampling: number of samples per TR_mri to improve the time resolution of tasks
    '''
    assert events[0, 0]=='onset', 'The first column of the events file should be the onset!'
    assert events[0, 1]=='duration', 'The second column of the events file should be the duration!'
    assert events[0, 2]=='trial_type', 'The third column of the events file should be the trial type!'
    
    Fs = float(1 / TR_mri) * oversampling
    num_time_task = int(num_time_mri * oversampling)
    event_labels = np.zeros((num_time_task, 1))
    for i in range(events.shape[0]):
        # skip the first row which is the header
        if i==0:
            continue

        if events[i, 2] in event_types:
            start_time = float(events[i, 0])
            end_time = float(events[i, 0]) + float(events[i, 1])
            start_TR = int(np.rint(start_time * Fs))
            end_TR = int(np.rint(end_time * Fs))
            event_labels[start_TR:end_TR] = event_types.index(events[i, 2])
   
    if return_0_1:
        event_labels = np.multiply(event_labels!=0, 1)
        
    return event_labels, Fs


################################# Validation Functions ####################################


def event_conv_hrf(event_signal, TR_mri, TR_task):
    time_length_HRF = 32.0 # in sec
    hrf_model = 'glover' # 'spm' or 'glover'

    TR_HRF = TR_task
    oversampling = TR_mri/TR_HRF # more samples per TR than the func data to have a better HRF resolution,  same as for event_labels
    if hrf_model=='glover':
        HRF = glm.first_level.glover_hrf(tr=TR_mri, oversampling=oversampling, time_length=time_length_HRF, onset=0.0)
    elif hrf_model=='spm':
        HRF = glm.first_level.spm_hrf(tr=TR_mri, oversampling=oversampling, time_length=time_length_HRF, onset=0.0)

    events_hrf = signal.convolve(HRF, event_signal, mode='full')[:len(event_signal)]

    return events_hrf


def event_labels_conv_hrf(event_labels, TR_mri, TR_task):
    '''
    event_labels: event labels including 0 and event ids at the time each event happens
    TR_mri: TR of MRI
    TR_task: TR of task
    assums that 0 is the resting state

    return: event labels convolved with HRF for each event type
    the convolved event labels have the same length as the event_labels
    event type i convolved with HRF is in events_hrf[:, i-1]
    '''
    
    event_labels = np.array(event_labels)
    L = event_labels.shape[0]
    event_ids = np.unique(event_labels)
    event_ids = event_ids.astype(int)
    events_hrf = np.zeros((L, len(event_ids))) # 0 is the resting state
    for i, event_id in enumerate(event_ids):
        # 0 is not an event, is the resting state
        if event_id == 0:
            continue
        event_signal = np.zeros(L)
        event_signal[event_labels[:, 0]==event_id] = 1.0

        events_hrf[:, i] = event_conv_hrf(event_signal, TR_mri, TR_task)

    # the time points that are not in any event are considered as resting state
    events_hrf[np.sum(events_hrf[:, 1:], axis=1)==0.0, 0] = 1.0

    # time_length_task = len(event_labels)*TR_task

    return events_hrf