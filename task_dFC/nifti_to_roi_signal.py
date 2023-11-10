from multi_analysis_dfc import (
    data_loader,
    task_utils,
)
import numpy as np
from scipy import signal
import os
import json
import warnings

warnings.simplefilter('ignore')

################################# Parameters #################################
# data paths
# main_root = '../../DATA/ds002785/' # for local
main_root = '../../../DATA/task-based/openneuro/ds002785/' # for server
fmri_prep_root = main_root + 'derivatives/fmriprep/'
output_root = main_root + 'ROI_timeseries/'

fmriprep_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

random_task_order = True
TR_common = 2.00 # not for all! check the json files

# for consistency we use 0 for resting state
TASKS = [
    'task-restingstate', 
    'task-anticipation', 
    'task-emomatching', 
    'task-faces', 
    'task-gstroop', 
    'task-workingmemory'
]

# find all subjects
ALL_SUBJs = os.listdir(fmri_prep_root)
ALL_SUBJs = [i for i in ALL_SUBJs if ('sub-' in i) and (not '.html' in i)]
ALL_SUBJs.sort()

# pick the subject
job_id = int(os.getenv("SGE_TASK_ID"))
subj = ALL_SUBJs[job_id-1] # SGE_TASK_ID starts from 1 not 0

print('subject-level ROI signal extraction CODE started running ... for subject: ' + subj + ' ...')
################################# FIND THE FUNC FILE #################################
ALL_TIME_SERIES = []
ALL_TASK_DATA = []

for task in TASKS:
    # find the func file for this subject and task
    ALL_TASK_FILES = os.listdir(fmri_prep_root + subj + '/func/')
    ALL_TASK_FILES = [i for i in ALL_TASK_FILES if (fmriprep_suffix in i) and (task in i)] # only keep the denoised files? or use the original files?
    # print(ALL_TASK_FILES)
    if not len(ALL_TASK_FILES) == 1:
        # if the func file is not found, exclude the subject
        print('Func file not found for ' + subj + ' ' + task)
        continue
    fmriprep_file = fmri_prep_root + subj + '/func/' + ALL_TASK_FILES[0]
    info_file = main_root + subj + '/func/' + ALL_TASK_FILES[0].replace(fmriprep_suffix, '_bold.json')

    ################################# LOAD JSON INFO #########################
    # Opening JSON file as a dictionary
    f = open(info_file)
    acquisition_data = json.load(f)
    f.close()
    TR = acquisition_data['RepetitionTime']

    ################################# EXTRACT TIME SERIES #########################
    # extract ROI signals
    time_series, labels, locs = data_loader.nifti2array(
        nifti_file=fmriprep_file, 
        confound_strategy='no_motion',
        standardize='zscore',
    )
    ALL_TIME_SERIES.append(time_series)
    Fs = 1/TR # the sampling rate
    ################################# EXTRACT TASK LABELS #########################
    if task == 'task-restingstate':
        events = []
        event_types = ['rest']
        event_labels = np.zeros((time_series.shape[0], 1))
        task_labels = np.zeros((time_series.shape[0], 1))
    else:
        task_events_root = main_root + subj + '/func/'
        ALL_EVENTS_FILES = os.listdir(task_events_root)
        ALL_EVENTS_FILES = [i for i in ALL_EVENTS_FILES if (subj in i) and (task in i) and ('events.tsv' in i)] 
        if not len(ALL_EVENTS_FILES) == 1:
            # if the events file is not found, exclude the subject
            print('Events file not found for ' + subj + ' ' + task)
            continue
        # load the tsv events file
        events_file = task_events_root + ALL_EVENTS_FILES[0]
        events = np.genfromtxt(events_file, delimiter='\t', dtype=str)
        # get the task labels
        event_types = ['rest'] + list(np.unique(events[1:, 2]))
        event_labels, Fs = task_utils.events_time_to_labels(
            events=events, TR_mri=TR, 
            num_time_mri=time_series.shape[0], 
            event_types=event_types, 
            oversampling=50,
            return_0_1=False
        )
        # fill task labels with 0 (rest) and k (task's index)
        task_labels = np.multiply(event_labels!=0, TASKS.index(task))
    ################################# SAVE #################################
    # save the ROI time series and labels
    region_locs = {'locs': locs}
    region_labels = {'labels': labels}
    region_signals = {'ROI_data': time_series}
    task_data = {
        'task':task, 
        'task_labels':task_labels, 'task_types': TASKS,
        'event_labels': event_labels, 'event_types': event_types, 'events': events, 
        'Fs_task': Fs, 'TR_mri': TR, 'num_time_mri': time_series.shape[0],
    }
    ALL_TASK_DATA.append(task_data)
    subj_folder = subj + '_' + task + '/'
    if not os.path.exists(output_root + subj_folder):
        os.makedirs(output_root + subj_folder)
    np.save(output_root + 'center_locs.npy', region_locs)
    np.save(output_root + 'region_labels.npy', region_labels)
    np.save(output_root + subj_folder + 'time_series.npy', region_signals)
    np.save(output_root + subj_folder + 'task_data.npy', task_data)


################################# CONCATENATE ALL TASKS #################################

# event_labels contains exact timing of the events/on and off of each task but not 
# the exact name of events (same as task_labels for each individual task). 
# task_labels shows the overall task index for each time point not the 
# event type of the time points. So all time points of a whole task run/sequence
#  have the same task label.

# check if all tasks exist
# ow do not concatenate
if len(ALL_TASK_DATA) == len(TASKS):
    task_labels_all = None
    event_labels_all = None
    time_series_all = None 
    Fs_common = 1/TR_common

    if not random_task_order:
        task_indices = np.arange(len(TASKS))
    else:
        task_indices = np.random.permutation(len(TASKS))

    for task_id in task_indices:
        task_data = ALL_TASK_DATA[task_id]

        # downsample to TR_common first
        if task_data['TR'] != TR_common:
            time_series_task = signal.resample(ALL_TIME_SERIES[task_id], int(ALL_TIME_SERIES[task_id].shape[0]*task_data['TR']/TR_common), axis=0)
        else:
            time_series_task = ALL_TIME_SERIES[task_id]

        TASKS = task_data['task_types']

        if task_data['task'] == 'task-restingstate':
            # rest task index is assumed to be 0
            event_labels = task_id * np.ones((time_series_task.shape[0], 1))
            task_labels = task_id * np.ones((time_series_task.shape[0], 1))
        else:
            task_index = TASKS.index(task_data['task'])
            events = task_data['events']
            event_types = task_data['event_types']
            event_labels = task_utils.events_time_to_labels(
                events=events, Fs=Fs_common, 
                num_time=time_series_task.shape[0], 
                event_types=event_types, 
                return_0_1=True
            )
            event_labels = event_labels * task_index
            task_labels = task_index * np.ones((time_series_task.shape[0], 1))

        if time_series_all is None:
            time_series_all = time_series_task
            event_labels_all = event_labels
            task_labels_all = task_labels
        else:
            time_series_all = np.concatenate((time_series_all, time_series_task), axis=0)
            event_labels_all = np.concatenate((event_labels_all, event_labels), axis=0)
            task_labels_all = np.concatenate((task_labels_all, task_labels), axis=0)
            
    # save time series for all tasks concatenated
    region_signals = {'ROI_data': time_series_all}
    task_data = {
        'task':'all', 
        'task_labels':task_labels_all, 'task_types': TASKS,
        'event_labels': event_labels_all, 'event_types': None, 'events': None, 
        'Fs': Fs_common, 'TR': TR_common, 'num_time': time_series_all.shape[0],
    }
    subj_folder = subj + '_' + 'task-all' + '/'
    if not os.path.exists(output_root + subj_folder):
        os.makedirs(output_root + subj_folder)
    np.save(output_root + subj_folder + 'time_series.npy', region_signals)
    np.save(output_root + subj_folder + 'task_data.npy', task_data)

print('subject-level ROI signal extraction CODE finished running ... for subject: ' + subj + ' ...')

####################################################################