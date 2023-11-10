from multi_analysis_dfc import (
    data_loader,
    task_utils,
)
import numpy as np
import os
import json
import warnings

warnings.simplefilter('ignore')

################################# Parameters #################################
# data paths
# main_root = '../../DATA/ds002785' # for local
main_root = '../../../DATA/task-based/openneuro/ds002785' # for server
fmriprep_root = f"{main_root}/derivatives/fmriprep"
output_root = f"{main_root}/derivatives/ROI_timeseries"

fmriprep_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

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
ALL_SUBJs = os.listdir(fmriprep_root)
ALL_SUBJs = [i for i in ALL_SUBJs if ('sub-' in i) and (not '.html' in i)]
ALL_SUBJs.sort()

# pick the subject
job_id = int(os.getenv("SGE_TASK_ID"))
subj = ALL_SUBJs[job_id-1] # SGE_TASK_ID starts from 1 not 0

print(f"subject-level ROI signal extraction CODE started running ... for subject: {subj} ...")
################################# FIND THE FUNC FILE #################################

for task in TASKS:
    # find the func file for this subject and task
    ALL_TASK_FILES = os.listdir(f"{fmriprep_root}/{subj}/func/")
    ALL_TASK_FILES = [i for i in ALL_TASK_FILES if (fmriprep_suffix in i) and (task in i)] # only keep the denoised files? or use the original files?
    # print(ALL_TASK_FILES)
    if not len(ALL_TASK_FILES) == 1:
        # if the func file is not found, exclude the subject
        print('Func file not found for ' + subj + ' ' + task)
        continue
    fmriprep_file = f"{fmriprep_root}/{subj}/func/{ALL_TASK_FILES[0]}"
    info_file = f"{main_root}/{subj}/func/{ALL_TASK_FILES[0].replace(fmriprep_suffix, '_bold.json')}"

    ################################# LOAD JSON INFO #########################
    # Opening JSON file as a dictionary
    f = open(info_file)
    acquisition_data = json.load(f)
    f.close()
    TR_mri = acquisition_data['RepetitionTime']
    ################################# EXTRACT TIME SERIES #########################
    # extract ROI signals
    time_series, labels, locs = data_loader.nifti2array(
        nifti_file=fmriprep_file, 
        confound_strategy='no_motion',
        standardize='zscore',
    )
    num_time_mri = time_series.shape[0]
    ################################# EXTRACT TASK LABELS #########################
    oversampling = 50 # more samples per TR than the func data to have a better event_labels time resolution
    if task == 'task-restingstate':
        events = []
        event_types = ['rest']
        event_labels = np.zeros((int(num_time_mri * oversampling), 1))
        task_labels = np.zeros((int(num_time_mri * oversampling), 1))
    else:
        task_events_root = f"{main_root}/{subj}/func/"
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
        event_labels, Fs_task = task_utils.events_time_to_labels(
            events=events, TR_mri=TR_mri, 
            num_time_mri=num_time_mri, 
            event_types=event_types, 
            oversampling=oversampling,
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
        'Fs_task': Fs_task, 'TR_mri': TR_mri, 'num_time_mri': num_time_mri,
    }
    subj_folder = f"{subj}_{task}"
    if not os.path.exists(f"{output_root}/{subj_folder}/"):
        os.makedirs(f"{output_root}/{subj_folder}/")
    np.save(f"{output_root}/center_locs.npy", region_locs)
    np.save(f"{output_root}/region_labels.npy", region_labels)
    np.save(f"{output_root}/{subj_folder}/time_series.npy", region_signals)
    np.save(f"{output_root}/{subj_folder}/task_data.npy", task_data)


print(f"subject-level ROI signal extraction CODE finished running ... for subject: {subj} ...")
####################################################################