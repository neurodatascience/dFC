
"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

from re import S
from tkinter import N
import numpy as np
import hdf5storage
import scipy.io as sio
import os

from .dfc_utils import intersection, label2network
from .time_series import TIME_SERIES

################################# DATA_LOADER functions ######################################

def find_subj_list(data_root, sessions):
    '''
    find the list of subjects in data_root
    the files must follow the format: subjectID_sessionID
    only these files should be in the data_root
    '''
    ALL_FILES = os.listdir(data_root)
    FOLDERS = [item for item in ALL_FILES if os.path.isdir(data_root+item)]
    
    FOLDERS.sort()
    SUBJECTS = list()
    for s in FOLDERS:
        num = s[:s.find('_')]
        SUBJECTS.append(num)
    # the subjects might be repeated because of different sessions
    SUBJECTS = list(set(SUBJECTS))
    SUBJECTS.sort()

    print( str(len(SUBJECTS)) + ' subjects were found. ')

    failed_subjs = []
    kept_subjs = []
    for subj in SUBJECTS:
        kept_subjs.append(subj)
        for session in sessions:
            if not os.path.exists(data_root+subj+'_'+session):
                failed_subjs.append(subj)
                kept_subjs.remove(subj)
                break

    print( str(len(failed_subjs)) + ' subjects had missing sessions. ' + str(len(kept_subjs)) + ' subjects were kept. ')

    return kept_subjs

def load_from_array(subj_id2load=None, **params):
    '''
    load fMRI data from numpy or mat files
    input time_series.shape must be (time, roi)
    returns a dictionary of TIME_SERIES objects
    each corresponding to a session

    - if the file_name is a .mat file, it will be loaded using hdf5storage
      if the file_name is a .npy file, it will be loaded using np.load

    - the roi locations should be in the same folder and a .npy file
      with the name: params['roi_locs_file']
      it must 

    - and the roi labels should be in the same folder and a .npy file
      with the name: params['roi_labels_file']
      it must be a list of strings

    - labels should be in the format: Hemisphere_Network_ID
        ow, the network2include will not work properly
    '''

    SESSIONs = params['SESSIONs'] # list of sessions
    if subj_id2load is None:
        SUBJECTS = find_subj_list(params['data_root'], sessions=SESSIONs)
    else:
        SUBJECTS = [subj_id2load]

    # LOAD Region Location DATA
    locs = np.load(params['data_root']+params['roi_locs_file'], allow_pickle='True').item()
    locs = locs['locs']

    # LOAD Region Labels DATA
    labels = np.load(params['data_root']+params['roi_labels_file'], allow_pickle='True').item()
    labels = labels['labels']

    assert type(locs) is np.ndarray, 'locs must be a numpy array'
    assert type(labels) is list, 'labels must be a list'
    assert locs.shape[0] == len(labels), 'locs and labels must have the same length'
    assert locs.shape[1] == 3, 'locs must have 3 columns'

    # apply networks2include
    # if params['networks2include'] is None, all the regions will be included
    if not params['networks2include'] is None:
        nodes2include = [i for i, x in enumerate(labels) if label2network(x) in params['networks2include']]
    else:
        nodes2include = [i for i, x in enumerate(labels)]
    locs = locs[nodes2include, :]
    labels = [x for node, x in enumerate(labels) if node in nodes2include]


    BOLD = {}
    for session in SESSIONs:
        BOLD[session] = None
        for subject in SUBJECTS:

            subj_fldr = subject + '_' + session

            # LOAD BOLD Data

            if params['file_name'][params['file_name'].find('.'):] == '.mat':
                DATA = hdf5storage.loadmat(params['data_root']+subj_fldr+'/'+params['file_name'])
            elif params['file_name'][params['file_name'].find('.'):] == '.npy':
                DATA = np.load(params['data_root']+subj_fldr+'/'+params['file_name'], allow_pickle='True').item()
            time_series = DATA['ROI_data'] # time_series.shape = (time, roi)

            # change time_series.shape to (roi, time)
            time_series = time_series.T

            # apply networks2include
            time_series = time_series[nodes2include, :]

            if BOLD[session] is None:
                BOLD[session] = TIME_SERIES(data=time_series, subj_id=subject,
                                    Fs=params['Fs'],
                                    locs=locs, node_labels=labels,
                                    TS_name='BOLD Real', session_name=session
                                )
            else:
                BOLD[session].append_ts(new_time_series=time_series, subj_id=subject)

        print( '*** Session ' + session + ': ' )
        print( 'number of regions= '+str(BOLD[session].n_regions) + ', number of time points= ' + str(BOLD[session].n_time) )

    return BOLD


def nifti2array(nifti_file, 
                confound_strategy='none', standardize=False,
                n_rois=100
    ):
    '''
    this function uses nilearn maskers to extract 
    BOLD signals from nifti files
    For now it only works with schaefer atlas,
    but you can set the number of rois to extract
    {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}

    returns a numpy array of shape (time, roi)
    and labels and locs of rois

    confound_strategy:
        'none': no confounds are used
        'no_motion': motion parameters are used
        'no_motion_no_gsr': motion parameters are used 
                            and global signal regression 
                            is applied.
    '''
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn import datasets
    from nilearn.plotting import find_parcellation_cut_coords
    from nilearn.interfaces.fmriprep import load_confounds

    parc = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    atlas_filename = parc.maps
    labels = parc.labels
    # The list of labels does not contain ‘Background’ by default. 
    # To have proper indexing, you should either manually add ‘Background’ to the list of labels:
    # Prepend background label
    labels = np.insert(labels, 0, 'Background')

    # extract locs
    # test!
    # check if order is the same as labels
    locs, labels_ = find_parcellation_cut_coords(
                        atlas_filename, 
                        background_label=0, 
                        return_label_names=True
                    )

    # create the masker for extracting time series
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename, 
        labels=labels, 
        resampling_target='data',
        standardize=standardize,
    )
    
    labels = np.delete(labels, 0) # remove the background label
    labels = [label.decode() for label in labels]
    
    ### extract the timeseries
    if confound_strategy=='none':
        time_series = masker.fit_transform(nifti_file)
    elif confound_strategy=='no_motion':
        confounds_simple, sample_mask = load_confounds(
            nifti_file,
            strategy=["high_pass", "motion", "wm_csf"],
            motion="basic", wm_csf="basic"
        )
        time_series = masker.fit_transform(
            nifti_file,
            confounds=confounds_simple,
            sample_mask=sample_mask
        )
    elif confound_strategy=='no_motion_no_gsr':
        confounds_simple, sample_mask = load_confounds(
            nifti_file,
            strategy=["high_pass", "motion", "wm_csf", "global_signal"],
            motion="basic", wm_csf="basic", global_signal="basic"
        )
        time_series = masker.fit_transform(
            nifti_file,
            confounds=confounds_simple,
            sample_mask=sample_mask
        )

    return time_series, labels, locs


def nifti2timeseries(
        nifti_file, 
        n_rois, Fs,
        subj_id,
        confound_strategy='none', 
        standardize=False,
        TS_name=None,
        session=None,
    ):
    '''
    this function is only for single subject and single session data loading 
    it uses nilearn maskers to extract ROI signals from nifti files
    and returns a TIME_SERIES object

    For now it only works with schaefer atlas,
    but you can set the number of rois to extract
    {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}
    '''
    time_series, labels, locs = nifti2array(
        nifti_file=nifti_file, 
        confound_strategy=confound_strategy, 
        standardize=standardize,
        n_rois=n_rois
    )

    assert type(locs) is np.ndarray, 'locs must be a numpy array'
    assert type(labels) is list, 'labels must be a list'
    assert locs.shape[0] == len(labels), 'locs and labels must have the same length'
    assert locs.shape[1] == 3, 'locs must have 3 columns'

    # change time_series.shape to (roi, time)
    time_series = time_series.T

    if TS_name is None:
        TS_name = subj_id + ' time series'
    
    BOLD = TIME_SERIES(
                data=time_series, subj_id=subj_id,
                Fs=Fs,
                locs=locs, node_labels=labels,
                TS_name=TS_name, session_name=session
    )

    return BOLD


def multi_nifti2timeseries(
        nifti_files_list,
        subj_id_list,
        n_rois, Fs,
        confound_strategy='none', 
        standardize=False,
        TS_name=None,
        session=None,
):
    '''
    loading data of multiple subjects, but single session, from their nifti files
    '''
    BOLD_multi = None
    for subj_id, nifti_file in zip(subj_id_list, nifti_files_list):
        if BOLD_multi is None:
            BOLD_multi = nifti2timeseries(
                            nifti_file=nifti_file, 
                            n_rois=n_rois, Fs=Fs,
                            subj_id=subj_id,
                            confound_strategy=confound_strategy,
                            standardize=standardize,
                            TS_name=TS_name,
                            session=session,
                    )
        else:
            BOLD_multi.concat_ts(
                nifti2timeseries(
                        nifti_file=nifti_file,
                        n_rois=n_rois, Fs=Fs,
                        subj_id=subj_id,
                        confound_strategy=confound_strategy, 
                        standardize=standardize,
                        TS_name=TS_name,
                        session=session,
                    )
            )
    return BOLD_multi


####################################################################################################################################