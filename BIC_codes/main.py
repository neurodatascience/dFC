from functions.dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio
import os
os.environ["MKL_NUM_THREADS"] = '64'
os.environ["NUMEXPR_NUM_THREADS"] = '64'
os.environ["OMP_NUM_THREADS"] = '64'

################################# Parameters #################################

DATA_type = 'Gordon' # 'Gordon' or 'simulated' or 'ICA'

num_subj = 100
select_nodes = True
rand_node_slct = False
num_select_nodes = 50

n_states = 6 #12
n_subj_clstrs = 20
n_hid_states = 4
n_overlap = 0.5
W_sw = 44 # in seconds, 44, choose even Ws!?
n_jobs = 8
n_jobs_methods = None
verbose=0

output_root = './../../../../RESULTs/methods_implementation/'
# output_root = '/data/origami/dFC/RESULTs/methods_implementation/'
# output_root = '/Users/mte/Documents/McGill/Project/dFC/RESULTs/methods_implementation/'

data_root_simul = './../../../DATA/TVB data/'
data_root_gordon = './../../../DATA/HCP/HCP_Gordon/'
data_root_ica = './../../../DATA/HCP/HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d50_ts2/'

################################# FINDING SUBJECTs LIST #################################

# ICA
ALL_RECORDS = os.listdir(data_root_ica)
ALL_RECORDS = [i for i in ALL_RECORDS if '.txt' in i]
ALL_RECORDS.sort()
SUBJECTS_ica = list()
for s in ALL_RECORDS:
    num = s[:s.find('.')]
    SUBJECTS_ica.append(num)
SUBJECTS_ica = list(set(SUBJECTS_ica))
SUBJECTS_ica.sort()

# GORDON
ALL_RECORDS = os.listdir(data_root_gordon)
ALL_RECORDS = [i for i in ALL_RECORDS if 'Rest' in i]
ALL_RECORDS.sort()
SUBJECTS_gordon = list()
for s in ALL_RECORDS:
    num = s[:s.find('_')]
    SUBJECTS_gordon.append(num)
SUBJECTS_gordon = list(set(SUBJECTS_gordon))
SUBJECTS_gordon.sort()

SUBJECTS = intersection(SUBJECTS_gordon, SUBJECTS_ica)

print( str(len(SUBJECTS)) + ' subjects were found. ' + str(num_subj) + ' subjects were selected.')

SUBJECTS = SUBJECTS[0:num_subj]

################################# Load ICA BOLD data (HCP) #################################

if DATA_type=='ICA':

    BOLD = None
    for subject in SUBJECTS:
        time_series = np.loadtxt( \
            data_root_ica + subject + '.txt', dtype='float64' \
            )
        time_series = time_series.T
        
        # time_series = time_series - np.repeat(np.mean(time_series, axis=1)[:,None], time_series.shape[1], axis=1) # ???????????????????????

        if BOLD is None:
            BOLD = TIME_SERIES(data=time_series, subj_id=subject, Fs=1/0.72, TS_name='BOLD ICA')
        else:
            BOLD.append_ts(new_time_series=time_series, subj_id=subject)

    print(BOLD.n_regions, BOLD.n_time)

################################# Load Real BOLD data (HCP) #################################

if DATA_type=='Gordon':

    session = '_Rest1_LR'

    # LOAD Region Location DATA

    locs = sio.loadmat(data_root_gordon+'Gordon333_LOCS.mat')
    locs = locs['locs']

    # LOAD Region Data

    file = data_root_gordon+'Gordon333_Key.txt'
    f = open(file, 'r')

    atlas_data = []
    for line in f:
        row = line.split()
        atlas_data.append(row)

    BOLD = None
    for subject in SUBJECTS:

        subj_fldr = subject + session

        locs = sio.loadmat(data_root_gordon+'Gordon333_LOCS.mat')
        locs = locs['locs']

        file = data_root_gordon+'Gordon333_Key.txt'
        f = open(file, 'r')

        atlas_data = []
        for line in f:
            row = line.split()
            atlas_data.append(row)

        DATA = hdf5storage.loadmat(data_root_gordon+subj_fldr+'/ROI_data_Gordon_333_surf.mat')
        time_series = DATA['ROI_data']

        time_series = time_series.T

        time_series = time_series - np.repeat(np.mean(time_series, axis=1)[:,None], time_series.shape[1], axis=1) # ???????????????????????

        if BOLD is None:
            BOLD = TIME_SERIES(data=time_series, subj_id=subject, Fs=1/0.72, locs=locs, nodes_info=atlas_data, TS_name='BOLD Real')
        else:
            BOLD.append_ts(new_time_series=time_series, subj_id=subject)

        # select nodes
        if select_nodes:
            if rand_node_slct:
                nodes_idx = np.random.choice(range(BOLD.n_regions), size=num_select_nodes, replace=False)
                nodes_idx.sort()
            else:
                nodes_idx = np.array(list(range(47, 88)) + list(range(224, 263)))
            BOLD.select_nodes(nodes_idx=nodes_idx)

    print(BOLD.n_regions, BOLD.n_time)


################################# Load Simulated BOLD data #################################

if DATA_type=='simulated':
    time_BOLD = np.load(data_root_simul+'bold_time.npy')/1e3    
    time_series_BOLD = np.load(data_root_simul+'bold_data.npy')

    BOLD = TIME_SERIES(data=time_series_BOLD.T, subj_id=1, Fs=1/0.5, time_array=time_BOLD, TS_name='BOLD Simulation')

################################# Load Simulated Tavg data #################################

if DATA_type=='simulated':
    time_Tavg = np.load(data_root_simul+'tavg_time.npy')/1e3    
    time_series_Tavg = np.load(data_root_simul+'tavg_data.npy')

    TAVG = TIME_SERIES(data=time_series_Tavg.T, subj_id=1, Fs=200, time_array=time_Tavg, TS_name='Tavg Simulation')

################################# Measure dFC #################################

params = {'W': int(W_sw*BOLD.Fs), 'n_overlap': n_overlap, \
    'n_states': n_states, 'n_subj_clstrs': n_subj_clstrs, 'n_hid_states': n_hid_states, \
    'n_jobs': n_jobs_methods, 'verbose': verbose, 'backend': 'loky' \
            }

hmm_cont = HMM_CONT(**params)
windowless = WINDOWLESS(**params)

sw_pc = SLIDING_WINDOW(sw_method='pear_corr', **params)
sw_mi = SLIDING_WINDOW(sw_method='MI', **params)
# sw_gLasso = SLIDING_WINDOW(sw_method='GraphLasso', **params)

time_freq_cwt = TIME_FREQ(method='CWT_mag', **params)
time_freq_cwt_r = TIME_FREQ(method='CWT_phase_r', **params)
time_freq_wtc = TIME_FREQ(method='WTC', **params)

swc_pc = SLIDING_WINDOW_CLUSTR(base_method='pear_corr', **params)
swc_mi = SLIDING_WINDOW_CLUSTR(base_method='MI', **params)
# swc_gLasso = SLIDING_WINDOW_CLUSTR(base_method='GraphLasso', **params)

hmm_disc_pc = HMM_DISC(base_method='pear_corr', **params)
hmm_disc_mi = HMM_DISC(base_method='MI', **params)
# hmm_disc_gLasso = HMM_DISC(base_method='GraphLasso', **params)

BOLD.visualize(start_time=0, end_time=50, nodes_lst=list(range(10)), \
     save_image=True, fig_name=output_root+'BOLD_signal')

BOLD.truncate(start_point=None, end_point=None)    #10000

MEASURES = [
    hmm_cont, \
    windowless, \
    sw_pc, \
    # sw_mi, \
    # sw_gLasso, \
    time_freq_cwt, \
    # time_freq_cwt_r, \
    time_freq_wtc, \
    swc_pc, \
    # swc_mi, \
    # swc_gLasso, \
    # swc_mi, \
    hmm_disc_pc,\
    # hmm_disc_gLasso, \
    # hmm_disc_mi \
            ]

dyn_conn_det_params = { \
    'N': 10, 'L': 1000, 'p': 100, \
    'n_jobs': n_jobs, 'backend': 'loky' \
}
params = { \
    # VISUALIZATION
    'vis_TR_idx': list(range(10, 20, 1)),'save_image': True, 'output_root': output_root, \
    # Parallelization Parameters
    'n_jobs': n_jobs, 'verbose': 1, 'backend': 'loky', \
    # Dynamic Connection Detector Parameters
    'dyn_conn_det_params': dyn_conn_det_params \
}

dFC_analyzer = DFC_ANALYZER(MEASURES_lst=MEASURES, \
    analysis_name='dyn_conn', \
    **params \
)

tic = time.time()
print('Measurement Started ...')
dFC_analyzer.analyze(time_series=BOLD)
print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

#########################################################################################