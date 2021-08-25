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

DATA_type = 'real' # 'real' or 'simulated'
num_subj = 100
select_nodes = True
rand_node_slct = False
num_select_nodes = 50

n_states = 6 #12
n_subj_clstrs = 20
n_hid_states = 6
n_overlap = 0.5
W_sw = 44 # in seconds, 44, choose even Ws!?
n_jobs = 8
n_jobs_methods = None
verbose=0

# output_root = '../../../../RESULTs/methods_implementation/'
output_root = '/data/origami/dFC/RESULTs/methods_implementation/'
# output_root = '/Users/mte/Documents/McGill/Project/dFC/RESULTs/methods_implementation/'

if DATA_type=='simulated':
    data_root = '../../../../DATA/TVB data/'
else:
    data_root = '../../../../DATA/HCP/HCP_Gordon/'


################################# Load Real BOLD data (HCP) #################################

session = '_Rest1_LR'

if DATA_type=='real':

    ALL_RECORDS = os.listdir(data_root)
    ALL_RECORDS = [i for i in ALL_RECORDS if 'Rest' in i]
    ALL_RECORDS.sort()
    SUBJECTS = list()
    for s in ALL_RECORDS:
        num = s[:s.find('_')]
        SUBJECTS.append(num)
    SUBJECTS = list(set(SUBJECTS))
    SUBJECTS.sort()

    SUBJECTS = SUBJECTS[0:num_subj]

    BOLD = None
    for subject in SUBJECTS:

        subj_fldr = subject + session

        locs = sio.loadmat(data_root+'Gordon333_LOCS.mat')
        locs = locs['locs']

        file = data_root+'Gordon333_Key.txt'
        f = open(file, 'r')

        atlas_data = []
        for line in f:
            row = line.split()
            atlas_data.append(row)

        DATA = hdf5storage.loadmat(data_root+subj_fldr+'/ROI_data_Gordon_333_surf.mat')
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
    time_BOLD = np.load(data_root+'bold_time.npy')/1e3    
    time_series_BOLD = np.load(data_root+'bold_data.npy')

    BOLD = TIME_SERIES(data=time_series_BOLD.T, subj_id=1, Fs=1/0.5, time_array=time_BOLD, TS_name='BOLD Simulation')

################################# Load Simulated Tavg data #################################

if DATA_type=='simulated':
    time_Tavg = np.load(data_root+'tavg_time.npy')/1e3    
    time_series_Tavg = np.load(data_root+'tavg_data.npy')

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

tic = time.time()
print('Measurement Started ...')
dFC_analyzer = DFC_ANALYZER(MEASURES_lst = MEASURES, vis_TR_idx=list(range(10, 20)),\
    save_image=True, output_root=output_root,
    n_jobs=n_jobs, verbose=1, backend='loky' \
    )
dFC_analyzer.analyze(time_series=BOLD)
print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

#########################################################################################