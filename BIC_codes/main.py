from functions.dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio
from joblib import Parallel, delayed
import os
os.environ["MKL_NUM_THREADS"] = '64'
os.environ["NUMEXPR_NUM_THREADS"] = '64'
os.environ["OMP_NUM_THREADS"] = '64'

################################# Parameters #################################

DATA_type = 'real' # 'real' or 'simulated'

n_overlap = 0.5
W_sw = 44 # in seconds, 44, choose even Ws!?

output_root = '../../../../RESULTs/methods_implementation/'
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

    SUBJECTS = SUBJECTS[0:100]

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

        # # select nodes
        # nodes_idx = np.random.choice(range(BOLD.n_regions), size = 50, replace=False)
        # nodes_idx.sort()
        # BOLD.select_nodes(nodes_idx=None)

    print(BOLD.n_regions, BOLD.n_time)


################################# Load Simulated BOLD data #################################

if DATA_type=='simulated':
    time_BOLD = np.load(data_root+'bold_time.npy')    
    time_series_BOLD = np.load(data_root+'bold_data.npy')

    BOLD = TIME_SERIES(data=time_series_BOLD.T, subj_id=1, Fs=1/0.5, time_array=time_BOLD, TS_name='BOLD Simulation')

################################# Load Simulated Tavg data #################################

if DATA_type=='simulated':
    time_Tavg = np.load(data_root+'tavg_time.npy')    
    time_series_Tavg = np.load(data_root+'tavg_data.npy')

    TAVG = TIME_SERIES(data=time_series_Tavg.T, subj_id=1, Fs=200, time_array=time_Tavg, TS_name='Tavg Simulation')

################################# Measure dFC #################################

hmm_cont = HMM_CONT()
windowless = WINDOWLESS(n_states=5)
sw_pc = SLIDING_WINDOW(sw_method='pear_corr', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
sw_mi = SLIDING_WINDOW(sw_method='MI', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
sw_gLasso = SLIDING_WINDOW(sw_method='GraphLasso', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
time_freq_cwt = TIME_FREQ(method='CWT_mag')
time_freq_cwt_r = TIME_FREQ(method='CWT_phase_r')
time_freq_wtc = TIME_FREQ(method='WTC')
swc_pc = SLIDING_WINDOW_CLUSTR(sw_method='pear_corr', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
swc_mi = SLIDING_WINDOW_CLUSTR(sw_method='MI', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
swc_gLasso = SLIDING_WINDOW_CLUSTR(sw_method='GraphLasso', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
hmm_disc_pc = HMM_DISC(sw_method='pear_corr', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
hmm_disc_mi = HMM_DISC(sw_method='MI', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)
hmm_disc_gLasso = HMM_DISC(sw_method='GraphLasso', W=int(W_sw*BOLD.Fs), n_overlap=n_overlap)

BOLD.visualize(interval=list(range(200)), save_image=True, fig_name=output_root+'BOLD_signal')

BOLD.truncate(start_point=None, end_point=None)    #10000

MEASURES = [
    hmm_cont, \
    windowless, \
    sw_pc, \
    sw_mi, \
    # sw_gLasso, \
    time_freq_cwt, \
    time_freq_cwt_r, \
    time_freq_wtc, \
    swc_pc, \
    swc_mi, \
    # swc_gLasso, \
    swc_mi, \
    hmm_disc_pc,\
    # hmm_disc_gLasso, \
    hmm_disc_mi \
            ]

tic = time.time()
print('Measurement Started ...')
MEASURES_NEW = Parallel(n_jobs=-1, verbose=1, backend='loky')(delayed(measure.calc)(time_series=BOLD) for measure in MEASURES)
print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

dFC_analyzer = DFC_ANALYZER(MEASURES_lst = MEASURES_NEW)

################################# Visualize dFC mats #################################

dFC_analyzer.visualize_FCS(normalize=True, \
                        threshold=0.0, \
                        save_image=True, \
                        output_root=output_root + 'FCS/' \
                        )

# dFC_analyzer.visualize_dFC_mats(TR_idx=list(range(200, 300, 10)))
dFC_analyzer.visualize_dFC_mats(TR_idx=list(range(200, 300, 10)), \
                                normalize=True, \
                                threshold=0.0, \
                                fix_lim=True, \
                                save_image=True, \
                                output_root=output_root+'dFC/' \
                                )

################################# Methods dFC Corr MAT #################################

dFC_analyzer.visualize_dFC_corr(
                                save_image=True, \
                                fig_name=output_root+'dFC_corr'
                                )