# %% 7. lesioned network simulation using the heterogeneity model

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
from matplotlib.ticker import MaxNLocator
from neurolib.utils.signal import RatesSignal
from statannotations.Annotator import Annotator
# matplotlib.use("Qt5Agg")
from neurolib.utils.loadData import Dataset
import neurolib.utils.functions as func
from neurolib.utils.atlases import AutomatedAnatomicalParcellation2
from neurolib.models.wc import WCModel
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
from scipy.signal import convolve
from spectral_connectivity import Multitaper, Connectivity
import itertools
import random
from scipy.signal import butter, filtfilt
from nilearn.plotting import plot_connectome
from mne.time_frequency import tfr_multitaper
from scipy.stats import pearsonr
from neurolib.utils.functions import matrix_correlation, ts_kolmogorov
from scipy.signal import resample_poly
from scipy.signal.windows import tukey
from neurolib.models.bold import BOLDModel
import scipy.signal as signal

def model_para_set(curr_cmat, time_s):
    """Create Wilson-cowan model and set parameters

    :param cmat: structral connect weight matrix
    :param dmat: tract length matrix (not used)
    :param seed: random seed to set differnt initial condition
    :return: WC model
    """
    model = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat, seed=4)
    model.params['duration'] = 1 * time_s * 1000    # time_s senconds
    model.params['signalV'] = 0
    model.params['sigma_ou'] = 0.03
    model.params['c_excexc'] = 0
    model.params['K_gl'] = 1
    model.params['Cmat'] = curr_cmat

    return model


def eeg_plot(model):
    """
    draw eeg signal
    :param model: WC model
    :return: 0
    """
    # compute sFC and correlation scores
    fr = RatesSignal.from_model_output(model, group="", time_in_ms=True)
    fr_exc = fr['exc'].data.data[:, :]
    fr_inh = fr['inh'].data.data[:, :]
    eeg = eeg_generate(fr_exc)
    eeg = eeg * 1e-6  # 单位转成V，mne的默认单位
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=500,  # frequency
    )
    location = {'FP1': [-0.309, 0.95, -0.0349], 'FP2': [0.309, 0.95, -0.0349], 'Fz': [-0, 0.719, 0.695],
                'F3': [-0.545, 0.673, 0.5],
                'F4': [0.545, 0.673, 0.5], 'F7': [-0.809, 0.587, -0.0349], 'F8': [0.809, 0.587, -0.0349],
                'FCz': [-0, 0.391, 0.921],
                'FC3': [-0.676, 0.36, 0.643], 'FC4': [0.676, 0.36, 0.643], 'FT7': [-0.95, 0.309, 0.0349],
                'FT8': [0.95, 0.309, -0.0349],
                'Cz': [0, 0, 1], 'C3': [-0.719, 0, 0.695], 'C4': [0.719, 0, 0.695], 'T3': [-0.999, 6.12e-17, -0.0349],
                'T4': [0.999, 6.12e-17, -0.0349], 'CP3': [-0.676, -0.36, 0.643], 'CP4': [0.676, -0.36, 0.643],
                'TP7': [-0.95, -0.309, -0.0349],
                'TP8': [0.95, -0.309, -0.0349], 'Pz': [-8.81e-17, -0.719, 0.695], 'P3': [-0.545, -0.673, 0.5],
                'P4': [0.545, -0.673, 0.5],
                'T5': [-0.809, -0.587, -0.0349], 'T6': [0.809, -0.587, -0.0349], 'Oz': [-1.22e-16, -0.999, -0.0349],
                'O1': [-0.309, -0.95, -0.0349],
                'O2': [0.309, -0.95, -0.0349]
                }
    for key in location:
        location[key] = 0.1 * np.array(location[key])
    montage = mne.channels.make_dig_montage(location)
    info.set_montage(montage)
    raw = mne.io.RawArray(eeg, info)
    raw.filter(6, 30)

    # Brain region classification mapping
    region_mapping = {
        "FP1": "Frontal", "FP2": "Frontal", "Fz": "Frontal",
        "F3": "Frontal", "F4": "Frontal", "F7": "Frontal",
        "F8": "Frontal", "FCz": "Frontal", "FC3": "Frontal",
        "FC4": "Frontal", "FT7": "Frontal", "FT8": "Frontal",
        "Cz": "Central", "C3": "Central", "C4": "Central",
        "T3": "Temporal", "T4": "Temporal", "CP3": "Parietal",
        "CP4": "Parietal", "TP7": "Temporal", "TP8": "Temporal",
        "Pz": "Parietal", "P3": "Parietal", "P4": "Parietal",
        "T5": "Temporal", "T6": "Temporal", "Oz": "Occipital",
        "O1": "Occipital", "O2": "Occipital"
    }

    # Define brain region colors
    region_colors = {
        "Frontal": "#206E9E",
        "Central": "#E47B26",
        "Temporal": "#35903A",
        "Parietal": "#BE2A2C",
        "Occipital": "#8363A1"
    }

    # sort channels by brain region
    region_order = ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]
    sorted_channels = []
    for region in region_order:
        for ch in raw.ch_names:
            if region_mapping[ch] == region:
                sorted_channels.append(ch)

    fig, axs = plt.subplots(len(raw.ch_names), 1, figsize=(13 / 2.54, 12 / 2.54), sharex=True)
    offset = 0
    for i, (ax, ch) in enumerate(zip(axs, sorted_channels)):
        ch_data = raw.get_data(picks=raw.ch_names.index(ch)) + offset
        times = raw.times
        region = region_mapping[ch]
        ax.plot(times, ch_data[0], color=region_colors[region], lw=0.8, label=ch)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(-0.025, 0.5, ch, transform=ax.transAxes, va='center', fontsize=8)
        offset += 20e-6

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Add a time scale line
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    scale_length = 500 * (x_max - x_min) / 5000
    scale_label = '1 s'
    scale_x_end = x_max - scale_length / 2
    scale_x_start = scale_x_end - scale_length
    scale_y = y_min - (y_max - y_min) * 0.3
    plt.hlines(y=scale_y, xmin=scale_x_start, xmax=scale_x_end,
               color='black', linewidth=1)
    plt.text(scale_x_start + scale_length / 2, scale_y - (y_max - y_min) * 1.4,
             scale_label, ha='center', fontsize=8)
    # plt.show()

    return 0


def plot_fc_band(model):
    """Using filted rate signal to calculate simulated functional connnectivity

    :param model: WC model
    :return: simulated fc matrix, correlations between sFC and eFC
    """
    # compute sFC and correlation scores
    fr = RatesSignal.from_model_output(model, group="", time_in_ms=True)
    fr_exc = fr['exc'].data.data[:, :]
    eeg = eeg_generate(fr_exc)
    eeg = eeg * 1e-6  # 单位转成V，mne的默认单位
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=500,  # frequency
    )
    location = {'FP1': [-0.309, 0.95, -0.0349], 'FP2': [0.309, 0.95, -0.0349], 'Fz': [-0, 0.719, 0.695],
                'F3': [-0.545, 0.673, 0.5],
                'F4': [0.545, 0.673, 0.5], 'F7': [-0.809, 0.587, -0.0349], 'F8': [0.809, 0.587, -0.0349],
                'FCz': [-0, 0.391, 0.921],
                'FC3': [-0.676, 0.36, 0.643], 'FC4': [0.676, 0.36, 0.643], 'FT7': [-0.95, 0.309, 0.0349],
                'FT8': [0.95, 0.309, -0.0349],
                'Cz': [0, 0, 1], 'C3': [-0.719, 0, 0.695], 'C4': [0.719, 0, 0.695], 'T3': [-0.999, 6.12e-17, -0.0349],
                'T4': [0.999, 6.12e-17, -0.0349], 'CP3': [-0.676, -0.36, 0.643], 'CP4': [0.676, -0.36, 0.643],
                'TP7': [-0.95, -0.309, -0.0349],
                'TP8': [0.95, -0.309, -0.0349], 'Pz': [-8.81e-17, -0.719, 0.695], 'P3': [-0.545, -0.673, 0.5],
                'P4': [0.545, -0.673, 0.5],
                'T5': [-0.809, -0.587, -0.0349], 'T6': [0.809, -0.587, -0.0349], 'Oz': [-1.22e-16, -0.999, -0.0349],
                'O1': [-0.309, -0.95, -0.0349],
                'O2': [0.309, -0.95, -0.0349]
                }
    for key in location:
        location[key] = 0.1 * np.array(location[key])
    montage = mne.channels.make_dig_montage(location)
    info.set_montage(montage)
    raw = mne.io.RawArray(eeg, info)
    raw.filter(6, 30)

    raw_new = raw.copy().crop(tmin=2, tmax=4)  # Compute FC from the stable middle 2s of the MI trial to match empirical data
    raw_data = raw_new.get_data()
    n_trials = 1
    manual_epochs_data = raw_data.reshape(-1, raw_data.shape[0], int(raw_data.shape[1]/n_trials))  
    time_series = np.transpose(manual_epochs_data, (2, 0, 1))  
    multitaper = Multitaper(time_series, sampling_frequency=500)
    connectivity = Connectivity.from_multitaper(multitaper, expectation_type="time_trials_tapers", blocks=1)
    fc = connectivity.imaginary_coherence()  # method: imagery coherence
    fc = np.abs(fc)
    f_range = np.where(np.logical_and(connectivity.frequencies >= 14, connectivity.frequencies <= 20))[0]    # low beta band
    connect = np.squeeze(np.mean(fc[f_range[0]:f_range[-1] + 1, :, :], axis=0))

    return connect

def compute_mst_leaf(FC):
    """Compute MST leaf number

    :param FC: simulated fc matrix
    :return: MST leaf number, betweeness centrality order index, betweeness centrality value
    """
    MST = nx.maximum_spanning_tree(nx.from_numpy_array(FC), weight='weight')
    leaf_nodes = [x for x in MST.nodes() if MST.degree(x) == 1]

    return len(leaf_nodes)

def draw_mst_star_path(FC):
    """Draw MST star and path structure

    :param FC: simulated fc matrix from lesion SC
    :return:
    """
    # create MST
    MST = nx.maximum_spanning_tree(nx.from_numpy_array(FC), weight='weight')
    leaf_nodes = [x for x in MST.nodes() if MST.degree(x) == 1]
    adj = nx.to_numpy_array(MST)
    adj = np.where(adj == 0, 0, 1)
    edge_prop = dict()
    edge_prop['lw'] = 2
    edge_prop['c'] = 'black'
    edge_prop['alpha'] = 0.5
    node_degree = MST.degree()
    r, g, b, c = 'orangered', 'limegreen', 'royalblue', 'orange'
    mst_fc_star = np.zeros_like(FC)     # mark star node connected edges as 1
    mst_fc_path = np.zeros_like(FC)     # mark path edges as 1

    # set node color and size
    node_color_star, node_size_star = [], []
    for i, d in node_degree:
        if d >= 4:
            node_color_star.append(r)   # star node as red color
            node_size_star.append(15)
        else:
            node_color_star.append('w')
            node_size_star.append(0)

    node_color_path = ['w'] * 29
    node_size_path = np.zeros(29)

    # find star (degree>=4) and path (degree=2) subnetwork
    for node1, node2 in MST.edges():
        if MST.degree(node1) >= 4 and MST.degree(node2) < 4:
            node_color_star[node2] = g      # other nodes as green color
            node_size_star[node2] = 7
            mst_fc_star[node1, node2] = 1
            mst_fc_star[node2, node1] = 1
        elif MST.degree(node1) < 4 and MST.degree(node2) >= 4:
            node_color_star[node1] = g
            node_size_star[node1] = 7
            mst_fc_star[node1, node2] = 1
            mst_fc_star[node2, node1] = 1
        elif MST.degree(node1) >= 4 and MST.degree(node2) >= 4:
            mst_fc_star[node1, node2] = 1
            mst_fc_star[node2, node1] = 1
    for node1, node2 in MST.edges():
        if MST.degree(node1) == 2 or MST.degree(node2) == 2:
            mst_fc_path[node1, node2] = 1
            mst_fc_path[node2, node1] = 1
            node_color_path[node1] = b      # path nodes as blue color
            node_color_path[node2] = b
            node_size_path[node1] = 9
            node_size_path[node2] = 9

    # plot star and path subnetwork using nilearn connectome
    coords = np.array([[-0.309, 0.95, -0.0349], [0.309, 0.95, -0.0349], [0, 0.719, 0.695], [-0.445, 0.673, 0.5],
                       [0.445, 0.673, 0.5], [-0.809, 0.587, -0.0349], [0.809, 0.587, -0.0349], [0, 0.391, 0.921],
                       [-0.576, 0.36, 0.643], [0.576, 0.36, 0.643], [-0.95, 0.309, 0.0349], [0.95, 0.309, -0.0349],
                       [0, 0, 1], [-0.619, 0, 0.695], [0.619, 0, 0.695], [-0.999, 6.12e-17, -0.0349],
                       [0.999, 6.12e-17, -0.0349], [-0.576, -0.36, 0.643], [0.576, -0.36, 0.643],
                       [-0.95, -0.309, -0.0349],
                       [0.95, -0.309, -0.0349], [0, -0.719, 0.921], [-0.445, -0.673, 0.5], [0.445, -0.673, 0.5],
                       [-0.809, -0.587, -0.0349], [0.809, -0.587, -0.0349], [0, -0.999, 0.695],
                       [-0.309, -0.95, -0.0349],
                       [0.309, -0.95, -0.0349]])
    coords = coords * 60
    coords[:, 1] = coords[:, 1] * 1.4 - 16
    coords[:, 0] = coords[:, 0] + 2
    edge_prop = dict()
    edge_prop['lw'] = 0.8
    edge_prop['alpha'] = 0.8
    edge_prop['c'] = r
    edge_prop['ls'] = '-'

    fig1, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    plot_connectome(mst_fc_star, coords, node_size=node_size_star, display_mode='z',
                    edge_kwargs=edge_prop, axes=ax, node_color=node_color_star, alpha=0.2, colorbar=False)
    ax.set_title('Starlike subnetwork')
    edge_prop['c'] = b


    fig2, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    plot_connectome(mst_fc_path, coords, node_size=node_size_path, display_mode='z',
                    edge_kwargs=edge_prop, axes=ax, node_color=node_color_path, alpha=0.2, colorbar=False)
    ax.set_title('Pathlike subnetwork')

    return fig1, fig2


def violin_plot_leaf(data, x, y, pairs):
    """Violin plot for ahand MI

    :param data: dataframe
    :param x: NIHSS group
    :param y: Leaf number
    :param pairs: paired group to statistical analysis
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    sns.stripplot(data=data, x=x, y=y, ax=ax, legend=False, size=2.5, jitter=0.13, palette='muted',
                  facecolors='none', order=['High', 'Low'])
    for dots in ax.collections:   # remove facecolor and set edgecolor
        facecolors = dots.get_facecolors()
        dots.set_edgecolors(facecolors.copy())
        dots.set_facecolors('none')
        dots.set_linewidth(1)

    sns.violinplot(data=data, x=x, y=y,  ax=ax, palette='muted', legend='brief', order=['High', 'Low'], fill=False, width=0.4)   # width=0.4, , dodge=True
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='Global efficiency')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    s1, p1 = scipy.stats.shapiro(data[data[x] == 'Low'][y])
    s2, p2 = scipy.stats.shapiro(data[data[x] == 'High'][y])
    if p1 < 0.05 or p2 < 0.05:  # nonnormal distribution
        test_method = 'Mann-Whitney'
    else:
        test_method = 't-test_ind'
    annotator = Annotator(ax, data=data, x=x, y=y, pairs=pairs)
    annotator.configure(test=test_method, hide_non_significant=False, loc='outside')
    annotator.apply_test()
    ax, test_results = annotator.annotate()
  
    for res in test_results:
        print(res.data)
    

def firingrate_P(SC):
    """
    Comparison of average power within the 8-30 Hz frequency band in C3 channel with and without MI stimulation
    :param model: WC model
    :return: simulated fc matrix, correlations between sFC and eFC
    """
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=500,  # frequency
    )
    location = {'FP1': [-0.309, 0.95, -0.0349], 'FP2': [0.309, 0.95, -0.0349], 'Fz': [-0, 0.719, 0.695],
                'F3': [-0.545, 0.673, 0.5],
                'F4': [0.545, 0.673, 0.5], 'F7': [-0.809, 0.587, -0.0349], 'F8': [0.809, 0.587, -0.0349],
                'FCz': [-0, 0.391, 0.921],
                'FC3': [-0.676, 0.36, 0.643], 'FC4': [0.676, 0.36, 0.643], 'FT7': [-0.95, 0.309, 0.0349],
                'FT8': [0.95, 0.309, -0.0349],
                'Cz': [0, 0, 1], 'C3': [-0.719, 0, 0.695], 'C4': [0.719, 0, 0.695], 'T3': [-0.999, 6.12e-17, -0.0349],
                'T4': [0.999, 6.12e-17, -0.0349], 'CP3': [-0.676, -0.36, 0.643], 'CP4': [0.676, -0.36, 0.643],
                'TP7': [-0.95, -0.309, -0.0349],
                'TP8': [0.95, -0.309, -0.0349], 'Pz': [-8.81e-17, -0.719, 0.695], 'P3': [-0.545, -0.673, 0.5],
                'P4': [0.545, -0.673, 0.5],
                'T5': [-0.809, -0.587, -0.0349], 'T6': [0.809, -0.587, -0.0349], 'Oz': [-1.22e-16, -0.999, -0.0349],
                'O1': [-0.309, -0.95, -0.0349],
                'O2': [0.309, -0.95, -0.0349]
                }
    for key in location:
        location[key] = 0.1 * np.array(location[key])
    montage = mne.channels.make_dig_montage(location)
    info.set_montage(montage)

    # Constructing the WC model
    model = model_para_set(SC, time_s=16)  # Removing the initial 10s of unstable data, 1s preparation,4s motor imagery phase, and 1s break. 
    duration = model.params['duration']
    x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
    z = MI_stimulus_function(x)
    stimulus = np.zeros((80, duration * 10))
    stimulus[1, :] = z
    stimulus[55, :] = z
    model.params["exc_ext"] = stimulus
    model.run(chunkwise=False, bold=False)
    fr = RatesSignal.from_model_output(model, group="", time_in_ms=True)
    eeg = eeg_generate(fr['exc'].data.data[:, :])
    eeg = eeg * 1e-6  # uV to V
    raw = mne.io.RawArray(eeg, info)
    raw.filter(6, 30)

    # calculate ERSP 
    raw0 = raw.copy().crop(tmin=0, tmax=1)  # resting
    raw1 = raw.copy().crop(tmin=3, tmax=5)  # MI (Extract middle 2s to match empirical epoch length)
    psd0 = raw0.compute_psd(method='multitaper', fmin=8, fmax=30, verbose=False)
    psd1 = raw1.compute_psd(method='multitaper', fmin=8, fmax=30, verbose=False)
    baseline_power = psd0.get_data().mean(axis=1)
    mi_power = psd1.get_data().mean(axis=1)
    ersp_diff = 10 * np.log10(mi_power / baseline_power)

    fig, ax = plt.subplots(1, 1, figsize=(4.5 / 2.54, 4.5 / 2.54))
    c4_idx = raw.info['ch_names'].index('C4')
    plt.plot(psd0.freqs, psd0.get_data()[c4_idx, :], label='Baseline')
    plt.plot(psd1.freqs, psd1.get_data()[c4_idx, :], label='MI')
    plt.legend()

    # spatial topography of ERSP
    ch_names = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
    plot_names = [None] * len(ch_names)
    plot_names[13] = "C3"
    plot_names[14] = "C4"
    plt.rc('font', size=11, family='Arial', weight='normal')
    mask = np.full(29, False, dtype=bool)
    mask[13:15] = 'True'
    fig, ax = plt.subplots(1, 1, figsize=(4.5 / 2.54, 4.5 / 2.54))
    im, _ = mne.viz.plot_topomap(ersp_diff, info, sphere=(0, 0, 0, 0.1), contours=1, cmap='RdBu_r', axes=ax,
                                 names=plot_names,
                                 mask=mask, mask_params={'markersize': 1, 'markerfacecolor': 'none'}, show=False)
    im.set_clim(-1.5, 1)  
    clb = fig.colorbar(im, ax=ax, shrink=0.61)  
    clb.ax.set_title('dB')
    clb.ax.tick_params(labelsize=9)
    ax.set_title('8-30 Hz', fontsize=11)
    plt.rc('font', size=9, family='Arial', weight='normal')
    plt.tight_layout()
    # plt.savefig('figure/mi_ersp.svg', bbox_inches='tight', dpi=300)
    plt.show()

    return 0


def eeg_generate(fr_exc):
    '''
    Generate EEG from the neuronal population activity results of the original model
    '''
    # Loading excitatory firing rates from WC output.
    rates = fr_exc
    tau_rise = 2  # ms
    tau_decay = 20  # 20ms
    sfreq = 10000
    rates = rates[:, int(10 * sfreq):]  # Removing the initial 10s of unstable data

    # Generating the Kernel Function
    kernel_length_ms = 5 * tau_decay
    t_k = np.arange(0, kernel_length_ms, 1 / sfreq * 1000)
    kernel = np.exp(-t_k / tau_decay) - np.exp(-t_k / tau_rise)
    kernel /= np.max(kernel)  

    # Performing Convolution
    n_rois, n_times = rates.shape
    eeg_source = np.zeros_like(rates)

    for i in range(n_rois):
        r_centered = rates[i, :] - np.mean(rates[i, :])
        eeg_source[i, :] = signal.convolve(r_centered, kernel, mode='same')


    scaling_factor = 0.01
    current_total = eeg_source * scaling_factor

    # Aligning brain region order with the Leadfield matrix
    current_total_left = current_total[::2, :]
    current_total_right = current_total[1::2, :]
    current_total_new = np.concatenate((current_total_left, current_total_right), axis=0)

    # Calculate scalp potential
    fwd = mne.read_forward_solution("data_load/model_data/custom_forward.fif", verbose=True)
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=True)
    leadfield = fwd_fixed['sol']['data']
    eeg0 = leadfield @ current_total_new   # EEG(μV)

    # downsampling
    down_factor = int(10000 / 500)  
    eeg = signal.resample_poly(eeg0, up=1, down=down_factor, axis=1)
    return eeg


def MI_stimulus_function(x): 
    """
    Function for simulating motor imagery
    """
    z = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        # Removing the initial 10s of unstable data, 1s preparation,4s motor imagery phase, and 1s break. 
        # Applying stimulation during the motor imagery (MI) phase
        if x[i] <= 11 or x[i] >= 15:
            z[i] = 0  # resting state
        else:
            z[i] = 0.192 / (1 + np.exp(15 * (np.abs(x[i] - 13) - 1.5)))  
           
    return z


def fmri_fit(bold_signal):
    sim_bold_raw = bold_signal[:, 15:]  
    fc = func.fc(sim_bold_raw)
    fc_corr = np.mean([matrix_correlation(fc, f) for f in ds.FCs])

    return fc, fc_corr


def MI_stimulation_draw(SC):
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=500,  # frequency
    )
    location = {'FP1': [-0.309, 0.95, -0.0349], 'FP2': [0.309, 0.95, -0.0349], 'Fz': [-0, 0.719, 0.695],
                'F3': [-0.545, 0.673, 0.5],
                'F4': [0.545, 0.673, 0.5], 'F7': [-0.809, 0.587, -0.0349], 'F8': [0.809, 0.587, -0.0349],
                'FCz': [-0, 0.391, 0.921],
                'FC3': [-0.676, 0.36, 0.643], 'FC4': [0.676, 0.36, 0.643], 'FT7': [-0.95, 0.309, 0.0349],
                'FT8': [0.95, 0.309, -0.0349],
                'Cz': [0, 0, 1], 'C3': [-0.719, 0, 0.695], 'C4': [0.719, 0, 0.695], 'T3': [-0.999, 6.12e-17, -0.0349],
                'T4': [0.999, 6.12e-17, -0.0349], 'CP3': [-0.676, -0.36, 0.643], 'CP4': [0.676, -0.36, 0.643],
                'TP7': [-0.95, -0.309, -0.0349],
                'TP8': [0.95, -0.309, -0.0349], 'Pz': [-8.81e-17, -0.719, 0.695], 'P3': [-0.545, -0.673, 0.5],
                'P4': [0.545, -0.673, 0.5],
                'T5': [-0.809, -0.587, -0.0349], 'T6': [0.809, -0.587, -0.0349], 'Oz': [-1.22e-16, -0.999, -0.0349],
                'O1': [-0.309, -0.95, -0.0349],
                'O2': [0.309, -0.95, -0.0349]
                }
    for key in location:
        location[key] = 0.1 * np.array(location[key])
    montage = mne.channels.make_dig_montage(location)
    info.set_montage(montage)

    # Constructing the WC model
    model = model_para_set(SC, time_s=16)  # Removing the initial 10s of unstable data, 1s preparation,4s motor imagery phase, and 1s break. 
    duration = model.params['duration']
    x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
    z = MI_stimulus_function(x)
    stimulus = np.zeros((80, duration * 10))
    stimulus[1, :] = z
    stimulus[55, :] = z
    model.params["exc_ext"] = stimulus
    model.run(chunkwise=False, bold=False)
    fr = RatesSignal.from_model_output(model, group="", time_in_ms=True)
    signal0 = fr["exc"].data.data

    signal0 = signal0[:, 10000 * 10:]  # Removing the initial 10s of unstable data
    signal0 = scipy.signal.resample_poly(signal0, up=1, down=int(10000 / 500), axis=1) # downsampling
    eeg0 = eeg_generate(fr['exc'].data.data[:, :])
    eeg0 = eeg0 * 1e-6  # uV to V
    raw0 = mne.io.RawArray(eeg0, info)
    raw0.filter(8, 30)

   
    # Schematic of the MI protocol
    x_all = np.linspace(10, 16, 3000)
    z_all = MI_stimulus_function(x_all)
    fig, ax = plt.subplots(5, 1, figsize=(6 / 2.54, 4.4 / 2.54), sharex=True, gridspec_kw={'hspace': 0.8})
    ax[0].plot(x_all, z_all, color='r')
    ax[0].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    index_chan = [1, 55]

    for i in range(1, 3):
        ax[i].plot(x_all[0:500], signal0[index_chan[i - 1], :500].T, color='cornflowerblue')
        ax[i].plot(x_all[500:2500], signal0[index_chan[i - 1], 500:2500].T, color='coral')
        ax[i].plot(x_all[2500:3000], signal0[index_chan[i - 1], 2500:].T, color='cornflowerblue')
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].spines[['right', 'top', 'bottom', 'left']].set_visible(False)

    ax[3].set_yticks([])
    ax[3].set_xticks([])
    ax[3].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ch_data0 = raw0.get_data(picks='C4')
    ax[4].plot(x_all[0:500], ch_data0[0, :500], color='cornflowerblue')
    ax[4].plot(x_all[500:2500], ch_data0[0, 500:2500], color='coral')
    ax[4].plot(x_all[2500:3000], ch_data0[0, 2500:], color='cornflowerblue')
    ax[4].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax[4].set_yticks([])
    ax[4].set_xticks([])

    ax[1].set_ylabel('Rpre')
    ax[2].set_ylabel('Rpost')
    ax[4].set_ylabel('C4')

    # Adding a time scale bar
    x_min, x_max = plt.xlim()  
    y_min, y_max = plt.ylim()
    scale_length = 500 * (x_max - x_min) / 4000   
    scale_label = '1 s'
    margin_x = (x_max - x_min) * 0.05
    scale_x_end = x_max - margin_x
    scale_x_start = scale_x_end - scale_length
    scale_y = y_min  
    plt.hlines(y=scale_y, xmin=scale_x_start, xmax=scale_x_end,
               color='black', linewidth=1)
    plt.text(scale_x_start + scale_length / 2, scale_y - (y_max - y_min) * 0.6,
             scale_label, ha='center', fontsize=8)
    plt.xlabel('MI')
    # plt.savefig('figure/MI_stimulation_draw.svg', dpi=300)
    plt.show()
    return 0


if __name__ == "__main__":
    # adjust figure format
    plt.ion()
    plt.rc('font', size=9, family='Arial', weight='normal')
    plt.rcParams['image.cmap'] = 'plasma'
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.labelweight'] = 'normal'
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['axes.titleweight'] = 'normal'
    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['svg.fonttype'] = 'none'

    # load data and model
    ds = Dataset('hcp')
    atlas = AutomatedAnatomicalParcellation2()
    cortex = atlas.names()

    # # Parameter fitting of the homogeneous model
    # corr_all_avg = []
    # for k_gl in np.arange(0, 8.0, 0.1):
    #     print(k_gl)
    #     model = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat, seed=4)
    #     duration = 10 * 60 * 1000  #  in ms, 10 min
    #     model.params['signalV'] = 0
    #     model.params['duration'] = duration  # simulation time duration
    #     model.params['sigma_ou'] = 0.03  # noise std
    #     model.params['K_gl'] = k_gl  # global coupling strength 
    #     model.run(chunkwise=False, bold=False)  # run simulation
    
    #     # firing rate to bold signal
    #     r = model.output
    #     dt_sim = model.params['dt']
    #     neural_drive = r.copy() 
    #     neural_drive[neural_drive < 0] = 0
    #     bold_model = BOLDModel(N=80, dt=dt_sim)
    #     bold_model.run(neural_drive)
    #     bold = bold_model.BOLD
    
    #     # calculate fc
    #     fc, fc_corr = fmri_fit(bold)
    #     corr_all_avg.append(np.mean(fc_corr))
    #     print(fc_corr)
    
    # fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    # ax.scatter(np.arange(0, 8.0, 0.1), corr_all_avg, color='royalblue', s=0.6)
    # ax.plot((np.argmax(corr_all_avg)) * 0.1, np.max(corr_all_avg), 's', color='red', fillstyle='none', ms=3.5, lw=0.8, mew=0.8)
    # ax.set(xlabel='Global coupling strength', ylabel='Mean correlation')
    # ax.spines[['top', 'right']].set_visible(False)
    # k_gl = np.argmax(corr_all_avg)*0.1   # initial optimal global coupling strength    3.2
    # max_corr = np.max(corr_all_avg)   # max correlation between the sFC and eFC   0.58
    # plt.savefig('figure/Global coupling strength_new.svg', dpi=300)
    # plt.show()
    

    # plot fc after fitting algorithm
    a = ds.Cmat
    b = ds.Dmat
    curr_cmat = np.load('data_load/model_data/curr_cmat.npy')

    model = model_para_set(curr_cmat, time_s=600)
    model.run(chunkwise=False, bold=False)  # run simulation
    
    # firing rate to bold signal
    r = model.output
    dt_sim = model.params['dt']
    neural_drive = r.copy()
    neural_drive[neural_drive < 0] = 0
    bold_model = BOLDModel(N=80, dt=dt_sim)
    bold_model.run(neural_drive)
    bold = bold_model.BOLD
    
    # calculate fc
    fc, fc_corr = fmri_fit(bold)
    # plot fc
    fig, axs = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    img1 = axs.imshow(fc)  # , cmap='rainbow'
    axs.set_xticks([0, 20, 40, 60])
    axs.set_xticklabels(['0', '20', '40', '60'])
    axs.set_yticks([0, 20, 40, 60])
    axs.set_yticklabels(['0', '20', '40', '60'])
    axs.set(xlabel='Region #', ylabel='Region #')
    axs.set_title('Simulated FC')
    clb = fig.colorbar(img1, ax=axs, shrink=0.5)
    clb.mappable.set_clim(vmin=-0, vmax=1)
    clb.ax.yaxis.set_major_formatter('{x:.1f}')
    # plt.savefig('figure/Simulated FC_final.svg', dpi=300)
    plt.show()

    # EEG generation
    model = model_para_set(curr_cmat, 20)  # Plotting a 10-second segment of the simulated EEG signals, the initial 10 seconds of data are discarded due to instability
    model.run(chunkwise=False, bold=False)
    eeg_plot(model)
    # plt.savefig('figure/eeg_signal.svg', dpi=300)

    # MI stimulation
    MI_stimulation_draw(curr_cmat)
    firingrate_P(curr_cmat)


    # fig7 c
    cmat_lesion = curr_cmat.copy()
    cmat_lesion[20, :] = 0  # set structural connectivity corresponding row and col to 0
    cmat_lesion[:, 20] = 0
    cmat_lesion[60, :] = 0  # set structural connectivity corresponding row and col to 0
    cmat_lesion[:, 60] = 0
    model = model_para_set(cmat_lesion, time_s=16)
    duration = model.params['duration']
    x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
    z = MI_stimulus_function(x)
    stimulus = np.zeros((80, duration * 10))
    stimulus[1, :] = z  # add external stimulus to region 1, 55(motor sensor correlated)
    stimulus[55, :] = z
    model.params["exc_ext"] = stimulus
    model.run(chunkwise=False, bold=False)
    sFC_lesion = plot_fc_band(model)  # corr_lesion
    fig, axs = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    img1 = axs.imshow(sFC_lesion)  # , cmap='rainbow'
    axs.set_xticks([0, 10, 20])
    axs.set_xticklabels(['0', '10', '20'])
    axs.set_yticks([0, 10, 20])
    axs.set_yticklabels(['0', '10', '20'])
    axs.set(xlabel='Channel #', ylabel='Channel #')
    axs.set_title('Lesioned FC')
    clb = fig.colorbar(img1, ax=axs, shrink=0.5)
    # clb.mappable.set_clim(vmin=-0, vmax=1)
    clb.ax.yaxis.set_major_formatter('{x:.1f}')
    clb.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    # plt.savefig('figure/Simulated FC_eeg.svg', dpi=300)
    plt.show()


    ## ***************************** localized lesion type ************************************** ##
    leaf_bc_all = []
    th_all = []
    ersp_0_all = []
    ersp_54_all = []
    ersp_0_ahand_all = []
    ersp_54_ahand_all = []
    ersp_0_uhand_all = []
    ersp_54_uhand_all = []

    ersp_c_ahand_all = []
    ersp_i_ahand_all = []
    ersp_c_uhand_all = []
    ersp_i_uhand_all = []

    # Randomly generate 20 sets of deletion sequences
    local_num = 20
    odd_numbers = [i for i in range(1, 80, 2)]  # Odd numbers within 80, right hemisphere serial numbers
    permutations = []
    random.seed(10)
    while len(permutations) < local_num:
        shuffled = odd_numbers.copy()
        random.shuffle(shuffled)  # Randomly sort the even_numbers
        if shuffled not in permutations:
            permutations.append(shuffled)

    for time in range(local_num):
        model = model_para_set(curr_cmat, time_s=16)
        duration = model.params['duration']

        x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
        z = MI_stimulus_function(x)
        stimulus = np.zeros((80, duration * 10))
        stimulus[1, :] = z  # add external stimulus to region 1, 55(motor sensor correlated)
        stimulus[55, :] = z

        # Unlesioned model
        model.params["exc_ext"] = stimulus
        model.run(chunkwise=False, bold=False)    # run simulation
        sFC_unlesion = plot_fc_band(model)  # calculate simulated functional connectivity (sFC)
        leaf_unlesion = compute_mst_leaf(sFC_unlesion) # calculate leaf number
       
        # delete node by distance order
        left_index = permutations[time]
        cmat_lesion = curr_cmat.copy()  # ds.Cmat.copy()
        leaf_lesion_bc = [leaf_unlesion]    # leaf number with lesioned nodes number increased

        # Remove the contralateral brain region nodes during motor imagery (Affected hand)
        for i in range(8): # lesion regions one by one according to distance order
            cmat_lesion[left_index[i], :] = 0    # set structural connectivity corresponding row and col to 0
            cmat_lesion[:, left_index[i]] = 0
            model = model_para_set(cmat_lesion, time_s=16)
            model.params["exc_ext"] = stimulus
            model.run(chunkwise=False, bold=False)
            sFC_lesion = plot_fc_band(model)
            leaf_lesion = compute_mst_leaf(sFC_lesion)
            leaf_lesion_bc.append(leaf_lesion)

        leaf_bc_all.append(leaf_lesion_bc)

    # localized lesion type result: lineplot for 20 sets of deletion sequences
    data = pd.DataFrame(columns=['Removed nodes number', 'Leaf number'])
    data['Removed nodes number'] = np.repeat(np.arange(9), local_num)
    data['Leaf number'] = np.reshape(np.array(leaf_bc_all), (9*local_num, 1), order='F')
    data['Lesion Extent (%)'] = (data['Removed nodes number'] / 40) * 100

    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    sns.lineplot(data=data, x='Lesion Extent (%)', y='Leaf number', errorbar="se", ax=ax, color='royalblue')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Damaged brain regions (%)', fontsize=10)
    ax.set_ylabel('Leaf number', fontsize=10)
    ax.tick_params(axis='both', which='major')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # plt.savefig('figure/leafnum_local_lesion_eeg.svg', dpi=300)


    ## *********************************** global lesion type *********************************** ##
    model = model_para_set(curr_cmat, time_s=16)
    duration = model.params['duration']
    x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
    z = MI_stimulus_function(x)
    stimulus = np.zeros((80, duration * 10))
    stimulus[1, :] = z  # add external stimulus to region 1, 55 (motor sensor correlated)
    stimulus[55, :] = z
    leaf_all = np.zeros(100)    # leaf number for lesioned MST by randomly removing node
    diameter_all = np.zeros(100)
    ecc_all = np.zeros(100)
    tree_hie_all = np.zeros(100)

    Eglo_all = np.zeros(100)    # global efficiency for lesioned MST by randomly removing node
    ERSP_0_all = np.zeros(100)
    ERSP_54_all = np.zeros(100)
    all_combinations = list(itertools.combinations(odd_numbers, 3))
    selected_combinations = random.sample(all_combinations, 100)   # Randomly select 100 non-repetitive combinations

    for i in range(len(leaf_all)):
        cmat_lesion = curr_cmat.copy()
        np.random.seed(i)
        ind = selected_combinations[i]
        cmat_lesion[ind, :] = 0 # random removed node
        cmat_lesion[:, ind] = 0
        model = model_para_set(cmat_lesion, time_s=16)
        model.params["exc_ext"] = stimulus
        model.run(chunkwise=False, bold=False)
        sFC_lesion = plot_fc_band(model)
        leaf_lesion = compute_mst_leaf(sFC_lesion)
        MST = nx.maximum_spanning_tree(nx.from_numpy_array(sFC_lesion), weight='weight')
        Eglo_all[i] = nx.global_efficiency(MST)
        leaf_all[i] = leaf_lesion

    data = pd.DataFrame(columns=['Global efficiency', 'Leaf number'])
    min_index = np.where(leaf_all == np.min(leaf_all))[0]
    sort_ind = np.argsort(Eglo_all) # using global efficiency to represent NIHSS score(low E_glo == high NIHSS)
    leaf_all_sort = leaf_all[sort_ind]
    data['Global efficiency'] = ['Low'] * 50 + ['High'] * 50
    data['Leaf number'] = leaf_all_sort

    # The comparison of MST leaf number between high (n=50) and low (n=50) global efficiency group
    violin_plot_leaf(data, x='Global efficiency', y='Leaf number', pairs=[('Low', 'High')])
    # plt.savefig('figure/leafnum_global_lesion_eeg.svg', dpi=300)

    # Representative MST subnetworks from lesioned SC  (stroke)
    cmat_lesion = curr_cmat.copy()
    ind = selected_combinations[min_index[0]]
    cmat_lesion[ind, :] = 0  # random removed node
    cmat_lesion[:, ind] = 0
    model = model_para_set(cmat_lesion, time_s=16)
    model.params["exc_ext"] = stimulus
    model.run(chunkwise=False, bold=False)
    sFC_lesion = plot_fc_band(model)
    leaf_lesion = compute_mst_leaf(sFC_lesion)
    fig1, fig2 = draw_mst_star_path(sFC_lesion)
    # fig1.savefig('figure/mst_star_lesion.svg', dpi=300)
    # fig2.savefig('figure/mst_path_lesion.svg', dpi=300)

    # Representative MST subnetworks from empirical SC  (healthy)
    model = model_para_set(curr_cmat, time_s=16)
    duration = model.params['duration']
    model.params["exc_ext"] = stimulus
    model.run(chunkwise=False, bold=False)
    fc = plot_fc_band(model)
    leaf_unlesion = compute_mst_leaf(fc)
    fig1, fig2 = draw_mst_star_path(fc)
    # fig1.savefig('figure/mst_star_healthy.svg', dpi=300)
    # fig2.savefig('figure/mst_path_healthy.svg', dpi=300)

    plt.ioff()
    plt.show()