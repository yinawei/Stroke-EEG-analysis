# %% 6. lesioned network simulation using Wilson-cowan equation

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
from matplotlib.ticker import MaxNLocator
from neurolib.utils.signal import RatesSignal
from statannotations.Annotator import Annotator
matplotlib.use("Qt5Agg")
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


def model_para_set(cmat, dmat, seed):
    """Create Wilson-cowan model and set parameters

    :param cmat: structral connect weight matrix
    :param dmat: tract length matrix (not used)
    :param seed: random seed to set differnt initial condition
    :return: WC model
    """
    model = WCModel(Cmat=cmat, Dmat=dmat, seed=seed)
    duration = 1 * 10 * 1000    # 10s   1 * 10 * 1000
    model.params['signalV'] = 0
    model.params['duration'] = duration # simulation time duration
    model.params['sigma_ou'] = 0.03  # noise std
    model.params['K_gl'] = 1.6

    return model


def plot_fc_band0(model):
    """Using filted rate signal to calculate simulated functional connnectivity

    :param model: WC model
    :return: simulated fc matrix, correlations between sFC and eFC
    """
    # compute sFC and correlation scores
    fc = func.fc(model.BOLD.BOLD)
    fcemp = np.mean(np.array(ds.FCs), axis=0)
    score = func.matrix_correlation(fc, fcemp)

    return fc.data, score

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
    eeg = eeg_generate(fr_exc, fr_inh)
    eeg = eeg * 1e-6  # 单位转成V，mne的默认单位
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=1000,  # frequency
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
    scale_length = 1000 * (x_max - x_min) / 10000
    scale_label = '1 s'
    scale_x_end = x_max - scale_length / 2
    scale_x_start = scale_x_end - scale_length
    scale_y = y_min - (y_max - y_min) * 0.3
    plt.hlines(y=scale_y, xmin=scale_x_start, xmax=scale_x_end,
               color='black', linewidth=1)
    plt.text(scale_x_start + scale_length / 2, scale_y - (y_max - y_min) * 1.4,
             scale_label, ha='center', fontsize=8)
    plt.show()

    return 0


def plot_fc_band(model):
    """Using filted rate signal to calculate simulated functional connnectivity

    :param model: WC model
    :return: simulated fc matrix, correlations between sFC and eFC
    """
    # compute sFC and correlation scores
    fr = RatesSignal.from_model_output(model, group="", time_in_ms=True)
    fr_exc = fr['exc'].data.data[:, :]
    fr_inh = fr['inh'].data.data[:, :]
    eeg = eeg_generate(fr_exc, fr_inh)
    eeg = eeg * 1e-6  # 单位转成V，mne的默认单位
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=1000,  # frequency
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

    raw_data = raw.get_data()
    n_trials = 1
    manual_epochs_data = raw_data.reshape(-1, raw_data.shape[0], int(raw_data.shape[1]/n_trials))  # (n_trials, 64, n_samples)
    time_series = np.transpose(manual_epochs_data, (2, 0, 1))  # (n_samples, n_trials, 64)
    multitaper = Multitaper(time_series, sampling_frequency=1000)
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
    bc = nx.betweenness_centrality(MST)
    bc_value = [bc[x] for x in bc.keys()]
    bc_sort = np.argsort(np.array(bc_value))    # ascend order

    return len(leaf_nodes), bc_sort[::-1], bc_value

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
        if MST.degree(node1) == 2 and MST.degree(node2) == 2:
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

    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    plot_connectome(mst_fc_star, coords, node_size=node_size_star, display_mode='z',
                    edge_kwargs=edge_prop, axes=ax, node_color=node_color_star, alpha=0.2, colorbar=False)
    ax.set_title('Starlike subnetwork')
    edge_prop['c'] = b

    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    plot_connectome(mst_fc_path, coords, node_size=node_size_path, display_mode='z',
                    edge_kwargs=edge_prop, axes=ax, node_color=node_color_path, alpha=0.2, colorbar=False)
    ax.set_title('Pathlike subnetwork')

def violin_plot_single(data, x, y, pairs):
    """
    Violin plot for MI and no MI
    :param data: dataframe
    :param x: NIHSS group
    :param y: Leaf number
    :param pairs: paired group to statistical analysis
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    sns.stripplot(data=data, x=x, y=y, ax=ax, legend=False, size=2.5, jitter=0.13, palette='muted',
                  facecolors='none', order=['No MI', 'MI']) # palette=colors
    for dots in ax.collections:   # remove facecolor and set edgecolor
        facecolors = dots.get_facecolors()
        dots.set_edgecolors(facecolors.copy())
        dots.set_facecolors('none')
        dots.set_linewidth(1)

    sns.violinplot(data=data, x=x, y=y, ax=ax, palette='muted', legend='brief', order=['No MI', 'MI'], fill=False, width=0.4)  # width=0.4, , dodge=True
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(ylabel='Power (μV²/Hz)')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    s1, p1 = scipy.stats.shapiro(data[data[x] == 'No MI'][y])
    s2, p2 = scipy.stats.shapiro(data[data[x] == 'MI'][y])
    if p1 < 0.05 or p2 < 0.05:  # nonnormal distribution
        test_method = 'Mann-Whitney'
    else:
        test_method = 't-test_ind'
    annotator = Annotator(ax, data=data, x=x, y=y, pairs=pairs)
    annotator.configure(test=test_method, hide_non_significant=False, loc='outside')
    annotator.apply_test()
    annotator.annotate()

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
    annotator.annotate()

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
        sfreq=1000,  # frequency
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

    duration = 1 * 10 * 1000
    x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
    z = MI_stimulus_function(x)
    stimulus = np.zeros((80, duration * 10))
    stimulus[0, :] = z  # add external stimulus to region 0, 54 (motor sensor correlated)
    stimulus[54, :] = z

    power_0 = []
    power_0MI = []

    for seed in range(1, 21):
        # No MI model
        model0 = model_para_set(SC, ds.Dmat, seed)
        model0.run(bold=True)
        fr0 = RatesSignal.from_model_output(model0, group="", time_in_ms=True)
        eeg0 = eeg_generate(fr0['exc'].data.data[:, :], fr0['inh'].data.data[:, :])
        eeg0 = eeg0 * 1e-6  # uV to V
        raw0 = mne.io.RawArray(eeg0, info)
        spectrum0 = raw0.compute_psd(method='welch', fmin=8, fmax=30, n_fft=1000 * 2)
        psd0, freqs = spectrum0.get_data(return_freqs=True)
        power0 = np.mean(psd0, axis=1)[13]
        power0 = power0 * 1e12  # uV/Hz

        # MI model
        model = model_para_set(SC, ds.Dmat, seed)
        model.params["exc_ext"] = stimulus
        model.run(bold=True)
        fr = RatesSignal.from_model_output(model, group="", time_in_ms=True)
        eeg1 = eeg_generate(fr['exc'].data.data[:, :], fr['inh'].data.data[:, :])
        eeg1 = eeg1 * 1e-6
        raw1 = mne.io.RawArray(eeg1, info)
        spectrum = raw1.compute_psd(method='welch', fmin=8, fmax=30, n_fft=1000 * 2)
        psd, freqs = spectrum.get_data(return_freqs=True)
        power = np.mean(psd, axis=1)[13]
        power = power * 1e12

        power_0.append(power0)
        power_0MI.append(power)

    # Plot violin plots comparing motor imagery (MI) simulations before and after
    data = {
        'Power': np.concatenate([power_0, power_0MI]),
        'Electrode Channel C3': ['No MI'] * len(power_0) + ['MI'] * len(power_0MI)
    }
    df = pd.DataFrame(data)
    violin_plot_single(df, x='Electrode Channel C3', y='Power', pairs=[('No MI', 'MI')])

    return 0

def simulate_k_gl(k_gl, SC, Dmat, duration):
    print(k_gl)
    model = WCModel(Cmat=SC, Dmat=Dmat, seed=4)
    model.params['signalV'] = 0
    model.params['duration'] = duration
    model.params['sigma_ou'] = 0.03
    model.params['K_gl'] = k_gl
    model.run(chunkwise=True, bold=True)
    sFC, corr = plot_fc_band0(model)
    return sFC, corr

def G_calculate(ds_sc, ds_dmat):
    """
    Obtain the optimal global coupling strength (G) using the iterative-fitting algorithm
    """
    # Determine the initial G and generate the corresponding sFC
    duration = 10 * 60 * 1000  # in ms, 10 min

    sFC_all = []
    corr_all = []
    k_gl_values = np.arange(4.5, 5.0, 0.1)   # initial optimal global coupling strength G = 4.7

    for k_gl in k_gl_values:
        sFC, corr = simulate_k_gl(k_gl, ds_sc, ds_dmat, duration) # 解包元组
        sFC_all.append(sFC)
        corr_all.append(corr)

    k_gl = 4.5 + np.argmax(corr_all)*0.1   # initial optimal global coupling strength G
    max_corr_0 = np.max(corr_all)  # initial max correlation
    print('initial optimal global coupling strength:', k_gl, ',', max_corr_0)

    ## Optimize the fitting effect by iteratively updating the value of SC
    eSC = ds_sc.copy()
    eFC = np.mean(np.array(ds.FCs), axis=0)
    T = 1    # initial tolerance level
    sFC = sFC_all[np.argmax(corr_all)]  # initial sFC

    best_k_gl_all = []
    max_corr_all = []
    sSC_all = []
    number_time = 0
    while T >= 0:
        print(T)
        # Update eSC
        for i in range(eFC.shape[0]):
            for j in range(eFC.shape[0]):
                if np.abs(eFC[i,j] - sFC[i,j]) > T:
                    if eFC[i,j] > 0:
                        eSC[i,j] = 0.15 * eFC[i,j]
                    else:
                        eSC[i,j] = 0 if eSC[i,j] == 0 else 0.0005
        sSC_all.append(eSC)
        # Update G
        corr_all = []
        last_kgl = k_gl
        k_gl_values = np.arange(last_kgl - 1, last_kgl + 1, 0.1)
        for k_gl in k_gl_values:
            sFC, corr = simulate_k_gl(k_gl, eSC, ds_dmat, duration)
            sFC_all.append(sFC)
            corr_all.append(corr)

        k_gl = last_kgl - 1 + np.argmax(corr_all) * 0.1
        max_corr = np.max(corr_all)
        print('current optimal global coupling strength:', k_gl, max_corr)

        # Update sFC
        sFC = sFC_all[np.argmax(corr_all)]

        # Update T
        T = T - 0.025

        # save data
        best_k_gl_all.append(k_gl)
        max_corr_all.append(max_corr)
        if max_corr == np.max(max_corr_all):
            best_k_gl = k_gl
            best_sSC = eSC
            best_sFC = sFC

        number_time += 1

    return best_k_gl_all, max_corr_all, best_sSC, best_k_gl, best_sFC, sSC_all

def eeg_generate(fr_exc, fr_inh):
    '''
    Generate EEG from the neuronal population activity results of the original model
    '''

    ## Convert the neuronal population firing rate into postsynaptic current
    # Downsample neuronal activity signals to 1000 Hz
    exc_downsample = scipy.signal.resample(fr_exc,
                                           int(fr_exc.shape[1] * 1000 / 10000), axis=1)
    inh_downsample = scipy.signal.resample(fr_inh,
                                           int(fr_inh.shape[1] * 1000 / 10000), axis=1)
    # Parameter setting
    tau_rise_ampa, tau_decay_ampa = 0.002, 0.010
    tau_rise_gabaa, tau_decay_gabaa = 0.001, 0.020
    I_peak_ampa = 1e-10
    I_peak_gabaa = 0.6e-10

    # Convert neuronal signals into postsynaptic currents
    epsc = generate_synaptic_current_batched(exc_downsample, tau_rise_ampa, tau_decay_ampa, I_peak_ampa)
    ipsc = generate_synaptic_current_batched(inh_downsample, tau_rise_gabaa, tau_decay_gabaa, I_peak_gabaa)

    # Define the excitatory/inhibitory weight ratio
    weight_e = 1
    weight_i = 0
    current_total = weight_e * epsc + weight_i * ipsc
    current_total_nA = current_total * 1e9
    current_total_nA_left = current_total_nA[::2, :]
    current_total_nA_right = current_total_nA[1::2, :]
    current_total_nA = np.concatenate((current_total_nA_left, current_total_nA_right), axis=0)

    # Calculate scalp potential
    fwd = mne.read_forward_solution("data_load/model_data/custom_forward.fif", verbose=True)
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=True)
    leadfield = fwd_fixed['sol']['data']
    eeg = leadfield @ current_total_nA   # EEG(μV)
    return eeg


def generate_synaptic_current_batched(rates, tau_rise, tau_decay, I_peak, sfreq=1000, batch_size=1000):
    """Generate postsynaptic currents in blocks"""
    rates = rates * 10
    n_rois, n_samples = rates.shape
    dt = 1 / sfreq
    kernel_length = int(5 * tau_decay * sfreq)

    # Calculate the kernel function area
    A_kernel = tau_decay - tau_rise

    # Define the synaptic response function (normalized)
    t_kernel = np.arange(0, kernel_length) * dt
    kernel = np.exp(-t_kernel / tau_decay) - np.exp(-t_kernel / tau_rise)
    kernel /= kernel.max()

    # Block-wise convolution
    current = np.zeros_like(rates)
    for i in range(0, n_samples, batch_size):
        batch = rates[:, i:i + batch_size]
        batch_padded = np.pad(batch, ((0, 0), (kernel_length, 0)), mode='edge')
        conv = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), 1, batch_padded)
        current[:, i:i + batch_size] = conv[:, :batch_size]

    # Convert to physical current unit (A)
    current *= I_peak * A_kernel
    return current


def bandpass_filter(signal, fs):
    """
    Band-pass filter (8-30Hz)
    """
    nyq = 0.5 * fs
    low, high = 8 / nyq, 30 / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal)


def MI_stimulus_function(x):
    """
    Function for simulating motor imagery
    """
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] <= 5:
            z[i] = -1.2 / (1 + np.exp(-5 * (x[i] - 1)))   # -1.3
        else:
            z[i] = -1.2 / (1 + np.exp(5 * (x[i] - 9)))   # 1.5

    return z


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

    # initial parameter optimization
    corr_all_avg = []
    for k_gl in np.arange(0, 8.0, 0.1):
        print(k_gl)
        model = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat, seed=4)
        duration = 10 * 60 * 1000  #  in ms, 10 min
        model.params['signalV'] = 0
        model.params['duration'] = duration  # simulation time duration
        model.params['sigma_ou'] = 0.03  # noise std
        model.params['K_gl'] = k_gl  # global coupling strength
        model.run(chunkwise=True, bold=True)  # run simulation
        sFC_unlesion, corr_unlesion = plot_fc_band0(model)
        corr_all_avg.append(np.mean(corr_unlesion))
    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    ax.scatter(np.arange(0, 8.0, 0.1), corr_all_avg, color='royalblue', s=0.6)
    ax.plot((np.argmax(corr_all_avg)) * 0.1, np.max(corr_all_avg), 's', color='red', fillstyle='none', ms=3.5, lw=0.8, mew=0.8)
    ax.set(xlabel='Global coupling strength', ylabel='Mean correlation')
    ax.spines[['top', 'right']].set_visible(False)
    k_gl = np.argmax(corr_all_avg)*0.1   # initial optimal global coupling strength
    max_corr = np.max(corr_all_avg)   # max correlation between the sFC and eFC

    # calculate the optimal global coupling strength (G) using the iterative-fitting algorithm
    a = ds.Cmat
    b = ds.Dmat
    # best_k_gl_all, max_corr_all, best_sSC, best_k_gl, best_sFC, sSC_all = G_calculate(a,b)    # the iterative-fitting algorithm
    # best_k_gl_all = np.array(best_k_gl_all)
    # max_corr_all = np.array(max_corr_all)
    # np.save('data_load/model_data/best_sSC.npy', best_sSC)
    # np.save('data_load/model_data/max_corr_all.npy', max_corr_all)
    # print(best_k_gl)
    best_sSC = np.load('data_load/model_data/best_sSC.npy')
    max_corr_all = np.load('data_load/model_data/max_corr_all.npy')

    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    ax.scatter(np.arange(0, 1, 0.025), max_corr_all[::-1], color='royalblue', s=1.2)
    plt.plot(np.arange(0, 1, 0.025), max_corr_all[::-1], color='royalblue', linewidth=1, alpha=0.9)
    ax.plot((np.argmax(max_corr_all[::-1])) * 0.025, np.max(max_corr_all), 's', color='red', fillstyle='none', ms=3.5, lw=0.8, mew=0.8)
    ax.set(xlabel='Tolerance (T)', ylabel='Correlation')
    ax.spines[['top', 'right']].set_visible(False)


    # EEG generation
    duration = 1 * 10 * 1000  # 10s
    model = model_para_set(best_sSC, ds.Dmat, 4)
    model.params['duration'] = duration
    model.run(bold=True)
    eeg_plot(model)


    # MI stimulation
    firingrate_P(best_sSC)


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

    roi_L = [index for index, item in enumerate(cortex) if item.endswith('L')]
    roi_R = [index for index, item in enumerate(cortex) if item.endswith('R')]

    # Randomly generate 20 sets of deletion sequences
    local_num = 20
    even_numbers = [i for i in range(0, 80, 2)]  # Even numbers within 80, left hemisphere serial numbers
    permutations = []
    random.seed(150)
    while len(permutations) < local_num:
        shuffled = even_numbers.copy()
        random.shuffle(shuffled)  # Randomly sort the even_numbers
        if shuffled not in permutations:
            permutations.append(shuffled)

    for time in range(local_num):
        seed = 4
        model = model_para_set(best_sSC, ds.Dmat, seed)
        duration = model.params['duration']

        x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
        z = MI_stimulus_function(x)
        stimulus = np.zeros((80, duration * 10))
        stimulus[0, :] = z  # add external stimulus to region 0, 54(motor sensor correlated)
        stimulus[54, :] = z

        # unlesioned network simulation
        model.params["exc_ext"] = stimulus
        model.run(bold=True)    # run simulation
        sFC_unlesion = plot_fc_band(model)  # calculate simulated functional connectivity (sFC)
        leaf_unlesion, bc, bc_value = compute_mst_leaf(sFC_unlesion) # calculate leaf number

        # delete node by distance order
        roi_coord = atlas.aal2_centers[:, atlas.cortex]  # Retrieve the central coordinates of all cortical brain regions in the AAL2 parcellation from the atlas object
        left_index = permutations[time]
        cmat_lesion = best_sSC.copy()  # ds.Cmat.copy()
        leaf_lesion_bc = [leaf_unlesion]    # leaf number with lesioned nodes number increased

        # Remove the contralateral brain region nodes during motor imagery (Affected hand)
        for i in range(40): # lesion regions one by one according to distance order
            cmat_lesion[left_index[i], :] = 0    # set structural connectivity corresponding row and col to 0
            cmat_lesion[:, left_index[i]] = 0
            model = model_para_set(cmat_lesion, ds.Dmat, seed)
            model.params["exc_ext"] = stimulus
            model.run(bold=True)
            sFC_lesion = plot_fc_band(model)   # , corr_lesion
            leaf_lesion, bc_lesion, bc_value_lesion = compute_mst_leaf(sFC_lesion)
            leaf_lesion_bc.append(leaf_lesion)

        leaf_bc_all.append(leaf_lesion_bc)

    # localized lesion type result: lineplot for 20 sets of deletion sequences
    data = pd.DataFrame(columns=['Removed nodes number', 'Leaf number'])
    data['Removed nodes number'] = np.repeat(np.arange(41), local_num)
    data['Leaf number'] = np.reshape(np.array(leaf_bc_all), (41*local_num, 1), order='F')

    fig, ax = plt.subplots(1, 1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    sns.lineplot(data, x='Removed nodes number', y='Leaf number', errorbar="se", ax=ax, color='royalblue')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Removed nodes number', fontsize=10)
    ax.set_ylabel('Leaf number', fontsize=10)
    ax.tick_params(axis='both', which='major')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.savefig('leafnum_local_lesion_eeg.svg', dpi=300)

    ## *********************************** global lesion type *********************************** ##
    duration = 1 * 10 * 1000
    x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
    z = MI_stimulus_function(x)
    stimulus = np.zeros((80, duration * 10))
    stimulus[0, :] = z  # add external stimulus to region 0, 54 (motor sensor correlated)
    stimulus[54, :] = z
    seed = 4
    leaf_all = np.zeros(100)    # leaf number for lesioned MST by randomly removing node
    diameter_all = np.zeros(100)
    ecc_all = np.zeros(100)
    tree_hie_all = np.zeros(100)

    Eglo_all = np.zeros(100)    # global efficiency for lesioned MST by randomly removing node
    ERSP_0_all = np.zeros(100)
    ERSP_54_all = np.zeros(100)
    even_numbers = [i for i in range(0, 80, 2)]
    all_combinations = list(itertools.combinations(even_numbers, 3))
    random.seed(150)   # 42
    selected_combinations = random.sample(all_combinations, 100)   # Randomly select 100 non-repetitive combinations

    for i in range(len(leaf_all)):
        cmat_lesion = best_sSC.copy()
        np.random.seed(i)
        ind = selected_combinations[i]
        cmat_lesion[ind, :] = 0 # random removed node
        cmat_lesion[:, ind] = 0
        model = model_para_set(cmat_lesion, ds.Dmat, seed)
        model.params["exc_ext"] = stimulus
        model.run(bold=True)
        sFC_lesion = plot_fc_band(model)
        leaf_lesion, bc_lesion, bv_value_lesion = compute_mst_leaf(sFC_lesion)
        MST = nx.maximum_spanning_tree(nx.from_numpy_array(sFC_lesion), weight='weight')
        Eglo_all[i] = nx.global_efficiency(MST)
        leaf_all[i] = leaf_lesion

    data = pd.DataFrame(columns=['Global efficiency', 'Leaf number'])
    min_index = np.where(leaf_all == np.min(leaf_all))[0]   # min_index = 47
    sort_ind = np.argsort(Eglo_all) # using global efficiency to represent NIHSS score(low E_glo == high NIHSS)
    leaf_all_sort = leaf_all[sort_ind]
    data['Global efficiency'] = ['Low'] * 50 + ['High'] * 50
    data['Leaf number'] = leaf_all_sort

    # The comparison of MST leaf number between high (n=50) and low (n=50) global efficiency group
    violin_plot_leaf(data, x='Global efficiency', y='Leaf number', pairs=[('Low', 'High')])
    plt.savefig('leafnum_global_lesion_eeg.svg', dpi=300)

    # Representative MST subnetworks from lesioned SC  (stroke)
    cmat_lesion = best_sSC.copy()
    ind = selected_combinations[min_index[0]]
    cmat_lesion[ind, :] = 0  # random removed node
    cmat_lesion[:, ind] = 0
    model = model_para_set(cmat_lesion, ds.Dmat, seed)
    model.params["exc_ext"] = stimulus
    model.run(bold=True)
    model0 = model_para_set(cmat_lesion, ds.Dmat, seed)
    model0.run(bold=True)
    sFC_lesion = plot_fc_band(model)
    leaf_lesion, bc_lesion, bv_value_lesion = compute_mst_leaf(sFC_lesion)
    draw_mst_star_path(sFC_lesion)

    # Representative MST subnetworks from empirical SC  (healthy)
    model = model_para_set(best_sSC, ds.Dmat, 4)
    duration = model.params['duration']
    x = np.arange(0.0001, duration / 1000 + 0.0001, 0.0001)
    z = MI_stimulus_function(x)
    stimulus = np.zeros((80, duration * 10))
    stimulus[0, :] = z  # add external stimulus to region 0, 54(motor sensor correlated)
    stimulus[54, :] = z
    model.params["exc_ext"] = stimulus
    model.run(bold=True)
    fc = plot_fc_band(model)
    leaf_lesion, bc_lesion, bv_value_lesion = compute_mst_leaf(fc)
    draw_mst_star_path(fc)

    plt.ioff()
    plt.show()

