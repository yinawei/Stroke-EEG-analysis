# %% 3. Figshare stroke data process with functional connectivity method
# % 0-2s:prepare, 2-6s:MI, 6-8s:break
# % labels: 1-left motor attempt, 2-right motor attemptimport matplotlib

import numpy as np
import matplotlib
import mne
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
import scipy.io as sio
import matplotlib.colors as colors
from spectral_connectivity import Multitaper, Connectivity
import os
from nilearn.plotting import plot_connectome, plot_matrix

# matplotlib.use("Qt5Agg")

def epochs_make(eeg, labels, trigger):
    """Convert numpy eeg data to mne epoch

    :param eeg: EEG data array, [trials x channels x samples]
    :param labels: left hand:1, right hand:2
    :param trigger: motor imager start time
    :return: mne epoch data
    """
    info = mne.create_info(
        ch_names=["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                  "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"],
        ch_types="eeg",  # channel type
        sfreq=500  # frequency
    )
    location = {'FP1':[-0.309, 0.95, -0.0349], 'FP2':[0.309, 0.95, -0.0349], 'Fz':[-0, 0.719, 0.695], 'F3':[-0.545, 0.673, 0.5],
                'F4':[0.545, 0.673, 0.5], 'F7':[-0.809, 0.587, -0.0349], 'F8':[0.809, 0.587, -0.0349], 'FCz':[-0, 0.391, 0.921],
                'FC3': [-0.676, 0.36, 0.643], 'FC4': [0.676, 0.36, 0.643], 'FT7': [-0.95, 0.309, 0.0349], 'FT8': [0.95, 0.309, -0.0349],
                'Cz': [0, 0, 1], 'C3': [-0.719, 0, 0.695], 'C4': [0.719, 0, 0.695], 'T3': [-0.999, 6.12e-17, -0.0349],
                'T4': [0.999, 6.12e-17, -0.0349], 'CP3': [-0.676, -0.36, 0.643], 'CP4': [0.676, -0.36, 0.643], 'TP7': [-0.95, -0.309, -0.0349],
                'TP8': [0.95, -0.309, -0.0349], 'Pz': [-8.81e-17, -0.719, 0.695], 'P3': [-0.545, -0.673, 0.5], 'P4': [0.545, -0.673, 0.5],
                'T5': [-0.809, -0.587, -0.0349], 'T6': [0.809, -0.587, -0.0349], 'Oz': [-1.22e-16, -0.999, -0.0349], 'O1': [-0.309, -0.95, -0.0349],
                'O2': [0.309, -0.95, -0.0349]}

    scale_factor = 0.09
    location = {ch: [x * scale_factor, y * scale_factor, z * scale_factor] for ch, (x, y, z) in location.items()}
    montage = mne.channels.make_dig_montage(location)
    info.set_montage(montage)
    trials = eeg.shape[0]
    marks = np.zeros(trials)

    for i in range(trials):
        if i == 0:
            raw_data = np.squeeze(eeg[0, :, :])
        else:
            raw_data = np.concatenate((raw_data, np.squeeze(eeg[i, :, :])), axis=-1)
        marks[i] = trigger[i] + 4000*i  # MI onset
    raw = mne.io.RawArray(raw_data, info)

    events = np.zeros((trials, 3), dtype='int')
    events[:, 0] = marks
    events[:, -1] = labels
    event_id = dict(left_hand=1, right_hand=2)
    tmin, tmax = -2., 4.999
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)

    return epochs


def fc_plot_spectral(epochs, low, high, fs, t0, t1):
    """Plot functional connectivity graph in specific band using spectral_connectivity package

    :param epochs: mne epoch data
    :param low: low freq
    :param high: high freq
    :param fs: sampling freq
    :param t0: MI start time
    :param t1: MI end time
    :return: functional connectivity matrix
    """

    # calculate connectivity using spectral domain method
    epochs_permute = epochs.transpose(2, 0, 1)
    multitaper = Multitaper(epochs_permute[t0:t1, :, :], sampling_frequency=fs)

    connectivity = Connectivity.from_multitaper(multitaper, expectation_type="time_trials_tapers", blocks=1)
    fc = connectivity.imaginary_coherence()  # method: imagery coherence
    fc = np.abs(fc)
    f_range = np.where(np.logical_and(connectivity.frequencies >= low, connectivity.frequencies <= high))[0]
    connect = np.squeeze(np.mean(fc[f_range[0]:f_range[-1] + 1, :, :], axis=0))

    # figure 4 left column for paper
    fig, ax = plt.subplots(1, 1, figsize=(5.5 / 2.54, 5.5 / 2.54), layout='constrained')
    colormap = 'Blues'
    labels = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
              "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
    pm = plot_matrix(connect, colorbar=False, labels=labels, cmap=colormap, axes=ax)
    ax.set_yticklabels([item.get_text() for item in ax.get_yticklabels()], rotation=0)
    ax.set_xticklabels([item.get_text() for item in ax.get_xticklabels()], rotation=90)

    fig, ax = plt.subplots(1, 1, figsize=(6.2 / 2.54, 5.8 / 2.54), layout='constrained')
    coords = np.array([[-0.309, 0.95, -0.0349], [0.309, 0.95, -0.0349], [-0, 0.719, 0.695], [-0.445, 0.673, 0.5],
                       [0.445, 0.673, 0.5], [-0.809, 0.587, -0.0349], [0.809, 0.587, -0.0349], [-0, 0.391, 0.921],
                       [-0.576, 0.36, 0.643], [0.576, 0.36, 0.643], [-0.95, 0.309, 0.0349], [0.95, 0.309, -0.0349],
                       [0, 0, 1], [-0.619, 0, 0.695], [0.619, 0, 0.695], [-0.999, 6.12e-17, -0.0349],
                       [0.999, 6.12e-17, -0.0349], [-0.576, -0.36, 0.643], [0.576, -0.36, 0.643],
                       [-0.95, -0.309, -0.0349],
                       [0.95, -0.309, -0.0349], [-8.81e-17, -0.719, 0.695], [-0.445, -0.673, 0.5], [0.445, -0.673, 0.5],
                       [-0.809, -0.587, -0.0349], [0.809, -0.587, -0.0349], [-1.22e-16, -0.999, -0.0349],
                       [-0.309, -0.95, -0.0349],
                       [0.309, -0.95, -0.0349]])
    coords = coords * 60
    coords[:,1] = coords[:,1] * 1.4 - 16
    coords[:, 0] = coords[:, 0] + 2
    pc = plot_connectome(connect, coords, edge_threshold='50%', edge_vmax=np.max(connect), node_size=20,
                         node_color='cornflowerblue',
                         edge_vmin=np.percentile(connect, 50), axes=ax, display_mode='z', edge_cmap=colormap,
                         colorbar=False, alpha=0.2)
    fig.colorbar(pm, ax=ax, shrink=0.6, pad=0.1)

    return 0

def calculate_fc_spectral(epochs, low, high, fs, t0, t1):
    """calculate functional connectivity graph in specific band using spectral_connectivity package

    :param epochs: mne epoch data
    :param low: low freq
    :param high: high freq
    :param fs: sampling freq
    :param t0: MI start time
    :param t1: MI end time
    :return: functional connectivity matrix
    """

    # calculate connectivity using spectral domain method
    epochs_permute = epochs.transpose(2, 0, 1)
    multitaper = Multitaper(epochs_permute[t0:t1, :, :], sampling_frequency=fs)

    connectivity = Connectivity.from_multitaper(multitaper, expectation_type="time_trials_tapers", blocks=1)
    fc = connectivity.imaginary_coherence()  # method: imagery coherence
    fc = np.abs(fc)
    f_range = np.where(np.logical_and(connectivity.frequencies >= low, connectivity.frequencies <= high))[0]
    connect = np.squeeze(np.mean(fc[f_range[0]:f_range[-1] + 1, :, :], axis=0))

    return connect

if __name__ == "__main__":
    plt.ion()  # if not showing figure: plt.ioff()
    # adjust figure format
    plt.rc('font', size=9, family='Arial', weight='normal')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.labelweight'] = 'normal'
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['axes.titleweight'] = 'normal'
    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['svg.fonttype'] = 'none'

    # result variable of  functional connectivity: 50 subjects, 29 channels, 3 frequency bands
    connect_left_all = np.zeros((50, 29, 29, 3))
    connect_right_all = np.zeros((50, 29, 29, 3))
    connect_left_ll_all = np.zeros((50, 29, 29, 3))
    connect_left_lr_all = np.zeros((50, 29, 29, 3))
    connect_left_rr_all = np.zeros((50, 29, 29, 3))
    connect_right_ll_all = np.zeros((50, 29, 29, 3))
    connect_right_lr_all = np.zeros((50, 29, 29, 3))
    connect_right_rr_all = np.zeros((50, 29, 29, 3))

    for s in range(1, 51):  # s:subject num(1-50)
        # Enable the following two lines for plotting; comment for fc calculation
        # s = 43   # Mild impairment (NIHSS=2), Subject 43
        # s = 22   # Severe impairment (NIHSS=11), Subject 22

        print(s)
        if s >= 10:
            data = sio.loadmat('data_load/data_clean/sub-' + str(s) + '_task-motor-imagery_eeg.mat')   # data preprocessed using EEGLAB
        else:
            data = sio.loadmat('data_load/data_clean/sub-0' + str(s) + '_task-motor-imagery_eeg.mat')
        data = data['EEG_save']
        eeg_data = data['eeg'][0, 0]
        epochs = eeg_data['rawdata'][0, 0]
        labels = np.squeeze(eeg_data['label'][0, 0])
        trigger = np.squeeze(np.where(epochs[:, -1, :] == 2))[1, :]  # the start time point of each MI Task
        epochs_train = epochs_make(epochs[:, 0:29, :], labels, trigger)

        fs = 500
        epochs_left = epochs_train.get_data()
        tmp = fft(epochs_left[:, :, 2 * fs:6 * fs], axis=-1)  # [0,4]s
        tmp1 = np.array(np.abs(tmp) / 1000)
        freq_x = fs * np.array(range(4 * fs)) / (4 * fs)

        # plot fc, figure 4 left (if not need, comment it)
        # fc_plot_spectral(epochs_train['left_hand'].get_data(), 14, 20, fs, 3 * fs, 5 * fs)

        # functional connectivity: MI for left and right hand, 3 frequency bands
        connect_freq1_l = calculate_fc_spectral(epochs_train['left_hand'].get_data(), 8, 13, fs, 3 * fs, 5 * fs)  # [1,3]s
        connect_freq1_r = calculate_fc_spectral(epochs_train['right_hand'].get_data(), 8, 13, fs, 3 * fs, 5 * fs)
        connect_freq2_l = calculate_fc_spectral(epochs_train['left_hand'].get_data(), 14, 20, fs, 3 * fs, 5 * fs)
        connect_freq2_r = calculate_fc_spectral(epochs_train['right_hand'].get_data(), 14, 20, fs, 3 * fs, 5 * fs)
        connect_freq3_l = calculate_fc_spectral(epochs_train['left_hand'].get_data(), 21, 29, fs, 3 * fs, 5 * fs)
        connect_freq3_r = calculate_fc_spectral(epochs_train['right_hand'].get_data(), 21, 29, fs, 3 * fs, 5 * fs)

        connect_left = np.stack((connect_freq1_l, connect_freq2_l, connect_freq3_l), axis=-1)
        connect_right = np.stack((connect_freq1_r, connect_freq2_r, connect_freq3_r), axis=-1)
        connect_left_all[s - 1, :, :, :] = connect_left
        connect_right_all[s - 1, :, :, :] = connect_right

    # save fc data (further used to calculate MST)
    save_dir = "data_load/ImCoh_data/"
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir + 'imcoh_left.npy', connect_left_all)
    np.save(save_dir + 'imcoh_right.npy', connect_right_all)

    plt.ioff()
    plt.show()
