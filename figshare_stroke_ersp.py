# %% 1. Figshare stroke data process with time-frequency method
# % 0-2s:prepare, 2-6s:MI, 6-8s:break
# % labels: 1-left motor attempt, 2-right motor attempt
import os
import matplotlib
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet, tfr_array_morlet, tfr_multitaper
from matplotlib.colors import Normalize, TwoSlopeNorm
import scipy.io as sio
import matplotlib.colors as colors
matplotlib.use("Qt5Agg")


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
        marks[i] = trigger[i] + 4000*i  # the start time of each MI Task
    raw = mne.io.RawArray(raw_data, info)

    events = np.zeros((trials, 3), dtype='int')
    events[:, 0] = marks
    events[:, -1] = labels
    event_id = dict(left_hand=1, right_hand=2)
    tmin, tmax = -2., 4.999
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)

    return epochs

# baseline correct: logratio
def ersp_plot_logratio(epochs, baseline, fs, info):
    """Plot ERSP and ERD topomap for C3 C4 channel in multi frequency band

    :param epochs: mne epoch data
    :param baseline: [t1, t2], time baseline used to correct ersp
    :param fs: sampling frequency
    :param info: montage info to plot topomap
    :return: ERSP value of 29 channels in 3 frequency bands for left and right hand MI
    """
    # time frequeny analysis
    ch_names = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
                "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
    decim = 1
    freqs = np.arange(6,30,0.5)
    tf = tfr_multitaper(
        epochs,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim = decim
    )
    t0 = baseline[0]
    tf1 = tf.copy()
    tf1.crop(t0, 4).apply_baseline(baseline, mode="logratio")   # crop time to (-1, 4)
    tf1.data = 10*tf1.data

    # plot C3,C4 channel ersp(figure 2 left column of paper, sub=22,43)
    fig, ax = plt.subplots(1, 2, figsize=(10/2.54, 4.5/2.54), gridspec_kw={'width_ratios': [0.5,0.5], 'wspace': 0.055}, sharey=True, layout='constrained')
    tf1['left_hand'].average().plot(picks='C3', cmap='RdBu_r', axes=ax[0], colorbar=False, vmin=-3.8, vmax=3.8)   # Affected hand MI
    tf1['left_hand'].average().plot(picks='C4', cmap='RdBu_r', axes=ax[1], colorbar=False, vmin=-3.8, vmax=3.8)
    # tf1['right_hand'].average().plot(picks='C3', cmap='RdBu_r', axes=ax[0], colorbar=False, vmin=-3.8, vmax=3.8)    # Unaffected hand MI
    # tf1['right_hand'].average().plot(picks='C4', cmap='RdBu_r', axes=ax[1], colorbar=False, vmin=-3.8, vmax=3.8)
    ax[0].plot(np.zeros(25), np.arange(6, 31), '--r')
    ax[1].plot(np.zeros(25), np.arange(6, 31), '--r')
    ax[1].set_ylabel('')
    ax[0].set_title('C3 channel')
    ax[1].set_title('C4 channel')
    clb = fig.colorbar(ax[-1].images[-1], ax=ax[-1], shrink=0.8)  # shrink=0.6
    clb.ax.set_title('dB')
    clb.ax.set(yticklabels=['', '-2.0', '0', '2.0', ''])

    # plot averaged ERSP topomap (figure 2 right column of paper)
    f = tf1.freqs
    plt.rc('font', size=11, family='Arial', weight='normal')
    mask = np.full(29, False, dtype=bool)
    mask[13:15] = 'True'
    fig, ax = plt.subplots(1, 1, figsize=(4.5/2.54, 4.5/2.54))
    ind1 = np.where(f >= 21)[0][0]   # High beta band
    ind2 = np.where(f >= 29)[0][0]
    # ind1 = np.where(f >= 14)[0][0]  # Low beta band
    # ind2 = np.where(f >= 20)[0][0]
    erd_right = tf1['left_hand'].data  # Affected hand MI
    # erd_right = tf1['right_hand'].data  # Unaffected hand MI
    erd_right = np.squeeze(np.mean(erd_right[:, :, ind1:ind2, (1-t0) * fs // decim:(3 - t0) * fs // decim], axis=(-2, -1, 0)))
    im, _ = mne.viz.plot_topomap(erd_right, info, sphere=(0, 0, 0, 0.1), contours=1, cmap='RdBu_r', axes=ax, names=ch_names,
                                 mask=mask, mask_params={'markersize': 1, 'markerfacecolor': 'none'})
    im.set_clim(-1.5, 0.5)    # fig2
    # im.set_clim(-2.5, 1.5)  # fig3
    # im.set_clim(-2.7, 1)  # figS1
    clb = fig.colorbar(ax.images[-1], ax=ax, shrink=0.61)  # shrink=0.6
    clb.ax.set_title('dB')
    clb.ax.tick_params(labelsize=9)
    # clb.set_clim(np.min(erd_right), 0.75)
    ax.set_title('High beta (21-29 Hz)', fontsize=11)
    # ax.set_title('Low beta (14-20 Hz)', fontsize=11)
    plt.rc('font', size=9, family='Arial', weight='normal')

    return 0

def ersp_logratio_band(epochs, baseline, fs, info):
    """Compute raw ERSP value (only averaged in time dimension)

    :param epochs: mne epochs data
    :param baseline: [t1, t2], time baseline used to correct ersp
    :param fs: sampling frequency
    :param info:  montage info
    :return: raw ERSP value for left and right hand, (50, 20, 29, 24)
             50 subs, 20 trials, 29 channels, 24 frequency
    """
    decim = 2
    freqs = np.arange(6,30)
    # EEG的多带频率表示
    tf = tfr_multitaper(
        epochs,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim = decim   # 降采样因子
    )
    t0 = baseline[0]
    tf1 = tf.copy()
    # baseline correct method: log first, average second
    tf1.crop(t0, 4).apply_baseline(baseline, mode="logratio")   # crop time to (-1, 4)
    tf1.data = 10*tf1.data
    erd_left = tf1['left_hand'].data
    erd_right = tf1['right_hand'].data

    return erd_left, erd_right


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap

if __name__ == "__main__":
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

    # s: subject num(1-50)
    for s in range(1, 51):
        print(str(s))
        # s = 43
        # s = 22

        if s >= 10:
            data = sio.loadmat('data_load/data_clean/sub-' + str(s) + '_task-motor-imagery_eeg.mat')    # data preprocessed using EEGLAB
        else:
            data = sio.loadmat('data_load/data_clean/sub-0' + str(s) + '_task-motor-imagery_eeg.mat')

        data = data['EEG_save']
        eeg_data = data['eeg'][0, 0]
        epochs = eeg_data['rawdata'][0, 0]
        labels = np.squeeze(eeg_data['label'][0, 0])
        trigger = np.squeeze(np.where(epochs[:, -1, :] == 2))[1, :]  # the start time point of each MI Task
        epochs_train = epochs_make(epochs[:, 0:29, :], labels, trigger)
        fs = 500

        # plot ERSP (if not need, comment it)
        ersp_plot_logratio(epochs_train, (-1, 0), fs, epochs_train.info)

        # calculate ERSP
        ERD_left, ERD_right = ersp_logratio_band(epochs_train, (-1, 0), fs, epochs_train.info)

        # save ERSP data (further used to statistical analysis)
        save_dir = "data_load/ERD_data_band/"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"erd_left_s{str(s)}"), ERD_left)
        np.save(os.path.join(save_dir, f"erd_right_s{str(s)}"), ERD_right)

    plt.ioff()
    plt.show()

