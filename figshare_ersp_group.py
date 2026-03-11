# %% 2. ERSP value statistical analysis

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
# matplotlib.use("Qt5Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'


def violin_plot_ahand(data, x, y, pairs):
    """Plot violinplot of Lateral index between two NIHSS group

    :param data: dataframe
    :param x: NIHSS group
    :param y: lateral index
    :param pairs: paired group used to statistical analysis
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    sns.stripplot(data=data, x=x, y=y, ax=ax, legend=False, size=2.5, jitter=0.13, palette='muted',
                  facecolors='none')
    for dots in ax.collections:
        facecolors = dots.get_facecolors()
        dots.set_edgecolors(facecolors.copy())
        dots.set_facecolors('none')
        dots.set_linewidth(1)
    sns.violinplot(data=data, x=x, y=y, ax=ax, palette='muted', legend='brief', width=0.4, fill=False)
    ax.legend(loc='upper center', ncols=2, bbox_to_anchor=(0.5, 0.98), frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    if y=='ERSP_Contra':
        ax.set(xlabel='NIHSS group', ylabel='Contralateral ERSP')
    if y == 'ERSP_Ipsilateral':
        ax.set(xlabel='NIHSS group', ylabel='Ipsilateral ERSP')
    if y == 'Lateral_Index':
        ax.set(xlabel='NIHSS group', ylabel='Lateral index')

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

if __name__ == "__main__":
    # adjust figure format
    plt.ion()
    plt.rc('font', size=9, family='Arial', weight='normal')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.labelweight'] = 'normal'
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['axes.titleweight'] = 'normal'
    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['svg.fonttype'] = 'none'

    # load subjects info
    sub = pd.read_csv('data_load/subject.csv')

    # load ERSP data
    num = 50
    erd_left = np.zeros((num, 29, 24))  # [subjects, trials, channels, frequency]
    erd_right = np.zeros((num, 29, 24))

    t0 = -1
    decim = 2
    fs = 500
    for s in range(1, 51):
        left = np.load('data_load/ERD_data_band/erd_left_s' + str(s) + '.npy')
        ERD_left = np.squeeze(np.mean(left[:, :, :, int((1 - t0) * fs // decim):int((3 - t0) * fs // decim)], axis=-1))  # 1-3s
        erd_left[s - 1, :, :] = np.mean(ERD_left, axis=0)

        right = np.load('data_load/ERD_data_band/erd_right_s' + str(s) + '.npy')
        ERD_right = np.squeeze(np.mean(right[:, :, :, int((1 - t0) * fs // decim):int((3 - t0) * fs // decim)], axis=-1))  # 1-3s
        erd_right[s - 1, :, :] = np.mean(ERD_right, axis=0)

    # split into 3 frequency bands: alpha(8-13Hz), low beta(14-20Hz), high beta(21-29Hz)
    erd_left_band = np.stack((np.mean(erd_left[:, :, 2:8], axis=-1), np.mean(erd_left[:, :, 8:15], axis=-1),
                              np.mean(erd_left[:, :, 15:], axis=-1)), axis=-1)
    erd_right_band = np.stack((np.mean(erd_right[:, :, 2:8], axis=-1), np.mean(erd_right[:, :, 8:15], axis=-1),
                               np.mean(erd_right[:, :, 15:], axis=-1)), axis=-1)

    # ERSP of contralateral hemishpere in C3/C4 channel
    ersp_contra_uhand, ersp_contra_ahand = np.zeros((50, 3)), np.zeros((50, 3))     # uhand: unaffected hand, ahand: affected hand
    ersp_Ipsilateral_uhand, ersp_Ipsilateral_ahand = np.zeros((50, 3)), np.zeros((50, 3))
    li_uhand, li_ahand = np.zeros((50, 3)), np.zeros((50, 3))   # li: lateral index = contralateral - ipsilateral
    for i in range(50):
        if sub['ParalysisSide'][i] == 'left':
            ersp_contra_uhand[i, :] = erd_right_band[i, 13, :]  # 13:C3 channel, 14:C4 channel
            ersp_contra_ahand[i, :] = erd_left_band[i, 14, :]
            ersp_Ipsilateral_uhand[i, :] = erd_right_band[i, 14, :]  # 13:C3 channel, 14:C4 channel
            ersp_Ipsilateral_ahand[i, :] = erd_left_band[i, 13, :]
            li_uhand[i, :] = erd_right_band[i, 13, :] - erd_right_band[i, 14, :]
            li_ahand[i, :] = erd_left_band[i, 14, :] - erd_left_band[i, 13, :]
        else:
            ersp_contra_uhand[i, :] = erd_left_band[i, 14, :]
            ersp_contra_ahand[i, :] = erd_right_band[i, 13, :]
            ersp_Ipsilateral_uhand[i, :] = erd_left_band[i, 13, :]  # 13:C3 channel, 14:C4 channel
            ersp_Ipsilateral_ahand[i, :] = erd_right_band[i, 14, :]
            li_uhand[i, :] = erd_left_band[i, 14, :] - erd_left_band[i, 13, :]
            li_ahand[i, :] = erd_right_band[i, 13, :] - erd_right_band[i, 14, :]
    ersp_all = np.concatenate((ersp_contra_ahand, ersp_contra_uhand), axis=-1)      # ahand-alpha, ahand-beta1, ahand-beta2, uhand-alpha, uhand-beta1, uhand-beta2
    ersp_ipsilateral_all = np.concatenate((ersp_Ipsilateral_ahand, ersp_Ipsilateral_uhand), axis=-1)
    li_all = np.concatenate((li_ahand, li_uhand), axis=-1)
    ersp_all = np.reshape(ersp_all, (300, ))
    ersp_ipsilateral_all = np.reshape(ersp_ipsilateral_all, (300,))
    li_all = np.reshape(li_all, (300, ))

    # make all data to one dataframe
    data = pd.DataFrame(np.repeat(sub.values, 6, axis=0), columns=sub.columns)
    data['MI'] = pd.Series(['Ahand','Ahand','Ahand', 'Uhand','Uhand','Uhand']*50)
    data['Freq'] = pd.Series(['alpha','beta1','beta2','alpha','beta1','beta2']*50)
    data['ERSP_Contra'] = pd.Series(ersp_all)
    data['ERSP_Ipsilateral'] = pd.Series(ersp_ipsilateral_all)
    data['Lateral_Index'] = pd.Series(li_all)
    data['NIHSS'] = data['NIHSS'].astype(float)
    data['Age'] = data['Age'].astype(float)
    data['Duration'] = data['Duration'].astype(float)
    sub = sub.drop(index=[4, 5, 14, 48])     # delete large artifacts and left handness
    data = data[~data['Participant_ID'].isin(['sub-05', 'sub-06', 'sub-15', 'sub-49'])]

    # comparision analysis: "low NIHSS group" VS "high NIHSS group" (for all subs and only left paralysis subs)
    data_nihss = data.copy()
    data_nihss['NIHSS_Group'] = pd.cut(x=data_nihss.NIHSS, bins=[0, 4, 11], labels=['Low', 'High'])   # Mild：[0,4]，severe(4,11)
    pairs=[('Low', 'High')]

    # violin_plot(data=data_nihss[(data_nihss.Freq == 'beta2')], x='NIHSS_Group', y='ERSP_Contra', hue='MI', pairs=pairs)
    data1 = data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Ahand') & (data_nihss.NIHSS_Group == 'Low')]
    data2 = data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Ahand') & (data_nihss.NIHSS_Group == 'High')]

    # Violin plot results
    # Affected hand MI
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Ahand')], x='NIHSS_Group', y='ERSP_Contra', pairs=pairs)
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Ahand')], x='NIHSS_Group', y='ERSP_Ipsilateral', pairs=pairs)
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Ahand')], x='NIHSS_Group', y='Lateral_Index', pairs=pairs)
    # Unaffected hand MI
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Uhand')], x='NIHSS_Group', y='ERSP_Contra', pairs=pairs)
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Uhand')], x='NIHSS_Group', y='ERSP_Ipsilateral', pairs=pairs)
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == 'beta2') & (data_nihss.MI == 'Uhand')], x='NIHSS_Group', y='Lateral_Index', pairs=pairs)

    # correct correlation coefficients using Spearman correlation and permutation test
    # linear regression between contra ERSP and NIHSS score
    data_factor = data[(data.Freq == 'beta2') & (data.MI == 'Ahand')]   # Affected hand MI, high beta
    # data_factor = data[(data.Freq=='beta2') & (data.MI=='Uhand')]       # Unaffected hand MI, high beta
    # data_factor = data[(data.Freq == 'beta1') & (data.MI == 'Uhand')]  # Unaffected hand MI, low beta

    # Contralateral ERSP
    fig, ax = plt.subplots(1,1, figsize=(4.8/2.54, 4.4/2.54), layout='constrained')
    metric = 'ERSP_Contra'
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color':'royalblue', 's':8}, line_kws={'color':'royalblue'})
    def statistic(x):  # explore all possible pairings by permuting `x`
        rs = stats.spearmanr(x, data_factor[metric]).statistic  # ignore pvalue
        transformed = rs * np.sqrt((len(x) - 2) / ((rs+1.0)*(1.0-rs)))
        return transformed
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.set(ylabel='Contralateral ERSP')
    ax.spines[['top', 'right']].set_visible(False)

    # Ipsilateral ERSP
    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    metric = 'ERSP_Ipsilateral'
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's': 8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.set(ylabel='Ipsilateral ERSP')
    ax.spines[['top', 'right']].set_visible(False)

    # Lateral Index
    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    metric = 'Lateral_Index'
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's': 8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.set(ylabel='Lateral Index')
    ax.spines[['top', 'right']].set_visible(False)

    plt.ioff()
    plt.show()

