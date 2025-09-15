# %% 4. convert functional connectivity to maximum spanning tree
## statistical analysis of MST measurements
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from nilearn.plotting import plot_connectome
from scipy import stats
from statannotations.Annotator import Annotator
import matplotlib.ticker as ticker

matplotlib.use("Qt5Agg")

def mst_network(connect):
    """Create maximum spanning tree from fc matrix

    :param connect: functional connectivity matrix
    :return: MST measures: [Diameter, Eccentricity, Leaf number, Tree hierarchy]
    """

    # set node postition and labels
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
    pos = coords[:, :-1]
    labels = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
              "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
    pos_dict = dict()
    labels_dict = dict()
    for i in range(pos.shape[0]):
        labels_dict[i] = labels[i]
        pos_dict[i] = pos[i, :]

    # create graph from adjacent matrix
    G = nx.from_numpy_array(connect)

    # create maximum spanning tree
    MST = nx.maximum_spanning_tree(G, weight='weight')

    # draw MST Figure 4
    # fig, ax = plt.subplots(1, 1, figsize=(6.2 / 2.54, 5.8 / 2.54), layout='constrained')
    # adj = nx.to_numpy_array(MST)
    # adj = np.where(adj == 0, 0, 1)
    # coords[:, 0] = coords[:, 0] * 0.9
    # coords[:, 1] = coords[:, 1] * 1.2
    # coords = coords * 70
    # coords[:, 1] = coords[:, 1] - 15
    # edge_prop = dict()
    # edge_prop['lw'] = 1.5
    # edge_prop['c'] = 'black'
    # edge_prop['alpha'] = 0.6
    # node_degree = MST.degree()
    # tmp = np.array([x[1] for x in node_degree])
    # node_size = tmp * 15
    # plot_connectome(adj, coords, node_size=node_size, display_mode='z', edge_kwargs=edge_prop, axes=ax, node_color='cornflowerblue', alpha=0.2, colorbar=False)

    # calculate MST measure
    diameter = nx.diameter(MST)
    ecc = nx.eccentricity(MST)
    ecc_mean = max(ecc.values()) - min(ecc.values())
    leaf_nodes = [x for x in MST.nodes() if MST.degree(x) == 1]
    bet_cen = nx.betweenness_centrality(MST)
    tree_hie = len(leaf_nodes) / (2 * 28 * max(bet_cen.values()))

    return np.array([diameter, ecc_mean, len(leaf_nodes), tree_hie])

def r2(data, **kws):
    """calculate personr correlation in selected freq band and MI hand

    :param data: dataframe
    :param kws:
    :return:
    """
    col = data.columns.values.tolist()
    col_mst = [x for x in col if x not in ['Freq', 'MI', 'NIHSS']]
    r, p = scipy.stats.pearsonr(data['NIHSS'], data[col_mst[0]])
    ax = plt.gca()
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)

def violin_plot_ahand(data, x, y, pairs):
    """Violin plot for ahand MI

    :param data: dataframe
    :param x: NIHSS group

    :param y: Leaf number
    :param pairs: paired group to statistical analysis
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    sns.stripplot(data=data, x=x, y=y, ax=ax, legend=False, size=2.5, jitter=0.13, palette='muted',
                  facecolors='none')
    for dots in ax.collections:  # remove facecolor and set edgecolor
        facecolors = dots.get_facecolors()
        dots.set_edgecolors(facecolors.copy())
        dots.set_facecolors('none')
        dots.set_linewidth(1)

    sns.violinplot(data=data, x=x, y=y, ax=ax, palette='muted', legend='brief', width=0.4, fill=False)
    ax.legend(loc='upper center', ncols=2, bbox_to_anchor=(0.5, 0.98), frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    s1, p1 = scipy.stats.shapiro(data[data[x] == 'Low'][y])
    s2, p2 = scipy.stats.shapiro(data[data[x] == 'High'][y])
    if p1<0.05 or p2<0.05:      # nonnormal distribution
        test_method = 'Mann-Whitney'
    else:
        test_method = 't-test_ind'

    annotator = Annotator(ax, data=data, x=x, y=y, pairs=pairs)
    annotator.configure(test=test_method, hide_non_significant=False, loc='outside')
    annotator.apply_test()
    annotator.annotate()

def mst_measure(index, pli_left, pli_right):
    """

    :param index: subject index
    :param pli_left: imagery coherence matrix in 3 freq bands for left paralysis sub
    :param pli_right: imagery coherence matrix in 3 freq bands for right paralysis sub
    :return: MST measures for left and right paralysis sub
    """
    mst_stat_left = np.zeros((index.shape[0], 3, 4))
    mst_stat_right = np.zeros((index.shape[0], 3, 4))
    for i, s in enumerate(index):
        data_left = pli_left[s, :, :, :]
        data_right = pli_right[s, :, :, :]

        for j in range(3):
            mst_stat_left[i, j, :] = mst_network(np.squeeze(data_left[:, :, j]))
            mst_stat_right[i, j, :] = mst_network(np.squeeze(data_right[:, :, j]))

    return mst_stat_left, mst_stat_right

if __name__ == "__main__":
    plt.ion()
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

    # load imagery coherence matrix: 50 subjects, 29 channels, 3 frequency bands
    pli_left = np.load('data_load/ImCoh_data/imcoh_left.npy')
    pli_right = np.load('data_load/ImCoh_data/imcoh_right.npy')

    # load subjects info
    sub = pd.read_csv('data_load/subject.csv')

    # plot mst of subject 43,22 (figure 4 right column for paper)
    # mst_example1 = mst_network(np.squeeze(pli_left[42, :, :, 1]))  # 1:low beta band
    # mst_example2 = mst_network(np.squeeze(pli_left[21, :, :, 1]))

    # calculate MST for all subjects
    mst_stat_left, mst_stat_right = mst_measure(np.arange(50), pli_left, pli_right)
    mst_stat_uhand, mst_stat_ahand = np.zeros((50, 3, 4)), np.zeros((50, 3, 4))     # uhand:unaffected hand, ahand:affected hand
    for i in range(50):
        if sub['ParalysisSide'][i] == 'left':   # for left paralysis sub, MI of right hand is MI of unaffected hand
            mst_stat_uhand[i, :, :] = mst_stat_right[i, :, :]
            mst_stat_ahand[i, :, :] = mst_stat_left[i, :, :]
        else:
            mst_stat_uhand[i, :, :] = mst_stat_left[i, :, :]
            mst_stat_ahand[i, :, :] = mst_stat_right[i, :, :]
    mst_stat_all = np.concatenate((mst_stat_ahand, mst_stat_uhand), axis=1)
    mst_stat_all = np.reshape(mst_stat_all, (300, 4))  # [ahand ahand ahand uhand uhand uhand]*50

    # make all data to one dataframe
    data = pd.DataFrame(np.repeat(sub.values, 6, axis=0), columns=sub.columns)
    data['MI'] = pd.Series(['Ahand', 'Ahand', 'Ahand', 'Uhand', 'Uhand', 'Uhand'] * 50)
    data['Freq'] = pd.Series(['alpha', 'beta1', 'beta2', 'alpha', 'beta1', 'beta2'] * 50)
    data['Diameter'] = pd.Series(mst_stat_all[:, 0])
    data['Eccentricity'] = pd.Series(mst_stat_all[:, 1])
    data['Leaf number'] = pd.Series(mst_stat_all[:, 2])
    data['Tree hierarchy'] = pd.Series(mst_stat_all[:, 3])
    data['NIHSS'] = data['NIHSS'].astype(float)
    data['Age'] = data['Age'].astype(float)
    data['Duration'] = data['Duration'].astype(float)

    sub = sub.drop(index=[4, 5, 14, 48])  # delete large artifacts and left handness
    data = data[~data['Participant_ID'].isin(['sub-05', 'sub-06', 'sub-15', 'sub-49'])]

    # regression analysis: only "leaf" metric in beta1 band is correlated with NIHSS
    f = 'beta1'

    # comparision analysis: "low NIHSS group" VS "high NIHSS group"
    data_nihss = data.copy()
    data_nihss['NIHSS group'] = pd.cut(x=data_nihss.NIHSS, bins=[0, 4, 11], labels=['Low', 'High'])
    pairs = [('Low', 'High')]
    # violin plot visual example for flowchart
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == f) & (data_nihss.MI == 'Ahand')], x='NIHSS group', y='Diameter', pairs=pairs)
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == f) & (data_nihss.MI == 'Ahand')], x='NIHSS group', y='Eccentricity', pairs=pairs)
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == f) & (data_nihss.MI == 'Ahand')], x='NIHSS group', y='Leaf number', pairs=pairs)
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == f) & (data_nihss.MI == 'Ahand')], x='NIHSS group', y='Tree hierarchy', pairs=pairs)

    # correct correlation coefficients using Spearman correlation and permutation test
    data_factor = data[(data.Freq == f) & (data.MI == 'Ahand')]      # filter dataframe to ahand MI in selected freq band

    def statistic(x):  # explore all possible pairings by permuting `x`
        rs = stats.spearmanr(x, data_factor[metric]).statistic  # ignore pvalue
        transformed = rs * np.sqrt((len(x) - 2) / ((rs + 1.0) * (1.0 - rs)))
        return transformed

    # linear regression between Leaf number and NIHSS scoe(figure 4 second row for paper)
    metric = 'Diameter'
    fig, ax = plt.subplots(1, 1, figsize=(4.4 / 2.54, 3.8 / 2.54), layout='constrained')
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's': 8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.locator_params(axis='y', nbins=4)

    metric = 'Eccentricity'
    fig, ax = plt.subplots(1, 1, figsize=(4.4 / 2.54, 3.8 / 2.54), layout='constrained')
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's': 8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.locator_params(axis='y', nbins=4)

    metric = 'Leaf number'
    fig, ax = plt.subplots(1, 1, figsize=(4.4 / 2.54, 3.8 / 2.54), layout='constrained')
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's': 8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.locator_params(axis='y', nbins=4)

    metric = 'Tree hierarchy'
    fig, ax = plt.subplots(1, 1, figsize=(4.4 / 2.54, 3.8 / 2.54), layout='constrained')
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's': 8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.locator_params(axis='y', nbins=4)

    plt.ioff()
    plt.show()
