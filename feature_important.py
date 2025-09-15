# %% 5. The importance of various features in stroke rehabilitation assessment

import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
import networkx as nx
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

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

    # calculate MST measure
    diameter = nx.diameter(MST)
    ecc = nx.eccentricity(MST)
    ecc_mean = max(ecc.values()) - min(ecc.values())
    leaf_nodes = [x for x in MST.nodes() if MST.degree(x) == 1]
    bet_cen = nx.betweenness_centrality(MST)
    tree_hie = len(leaf_nodes) / (2 * 28 * max(bet_cen.values()))

    return np.array([diameter, ecc_mean, len(leaf_nodes), tree_hie])


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
    erd_left = np.zeros((num, 20, 29, 24))  # [subjects, trials, channels, frequency]
    erd_right = np.zeros((num, 20, 29, 24))
    t0 = -1 # 起始时间点
    decim = 2
    fs = 500
    for s in range(1, 51):
        left = np.load('data_load/ERD_data_band/erd_left_s' + str(s) + '.npy')
        ERD_left = np.squeeze(
            np.mean(left[:, :, :, int((1 - t0) * fs // decim):int((3 - t0) * fs // decim)], axis=-1))  # 1-3s
        erd_left[s - 1, :, :] = np.mean(ERD_left, axis=0)

        right = np.load('data_load/ERD_data_band/erd_right_s' + str(s) + '.npy')
        ERD_right = np.squeeze(
            np.mean(right[:, :, :, int((1 - t0) * fs // decim):int((3 - t0) * fs // decim)], axis=-1))
        erd_right[s - 1, :, :] = np.mean(ERD_right, axis=0)

    # calculate ipsilateral, contralateral ERSP and lateral index
    erd_left = np.squeeze(np.mean(erd_left, axis=1))    # averaged in trials
    erd_right = np.squeeze(np.mean(erd_right, axis=1))

    # split into 3 frequency bands: alpha(8-13Hz), low beta(14-20Hz), high beta(21-29Hz)
    erd_left_band = np.stack((np.mean(erd_left[:, :, 2:8], axis=-1), np.mean(erd_left[:, :, 8:15], axis=-1),
                              np.mean(erd_left[:, :, 15:], axis=-1)), axis=-1)
    erd_right_band = np.stack((np.mean(erd_right[:, :, 2:8], axis=-1), np.mean(erd_right[:, :, 8:15], axis=-1),
                               np.mean(erd_right[:, :, 15:], axis=-1)), axis=-1)

    # load imagery coherence matrix: 50 subjects, 29 channels, 3 frequency bands
    pli_left = np.load('data_load/ImCoh_data/imcoh_left.npy')
    pli_right = np.load('data_load/ImCoh_data/imcoh_right.npy')

    # features
    ersp_contra_uhand, ersp_contra_ahand, ersp_imcoh_uhand = np.zeros((50, 3)), np.zeros((50, 3)), np.zeros((50, 3))     # uhand: unaffected hand, ahand: affected hand
    li_uhand, li_ahand = np.zeros((50, 3)), np.zeros((50, 3))   # li: lateral index = contralateral - ipsilateral
    mst_stat_left, mst_stat_right = mst_measure(np.arange(50), pli_left, pli_right)
    mst_stat_uhand, mst_stat_ahand = np.zeros((50, 3, 4)), np.zeros((50, 3, 4))
    ersp_left_brain = np.zeros((50, 3))
    gender = np.zeros(50)    # 0:male, 1:female
    for i in range(50):
        if sub['ParalysisSide'][i] == 'left':
            ersp_contra_uhand[i, :] = erd_right_band[i, 13, :]  # 13:C3 channel, 14:C4 channel
            ersp_contra_ahand[i, :] = erd_left_band[i, 14, :]
            ersp_imcoh_uhand[i, :] = erd_right_band[i, 14, :]
            li_uhand[i, :] = erd_right_band[i, 13, :] - erd_right_band[i, 14, :]
            li_ahand[i, :] = erd_left_band[i, 14, :] - erd_left_band[i, 13, :]
            mst_stat_uhand[i, :, :] = mst_stat_right[i, :, :]
            mst_stat_ahand[i, :, :] = mst_stat_left[i, :, :]
            ersp_left_brain[i, :] = erd_right_band[i, 13, :]
        else:
            ersp_contra_uhand[i, :] = erd_left_band[i, 14, :]
            ersp_contra_ahand[i, :] = erd_right_band[i, 13, :]
            ersp_imcoh_uhand[i, :] = erd_left_band[i, 13, :]
            li_uhand[i, :] = erd_left_band[i, 14, :] - erd_left_band[i, 13, :]
            li_ahand[i, :] = erd_right_band[i, 13, :] - erd_right_band[i, 14, :]
            mst_stat_uhand[i, :, :] = mst_stat_left[i, :, :]
            mst_stat_ahand[i, :, :] = mst_stat_right[i, :, :]
            ersp_left_brain[i, :] = erd_left_band[i, 13, :]

        if sub['Gender'][i] == 'female':
            gender[i] = 1

    leafnum_ahand = mst_stat_ahand[:, 1, 2]  # leaf number in beta1
    ersp_ahand = ersp_contra_ahand[:, 2]  # ersp_contra_ahand in beta2
    ersp_uhand = ersp_contra_uhand[:, 2]
    ersp_uhand_beta1 = ersp_contra_uhand[:, 1]
    ersp_i_uhand = ersp_imcoh_uhand[:, 2]
    ersp_i_uhand_beta1 = ersp_imcoh_uhand[:, 1]
    li_ahand = li_ahand[:, 2]
    ersp_left_brain = ersp_left_brain[:, 2]    # 取beta2
    ecc_ahand = mst_stat_ahand[:, 2, 1]

    leafnum_ahand = np.reshape(leafnum_ahand, (50,))
    ersp_ahand = np.reshape(ersp_ahand, (50,))
    ersp_uhand = np.reshape(ersp_uhand, (50,))
    ersp_i_uhand = np.reshape(ersp_i_uhand, (50,))
    ersp_i_uhand_beta1 = np.reshape(ersp_i_uhand_beta1, (50,))
    li_ahand = np.reshape(li_ahand, (50,))
    gender = np.reshape(gender, (50,))
    ersp_left_brain = np.reshape(ersp_left_brain, (50,))
    ecc_ahand = np.reshape(ecc_ahand, (50,))

    heterogeneity_ahand = mst_stat_ahand[:, 1, 3]  # leaf number in beta1
    heterogeneity_ahand = np.reshape(heterogeneity_ahand, (50,))

    # make all data to one dataframe
    data = pd.DataFrame(sub.values, columns=sub.columns)
    data['HE(Al)'] = pd.Series(heterogeneity_ahand)
    data['CE(Ah)'] = pd.Series(ersp_ahand)
    data['CE(Uh)'] = pd.Series(ersp_uhand)
    data['CE(Ul)'] = pd.Series(ersp_uhand_beta1)
    data['IE(Uh)'] = pd.Series(ersp_i_uhand)
    data['IE(Ul)'] = pd.Series(ersp_i_uhand_beta1)
    data['LI(Ah)'] = pd.Series(li_ahand)
    data['LN(Al)'] = pd.Series(leafnum_ahand)
    data['ECC(Ah)'] = pd.Series(ecc_ahand)
    data['Left-brain ERSP'] = pd.Series(ersp_left_brain)
    data['NIHSS'] = data['NIHSS'].astype(float)
    data['Age'] = data['Age'].astype(float)
    data['Duration'] = data['Duration'].astype(float)
    data['Gender'] = pd.Series(gender)
    sub = sub.drop(index=[4, 5, 14, 48])     # delete large artifacts and left handness
    data = data[~data['Participant_ID'].isin(['sub-05', 'sub-06', 'sub-15', 'sub-49'])]

    # Multiple linear regression of EEG features and NIHSS scores
    selected_features = ['CE(Ah)', 'CE(Uh)', 'CE(Ul)', 'IE(Uh)', 'IE(Ul)', 'LI(Ah)', 'LN(Al)'] #
    X_features = data[selected_features]
    Y = data['NIHSS']

    com_score = []
    for n_com in range(1, len(selected_features) + 1):
        model = PLSRegression(n_components=n_com)
        kf = KFold(n_splits=5, shuffle=True, random_state=65)
        scores = cross_val_score(model, X_features, Y, cv=kf, scoring='r2')
        mean_score = np.mean(scores)
        com_score.append(mean_score)

    # The performance of the model changes with the number of components of the PLS model
    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.8 / 2.54), layout='constrained')
    ax.plot(np.linspace(1, len(selected_features), len(selected_features)), com_score, marker='o', linestyle='-', color='royalblue')
    ax.set(xlabel='Number of Components', ylabel='Mean ' + r'R$^2$'+' (5-fold CV)')
    ax.spines[['top', 'right']].set_visible(False)

    index = np.where(com_score == max(com_score))[0]
    n_com = index[0] + 1

    model = PLSRegression(n_components=n_com)
    kf = KFold(n_splits=5, shuffle=True, random_state=65)
    scores = cross_val_score(model, X_features, Y, cv=kf, scoring='r2')
    mse = cross_val_score(model, X_features, Y, cv=kf, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-mse)
    mean_score = np.mean(scores)
    mean_rmse = np.mean(rmse)
    index = np.where(scores == np.max(scores))[0]
    n_index = 0

    # Fold with the max R²
    for train_index, test_index in kf.split(X_features):
        X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        n_index = n_index + 1
        if n_index > index[0]:
            break

    model = PLSRegression(n_components=n_com)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)  # R²

    # Relationship between the predicted NIHSS and the actual NIHSS in the model with the highest R² from 5-fold CV
    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.8 / 2.54), layout='constrained')
    plt.scatter(y_test, y_pred, color='royalblue', s=10)
    plt.plot(y_test, y_test, color='#2d5ff5', linewidth=2, label='Regression line')
    ax.xaxis.set_ticks([0, 2, 4, 6, 8])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlabel='Actual NIHSS', ylabel='Predicted NIHSS')
    ax.text(.05, .9, r'R$^2$=%.2f' % r2, transform=ax.transAxes, fontsize=9)

    # The importance of each EEG feature in the PLS regression model (SHAP method)
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(X_train)
    fig, ax = plt.subplots(1, 1, figsize=(9.5 / 2.54, 6.5 / 2.54), layout='constrained')
    shap.plots.beeswarm(shap_values, plot_size=(9.5 / 2.54, 6.5 / 2.54))
    fig, ax = plt.subplots(1, 1, figsize=(9.5 / 2.54, 6 / 2.54), layout='constrained')
    shap.plots.bar(shap_values, ax=ax)
    plt.show()