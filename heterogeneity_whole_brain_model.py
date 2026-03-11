# %% 6. Optimization of the whole brain model.

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from neurolib.models.wc import WCModel
import neurolib.utils.functions as func
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neurolib.utils.loadData import Dataset
from scipy.stats import zscore
from neurolib.utils.loadData import Dataset
from nilearn.datasets import fetch_atlas_aal
from neurolib.utils.atlases import AutomatedAnatomicalParcellation2
import optuna
from sklearn.decomposition import PCA, NMF
import random
from sklearn.preprocessing import MinMaxScaler
from neurolib.models.bold import BOLDModel
from neurolib.utils.functions import matrix_correlation
import nibabel as nib
from nilearn import plotting
from nilearn.plotting import find_parcellation_cut_coords
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker



# plt.rcParams['svg.fonttype'] = 'none'
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
# matplotlib.rcParams['mathtext.fontset'] = 'cm'


# Core Config & Data Load
ds = Dataset('hcp')
SC_raw = ds.Cmat
FC_emp = np.mean(np.array(ds.FCs), axis=0)
num_nodes = SC_raw.shape[0]
SC = SC_raw
mask = np.triu_indices(num_nodes, k=1)
y_true = FC_emp[mask]
total_duration = 10 * 60 * 1000  # 10 min
sampling_dt_bold = 2000    # sfreq of the bold signal

# load neurochemical densities
receptor_all = np.genfromtxt('data_load/model_data/receptor_data_aal2_neurolib.csv', delimiter=',')
receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])

# Z-score Normalization
density_special_norm = np.zeros_like(receptor_all)
for i in range(receptor_all.shape[1]):
    density_special_norm[:, i] = zscore(receptor_all[:, i])


# Categorizing 19 neurotransmitters into 9 neurotransmitter systems
nt_systems_dict = {
    "Serotonin": ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    "Dopamine": ["D1", "D2", "DAT"],
    "Acetylcholine": ["M1", "A4B2", "VAChT"],
    "GABA": ["GABAa"],
    "Glutamate": ["NMDA", "mGluR5"],
    "Cannabinoid": ["CB1"],
    "Opioid": ["MOR"],
    "Histamine": ["H3"],
    "Norepinephrine": ["NET"]
}

# Extracting the principal gradients for each system（PC1）
system_representative_maps = []
system_names = []

for sys_name, receptors in nt_systems_dict.items():
    valid_receptors = [r for r in receptors if r in receptor_names]
    indices = [np.where(receptor_names == r)[0][0] for r in valid_receptors]
    sub_data = density_special_norm[:, indices]

    # PC1 represents the common spatial variance pattern of receptors within the system.
    if len(valid_receptors) > 1:
        pca = PCA(n_components=1)
        rep_map = pca.fit_transform(sub_data).flatten()
        if np.corrcoef(rep_map, sub_data[:, 0])[0, 1] < 0:
            rep_map = -rep_map
    else:
        rep_map = sub_data.flatten()

    system_representative_maps.append(zscore(rep_map))
    system_names.append(sys_name)

PC_maps_full = np.array(system_representative_maps).T
n_systems = len(system_names)


def fmri_fit(bold_signal):
    sim_bold_raw = bold_signal[:, 15:]  # Exclude the first 30 seconds of unstable data to ensure the model reaches a steady state.
    fc = func.fc(sim_bold_raw)
    fc_corr = np.mean([matrix_correlation(fc, f) for f in ds.FCs])

    return fc, fc_corr

# Core simulation function (single run)
def run_simulation(G, combined_map_data):
    #  initial wEE
    wee = np.concatenate([np.ones(num_nodes) * 16.0])  
    multiplier = 1.0 + combined_map_data
    # Restricting the range to prevent wEE from becoming negative.
    multiplier = np.clip(multiplier, 0.1, 5.0)  
    wEEs = wee * multiplier

    #  In neurolib, wEE is treated as a global constant. 
    # Therefore, local heterogeneity of Wee is implemented by adjusting 
    # the diagonal elements of the structural connectivity (SC) matrix.
    curr_cmat = ds.Cmat.copy() * G
    np.fill_diagonal(curr_cmat, wEEs)
    # np.save("data_load/model_data/curr_cmat.npy", curr_cmat)

    # Constructing the WC model with the adjusted wEE.
    model = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat, seed=4)
    model.params['duration'] = total_duration
    model.params['signalV'] = 0
    model.params['sigma_ou'] = 0.03
    model.params['c_excexc'] = 0
    model.params['K_gl'] = 1
    model.params['Cmat'] = curr_cmat
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

    return fc_corr


# objective function
def objective(trial):
    # Define the search space
    G = trial.suggest_float("G", 2.5, 5.5)  
    a = np.array([
        trial.suggest_float(f"a_{k}", -0.5, 0.5) for k in range(PC_maps_full.shape[1])
    ])

    combined_map_data = PC_maps_full @ a

    # Operating model
    sim_correlation = run_simulation(G, combined_map_data)

    # L2 regularization
    alpha = 0.01
    penalty = alpha * np.sum(a ** 2)

    return sim_correlation - penalty


if __name__ == '__main__':
    # Operation optimization (The process takes a long time; the optimal result has been saved in subsequent steps.)
    # n_trials = 1000
    # sampler = optuna.samplers.CmaEsSampler(seed=42)
    # study = optuna.create_study(direction="maximize", sampler=sampler)
    # study.optimize(objective, n_trials=n_trials)
    
    # values = [t.value for t in study.trials if t.value is not None]
    
    # best_so_far = []
    # current_best = -1e9
    # for v in values:
    #     if v > current_best:
    #         current_best = v
    #     best_so_far.append(current_best)
    
    # plt.figure(figsize=(10.5 / 2.54, 5/ 2.54))
    # plt.plot(values, alpha=0.5, label="Each trial")
    # plt.plot(best_so_far, linewidth=2, label="Best")
    # plt.xlabel("Iteration")
    # plt.ylabel("Correlation")
    # plt.legend()
    # plt.tight_layout()
    # ax = plt.gca()  
    # ax.spines[['right', 'top']].set_visible(False)
    # # plt.savefig("figure/optuna_convergence.svg", dpi=300)
    # plt.show()
    
    # print(f"The best parameters: {study.best_params}")
    # print(f"The best correlation: {study.best_value}")


    # Save optimal parameters
    G = 3.3067495525047335
    best_a = np.array([-0.04840652213420377, 0.2604545008707091, 0.039867898329825735, 0.031276915233522495,
                  -0.22626310960304485, 0.21573389858264636, -0.23416705206682936, -0.0035664354719125435,
                  0.12042318451833012])
    combined_map_data = PC_maps_full @ best_a
    wee = np.concatenate([np.ones(num_nodes) * 16.0])  
    multiplier = 1.0 + combined_map_data
    multiplier = np.clip(multiplier, 0.1, 5.0)  
    wEEs = wee * multiplier
    curr_cmat = ds.Cmat.copy() * G
    np.fill_diagonal(curr_cmat, wEEs)
    # np.save("data_load/model_data/curr_cmat.npy", curr_cmat)


    # Spatial distribution of neurochemical density
    atlas = AutomatedAnatomicalParcellation2()
    aal2_name = atlas.node_names  
    cortex = atlas.names()  
    cortex_idx = [aal2_name.index(name) for name in cortex]
    aal2_nii_path = r"data_load/model_data/aal2.nii.gz"
    aal2_img = nib.load(aal2_nii_path)
    all_coords = find_parcellation_cut_coords(labels_img=aal2_img)
    node_coords = all_coords[cortex_idx]
    target_system_idx = 0
    system_name = system_names[target_system_idx]
    system_pc1_map = PC_maps_full[:, target_system_idx]

    fig1 = plt.figure(figsize=(4/2.54, 3.6/2.54))
    vmin, vmax = np.min(system_pc1_map), np.max(system_pc1_map)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('RdBu_r')
    no_connections = np.zeros((len(node_coords), len(node_coords)))
    node_colors = [cmap(norm(val)) for val in system_pc1_map]
    display = plotting.plot_connectome(
        adjacency_matrix=no_connections,  
        node_coords=node_coords,  
        node_color=node_colors,  
        node_size=10, 
        display_mode='z',  
        colorbar=False, 
        alpha=0.2,
        figure=fig1
    )
    img1 = cm.ScalarMappable(norm=norm, cmap='RdBu_r')

    # Colorbar
    axs = fig1.axes  
    clb = fig1.colorbar(img1, ax=axs, shrink=0.5)
    clb.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
    clb.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))  

    # fig1.savefig(f"figure/{system_name}_Surface_Map.svg", dpi=300, bbox_inches='tight')

    # Estimated contributions (a_k) of different neurotransmitter systems to regional heterogeneity
    fig2, ax = plt.subplots(figsize=(4.8/2.54, 4.8/2.54))
    sort_idx = np.argsort(best_a)
    sorted_weights = best_a[sort_idx]
    sorted_names = np.array(system_names)[sort_idx]

    colors = ['#d62728' if w < 0 else '#1f77b4' for w in sorted_weights]

    bars = ax.barh(sorted_names, sorted_weights, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel(r'$a_k$')
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    # plt.savefig("figure/System_Weights_Barplot.svg", dpi=300)
    plt.close()

    # Spatial topography of the optimized wEEs
    fig3 = plt.figure(figsize=(4/2.54, 3.6/2.54))
    vmin3, vmax3 = np.min(wEEs), np.max(wEEs)
    norm3 = mcolors.Normalize(vmin=vmin3, vmax=vmax3)
    cmap3 = plt.get_cmap('plasma')  
    node_colors3 = [cmap3(norm3(val)) for val in wEEs]

    plotting.plot_connectome(
        adjacency_matrix=no_connections,  
        node_coords=node_coords,
        node_color=node_colors3,  
        node_size=10,  
        display_mode='z',
        colorbar=False,
        alpha = 0.2,
        figure=fig3
    )

    img3 = cm.ScalarMappable(norm=norm3, cmap='plasma')
    axs3 = fig3.axes
    clb3 = fig3.colorbar(img3, ax=axs3, shrink=0.5)
    clb3.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
    clb3.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    # fig3.savefig("figure/Combined_Heterogeneity_Classic_Nodes_Z.svg", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    