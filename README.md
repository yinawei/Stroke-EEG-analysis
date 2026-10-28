# Stroke-EEG-analysis

Analysis of motor-imagery EEG data in stroke patients and mechanistic
interpretation based on a whole-brain model.

This repository contains the preprocessing, feature-extraction, statistical,
prediction, sensitivity, and whole-brain simulation analyses used in the
study. All commands should be run from the repository root. They use only the
files under `data_load/`; no local absolute paths or unshared intermediate
results are required. Generated figures and numerical results are saved under
`figures/` and `results/`, respectively.

## Installation

The analyses were tested on Ubuntu 22.04 with Python 3.10.

```bash
conda create -n stroke_eeg_repro python=3.10
conda activate stroke_eeg_repro
pip install -r requirements.txt
```

## Dataset

The stroke EEG dataset was provided by Liu et al. and is publicly
available from [Figshare](https://doi.org/10.6084/m9.figshare.21679035.v5). The
preprocessed EEG files used here are included in `data_load/data_clean/`.

The repository contains 50 preprocessed participant files. The statistical and
prediction scripts automatically apply the prespecified exclusions and analyze
the final cohort of 46 participants. The participant-level ERSP and
functional-connectivity extraction scripts process all 50 files.

The analyses require these inputs:

```text
data_load/
├── subject.csv
├── data_clean/
│   ├── sub-01_task-motor-imagery_eeg.mat
│   ├── ...
│   └── sub-50_task-motor-imagery_eeg.mat
├── ImCoh_data/
│   ├── imcoh_left.npy
│   └── imcoh_right.npy
└── model_data/
    ├── aal2.nii.gz
    ├── curr_cmat.npy
    ├── custom_forward.fif
    └── receptor_data_aal2_neurolib.csv
```

The analyses reconstruct the required features from these repository inputs.
Precomputed result tables, fitted-result archives, and external
`source_data/` directories are not required.

## Usage

### 1. Participant-level ERSP extraction

Python file: `figshare_stroke_ersp.py`

- Calculates event-related spectral perturbation (ERSP) during left- and
  right-hand motor imagery for 50 stroke participants in the alpha, low-beta,
  and high-beta bands.
- Plots time-frequency ERSP maps and C3/C4 spatial topographies.
- Saves participant ERSP arrays as `erd_left_sX.npy` and `erd_right_sX.npy`
  under `data_load/ERD_data_band/`.

```bash
python figshare_stroke_ersp.py
```

The ERSP partial-correlation analysis below reconstructs the required C3/C4
features directly from the uploaded `.mat` files and therefore does not require
the generated `ERD_data_band/` directory.

### 2. ERSP partial-correlation analysis

Python file: `ersp_partial_correlation.py`

- Reconstructs the 18 ERSP features directly from the 50 uploaded EEG files,
  applies the four prespecified participant exclusions, and analyzes the final
  sample of 46 participants.
- Runs the 18 primary partial-Spearman tests while controlling for age, sex,
  post-stroke duration, paralysis side, and first versus recurrent stroke.
- Uses 9,999 two-sided residual permutations, 5,000 paired bootstrap samples,
  and Benjamini-Hochberg FDR correction within each frequency-band-by-hand
  family.
- Creates 18 separate residual-correlation PNG panels. The displayed `p` value
  is the within-family FDR-adjusted permutation value.

```bash
python ersp_partial_correlation.py
```

Outputs:

```text
results/ersp_partial_correlation/partial_spearman_results.csv
results/ersp_partial_correlation/partial_correlation_plot_source.csv
figures/ersp_partial_correlation/*.png
```

### 3. Participant-level functional-connectivity extraction

Python file: `figshare_stroke_fc.py`

- Calculates imaginary-coherence functional connectivity during left- and
  right-hand motor imagery for 50 stroke participants in the alpha, low-beta,
  and high-beta bands.
- Plots functional-connectivity matrices and their corresponding topographic
  maps.
- Saves the participant connectivity arrays as `imcoh_left.npy` and
  `imcoh_right.npy` under `data_load/ImCoh_data/`.

```bash
python figshare_stroke_fc.py
```

The two connectivity arrays produced by this script are included in the
repository and are used by the MST analysis.

### 4. MST partial-correlation analysis

Python file: `mst_partial_correlation.py`

- Reconstructs 24 maximum-spanning-tree features from the uploaded
  imaginary-coherence arrays: diameter, eccentricity, leaf number, and tree
  hierarchy for two motor-imagery conditions and three frequency bands.
- Applies the same 46-participant sample, five-covariate partial-Spearman
  procedure, two-sided permutations, paired bootstrap confidence intervals,
  and within-family FDR correction used for the ERSP analysis.
- Creates 24 separate residual-correlation PNG panels. The displayed `p` value
  is the within-family FDR-adjusted permutation value.

```bash
python mst_partial_correlation.py
```

Outputs:

```text
results/mst_partial_correlation/partial_spearman_results.csv
results/mst_partial_correlation/partial_correlation_plot_source.csv
figures/mst_partial_correlation/*.png
```

### 5. Nested PLS prediction and SHAP interpretation

Python file: `feature_important.py`

- Reconstructs all 42 candidate features (18 ERSP and 24 MST) from repository
  inputs.
- Performs leakage-safe nested five-fold partial least-squares regression.
  Feature selection, standardization, and component tuning are performed only
  within the corresponding training folds.
- Fits the descriptive full-cohort PLS-SHAP model used for feature
  interpretation.
- Generates the four individual study panels b-e; no additional
  composite figure is produced.

```bash
python feature_important.py
```

Optional individual stages are:

```bash
python feature_important.py --analysis nested
python feature_important.py --analysis shap
```

Figure outputs:

```text
figures/pls_nested5/panel_b_oof_predictions.png
figures/pls_nested5/panel_c_fold_metrics.png
figures/pls_nested5/panel_d_shap_importance.png
figures/pls_nested5/panel_e_shap_values.png
```

Nested-CV numerical results are saved under `results/pls_nested5/`, and SHAP
results are saved under `results/pls_shap/`. Generalization performance should
be assessed from the nested out-of-fold results; the full-cohort SHAP model is
descriptive.

### 6. FC trial-bootstrap and time-window sensitivity analysis

Python file: `fc_sensitivity_bootstrap_windows.py`

- Recomputes affected-hand low-beta imaginary-coherence matrices from the
  uploaded trial EEG.
- Performs 1,000 trial-bootstrap repetitions.
- Compares the alternative 0-3, 1-4, and 0-4 s motor-imagery windows with the
  primary 1-3 s window.
- Produces the FC-correlation and four-MST-metric sensitivity figures. The
  split-half analysis is intentionally not included.

```bash
python fc_sensitivity_bootstrap_windows.py
```

Use `--workers N` to select the number of parallel participant workers. The
default `--n-bootstrap 1000` is the setting used in the study.

Outputs:

```text
figures/fc_sensitivity_bootstrap_windows/fc_reliability_ahand_low_beta.png
figures/fc_sensitivity_bootstrap_windows/mst_reliability_ahand_low_beta.png
results/fc_sensitivity_bootstrap_windows/*.csv
```

### 7. Neurochemical whole-brain model

Python file: `heterogeneity_whole_brain_model.py`

- Employs a whole-brain model consisting of 80 brain regions, each containing
  interacting excitatory and inhibitory neuronal populations governed by the
  Wilson-Cowan model.
- Implements model heterogeneity using neurotransmitter-density gradients.

```bash
python heterogeneity_whole_brain_model.py
```

### 8. Neurochemical-model cross-validation

Python file: `plot_neurochemical_validation.py`

- Reconstructs nine neurotransmitter-system representative maps from the
  uploaded receptor-density data.
- Runs seven leave-one-subject-out Optuna fits of the 10-min
  Wilson-Cowan/BOLD model and evaluates each held-out HCP functional-connectivity
  matrix.
- Generates the LOOCV performance, fitted system-weight, and spatial
  collinearity panels.

Check that all required inputs and dependencies are available:

```bash
python plot_neurochemical_validation.py --validate-only
```

Run the registered analysis:

```bash
python plot_neurochemical_validation.py
```

The default is 1,000 optimization trials per fold. A fresh run is highly
computationally intensive and may require approximately 10-14 days, depending
on the computer. Completed trials are retained in
`results/neurochemical_validation/optuna_loso.sqlite3`, so the same command can
resume an interrupted run.

Output:

```text
figures/neurochemical_validation/neurochemical_validation.png
```

### 9. Stroke whole-brain simulation

Python file: `Stroke_simulation_whole_brain_model.py`

- Transforms population neuronal signals into scalp EEG signals.
- Simulates the effects of motor imagery and lesions on the whole-brain model.
- Analyzes brain-network topology with and without simulated lesions.

```bash
python Stroke_simulation_whole_brain_model.py
```

### 10. Lesion-model sensitivity analysis

Python file: `plot_lesion_sensitivity.py`

- Runs the Wilson-Cowan-to-scalp-EEG simulation for 25%, 50%, 75%, and 100%
  attenuation of connections incident on selected nodes.
- Evaluates random node deletion, strength-hub-centered deletion, and
  anatomically constrained deletion sequences.
- Calculates diameter, eccentricity, leaf number, and tree hierarchy and
  generates the four individual analysis rows corresponding to study
  panels b-e.

Check the model inputs and lesion sequences:

```bash
python plot_lesion_sensitivity.py --validate-only
```

Run the registered analysis:

```bash
python plot_lesion_sensitivity.py
```

The analysis contains 1,620 registered conditions and generally requires
approximately 1.5-2 hours on the tested workstation. Every completed condition
is saved to `results/lesion_sensitivity/lesion_sensitivity_results.csv`, so the
same command can resume an interrupted run.

Output:

```text
figures/lesion_sensitivity/lesion_sensitivity_panels_b_to_e.png
```

## Directory structure

Generated `figures/` and `results/` directories do not need to be present
before running the scripts.

```text
├── README.md
├── requirements.txt
├── figshare_stroke_ersp.py
├── ersp_partial_correlation.py
├── figshare_stroke_fc.py
├── mst_partial_correlation.py
├── feature_important.py
├── fc_sensitivity_bootstrap_windows.py
├── heterogeneity_whole_brain_model.py
├── plot_neurochemical_validation.py
├── Stroke_simulation_whole_brain_model.py
├── plot_lesion_sensitivity.py
├── data_load/
│   ├── subject.csv
│   ├── data_clean/
│   ├── ImCoh_data/
│   └── model_data/
├── figures/                         # generated PNG figures
└── results/                         # created results and checkpoints
```
