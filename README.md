# Stroke-EEG-analysis

Analysis of motor-imagery EEG data in stroke patients and mechanistic
interpretation based on a whole-brain model.

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

### 2. ERSP partial-correlation analysis

Python file: `ersp_partial_correlation.py`

- Performs partial Spearman correlation tests for ERSP features, controlling for age, sex,
  post-stroke duration, paralysis side, and first versus recurrent stroke.

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

### 4. MST partial-correlation analysis

Python file: `mst_partial_correlation.py`

- Performs partial Spearman correlation tests for MST features, controlling for age, sex,
  post-stroke duration, paralysis side, and first versus recurrent stroke.

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

- Performs leakage-safe nested five-fold partial least-squares regression.
- Fits the PLS-SHAP model used for feature interpretation.

```bash
python feature_important.py
```

Figure outputs:

```text
figures/pls_nested5/panel_b_oof_predictions.png
figures/pls_nested5/panel_c_fold_metrics.png
figures/pls_nested5/panel_d_shap_importance.png
figures/pls_nested5/panel_e_shap_values.png
```

Nested-CV numerical results are saved under `results/pls_nested5/`, and SHAP
results are saved under `results/pls_shap/`.

### 6. FC trial-bootstrap and time-window sensitivity analysis

Python file: `fc_sensitivity_bootstrap_windows.py`

- Performs 1,000 trial-bootstrap repetitions.
- Compares the alternative 0-3, 1-4, and 0-4 s motor-imagery windows with the
  primary 1-3 s window.

```bash
python fc_sensitivity_bootstrap_windows.py
```

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

- Runs seven leave-one-subject-out Optuna fits of the 10-min
  Wilson-Cowan/BOLD model and evaluates each held-out HCP functional-connectivity
  matrix.
- Generates the LOOCV performance, fitted system-weight, and spatial
  collinearity panels.

```bash
python plot_neurochemical_validation.py
```

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

```bash
python plot_lesion_sensitivity.py
```
Output:

```text
figures/lesion_sensitivity/lesion_sensitivity_panels_a_to_d.png
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
