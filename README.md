# Stroke-EEG-analysis

Analysis of motor imagery EEG data in stroke patients and mechanistic interpretation based on a whole-brain model.


## Installation
**Operating System:**  Ubuntu 22.04.5 LTS
1. Create a conda environment
```bash
conda create -n env_name python=3.10.16
```
2. Install pip package
```bash
pip install -r requirements.txt
```


## Dataset
The EEG dataset of stroke patients is provided by Liu et.al in https://doi.org/10.6084/m9.figshare.21679035.v5  
You just need to download "sourcedata.zip" through this link and unzip it to the "dataset/sourcedata" directory.

The EEG data preprocessed using the EEGLAB toolbox of MATLAB in this work is stored in 'data_load/data_clean'.


## Usage
1. Python file: figshare_stroke_ersp.py
-   Calculate the Event-Related Spectral Perturbation (ERSP) values of EEG signals during left/right hand motor imagery for 50 stroke patients, covering three frequency bands: alpha, low beta, and high beta.
-   Plot time-frequency ERSP maps and the spatial topography of ERSP for the C3 and C4 channels.
-   Save the subject-specific ERSP values (named as `erd_left_sx.npy` and `erd_right_sx.npy`, where "x" denotes the subject index) to the directory `data_load/ERD_data_band`.
```bash
python figshare_stroke_ersp.py
```

2. Python file: figshare_ersp_group.py
- Comparision analysis: violinplot of the ERSP metrics (Ipsilateral ERSP, Contralateral ERSP and Lateral Index) under the low NIHSS group and high NIHSS group.
- Correlation analysis: regplot between the NIHSS score and various ERSP metrics.
```bash
python figshare_ersp_group.py
```

3. Python file: figshare_stroke_fc.py
-   For 50 stroke patients, calculate the functional connectivity of EEG signals during left/right hand motor imagery using the imaginary part of coherence, with analyses covering three frequency bands: alpha, low beta, and high beta.
-   Plot functional connectivity matrices and their corresponding topographic maps.
-   Save the functional connectivity data files (named `imcoh_left.npy` for left-hand imagery and `imcoh_right.npy` for right-hand imagery) to the directory `data_load/ImCoh_data`.
```bash
python figshare_stroke_fc.py
```

4. Python file: figshare_fc_mst.py
- Calculate and visualize the maximum spanning tree (MST) transformed from the function connectivity matrix.
- Correlation analysis: regplot between the NIHSS score and various MST metrics (diameter, eccentricity, leaf number, tree hierarchy).
- Comparision analysis: violinplot of the MST metrics under the low NIHSS group and high NIHSS group.
```bash
python figshare_fc_mst.py
```

5. Python file: feature_important.py
- Perform multiple linear regression analysis to examine the relationship between EEG features (that show significant correlations with stroke severity) and NIHSS scores.
- Assess the predictive importance of each EEG feature for stroke severity using SHapley Additive exPlanations (SHAP) analysis.
```bash
python feature_important.py
```

6. Python file: heterogeneity_whole_brain_model.py
- Employ a whole-brain model consisting of 80 brain regions, each containing interacting excitatory and inhibitory neuronal populations governed by the Wilson-Cowan model.
- Implementing model heterogeneity using neurotransmitter density gradients.
```bash
python heterogeneity_whole_brain_model.py
```

7. Python file: Stroke_simulation_whole_brain_model.py
- Transform population neuronal signals into EEG signals.
- Simulate the effects of motor imagery and lesions on the whole-brain model.
- Analyze the topological features of brain networks both with and without simulated lesions.
```bash
python Stroke_simulation_whole_brain_model.py
```

## Directory
```text
│  figshare_stroke_ersp.py
│  figshare_ersp_group.py
│  figshare_stroke_fc.py
│  figshare_fc_mst.py
│  feature_important.py
│  heterogeneity_whole_brain_model.py
│  Stroke_simulation_whole_brain_model.py
│  
├─data_load
│  │  subject.csv
│  │  
│  └─data_clean
│      	sub-01_task-motor-imagery_eeg.mat
│      	sub-02_task-motor-imagery_eeg.mat
│      	...
│      	sub-50_task-motor-imagery_eeg.mat
|   │ 
│   └─ERD_data_band
│      	erd_left_s1.npy
│      	erd_left_s2.npy
│      	...
│      	erd_left_s50.npy
│      	erd_right_s1.npy
│      	erd_right_s2.npy
│      	...
│      	erd_right_s50.npy
|   │ 
│   └─ImCoh_data
|       imcoh_left.npy
|       imcoh_right.npy 
|   │ 
│   └─model_data
│      	aal2.nii.gz
│      	curr_cmat.npy
│      	custom_forward.fif
│      	receptor_data_aal2_neurolib.csv


```

