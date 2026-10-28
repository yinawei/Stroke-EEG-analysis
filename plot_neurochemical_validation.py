"""Run neurochemical-model validation and plot the validation panels."""
import argparse
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neurochemical_validation_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
OUT = SCRIPT_DIR / "figures/neurochemical_validation"
RESULTS = SCRIPT_DIR / "results/neurochemical_validation"
SYSTEMS = {
    "Serotonin": ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    "Dopamine": ["D1", "D2", "DAT"], "Acetylcholine": ["M1", "A4B2", "VAChT"],
    "GABA": ["GABAa"], "Glutamate": ["NMDA", "mGluR5"],
    "Cannabinoid": ["CB1"], "Opioid": ["MOR"], "Histamine": ["H3"],
    "Norepinephrine": ["NET"],
}
RECEPTORS = ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
             "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
             "MOR", "NET", "NMDA", "VAChT"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def system_maps():
    path = ROOT / "data_load/model_data/receptor_data_aal2_neurolib.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing uploaded receptor data: {path}")
    raw = np.loadtxt(path, delimiter=",")
    standardized = zscore(raw, axis=0)
    maps = []
    for receptors in SYSTEMS.values():
        subset = standardized[:, [RECEPTORS.index(name) for name in receptors]]
        if subset.shape[1] > 1:
            representative = PCA(n_components=1).fit_transform(subset).ravel()
            if np.corrcoef(representative, subset[:, 0])[0, 1] < 0:
                representative *= -1
        else:
            representative = subset.ravel()
        maps.append(zscore(representative))
    return raw, standardized, np.asarray(maps).T


def load_backend():
    try:
        import optuna
        from neurolib.models.bold import BOLDModel
        from neurolib.models.wc import WCModel
        import neurolib.utils.functions as functions
        from neurolib.utils.loadData import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "This analysis requires neurolib, optuna and cmaes from requirements.txt."
        ) from exc
    dataset = Dataset("hcp")
    empirical = np.asarray(dataset.FCs, dtype=float)
    if empirical.shape != (7, 80, 80):
        raise ValueError(f"Expected seven 80-region HCP FC matrices, found {empirical.shape}")
    return optuna, BOLDModel, WCModel, functions, dataset, empirical


def simulate(parameters, maps, backend):
    _, BOLDModel, WCModel, functions, dataset, empirical = backend
    coupling, weights = parameters
    multiplier = np.clip(1.0 + maps @ weights, .1, 5.0)
    cmat = dataset.Cmat.copy() * coupling
    np.fill_diagonal(cmat, 16.0 * multiplier)
    model = WCModel(Cmat=dataset.Cmat, Dmat=dataset.Dmat, seed=4)
    model.params["duration"] = 10 * 60 * 1000
    model.params["signalV"] = 0
    model.params["sigma_ou"] = .03
    model.params["c_excexc"] = 0
    model.params["K_gl"] = 1
    model.params["Cmat"] = cmat
    model.run(chunkwise=False, bold=False)
    neural = model.output.copy()
    neural[neural < 0] = 0
    bold_model = BOLDModel(N=80, dt=model.params["dt"])
    bold_model.run(neural)
    simulated_fc = functions.fc(bold_model.BOLD[:, 15:])
    correlations = np.asarray([
        functions.matrix_correlation(simulated_fc, matrix) for matrix in empirical
    ])
    return simulated_fc, correlations


def run_loso(n_trials, seed, maps):
    backend = load_backend()
    optuna, _, _, _, _, empirical = backend
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / "loso_fold_results.csv"
    if output_path.exists():
        existing = pd.read_csv(output_path)
        rows = {int(row.fold): row.to_dict() for _, row in existing.iterrows()}
    else:
        rows = {}
    for held_out in range(len(empirical)):
        fold = held_out + 1
        training = [index for index in range(len(empirical)) if index != held_out]
        storage = (RESULTS / "optuna_loso.sqlite3").resolve()
        study = optuna.create_study(
            study_name=f"neurochemical_loso_fold_{fold:02d}",
            direction="maximize", sampler=optuna.samplers.CmaEsSampler(seed=seed),
            storage=f"sqlite:///{storage}", load_if_exists=True,
        )

        def objective(trial):
            coupling = trial.suggest_float("G", 2.5, 5.5)
            weights = np.asarray([
                trial.suggest_float(f"a_{index}", -.5, .5) for index in range(maps.shape[1])
            ])
            _, correlations = simulate((coupling, weights), maps, backend)
            return float(correlations[training].mean() - .01 * np.sum(weights ** 2))

        complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if complete < n_trials:
            study.optimize(objective, n_trials=n_trials - complete)
        parameters = study.best_params
        coupling = float(parameters["G"])
        weights = np.asarray([parameters[f"a_{index}"] for index in range(maps.shape[1])])
        _, correlations = simulate((coupling, weights), maps, backend)
        row = {
            "fold": fold, "held_out_subject": f"subject_{fold:02d}",
            "held_out_index_zero_based": held_out,
            "training_indices_zero_based": ";".join(map(str, training)),
            "n_train": 6, "n_test": 1,
            "n_trials_complete": sum(
                t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
            ),
            "n_trials_all_states": len(study.trials),
            "best_penalized_training_objective": study.best_value,
            "training_mean_fc_correlation": correlations[training].mean(),
            "held_out_fc_correlation": correlations[held_out], "G": coupling,
        }
        row.update({f"weight_{name}": value for name, value in zip(SYSTEMS, weights)})
        row.update({f"fc_correlation_subject_{index+1:02d}": value
                    for index, value in enumerate(correlations)})
        rows[fold] = row
        pd.DataFrame([rows[key] for key in sorted(rows)]).to_csv(output_path, index=False)
        print(f"Fold {fold}/7 complete: held-out r={correlations[held_out]:.3f}", flush=True)
    return pd.DataFrame([rows[key] for key in sorted(rows)])


def plot_validation(folds, raw, standardized, maps):
    OUT.mkdir(parents=True, exist_ok=True)
    folds = folds.sort_values("fold")
    assert folds.fold.tolist() == list(range(1, 8))
    assert folds.held_out_subject.nunique() == 7
    names = list(SYSTEMS)
    weights = folds[["weight_" + name for name in names]].to_numpy().T
    assert raw.shape == (80, 19)
    corr = np.corrcoef(maps.T)
    original_corr = np.corrcoef(standardized.T)
    summary = []
    for label, matrix, correlation in [("19 receptor maps", standardized, original_corr),
                                        ("9 system maps", maps, corr)]:
        pairs = correlation[np.tril_indices(len(correlation), -1)]
        summary.append(dict(Maps=label, median_absolute_r=np.median(np.abs(pairs)),
                            maximum_absolute_r=np.abs(pairs).max(),
                            pairs_abs_r_above_0_7=int((np.abs(pairs) > .7).sum()),
                            maximum_VIF=np.diag(np.linalg.inv(correlation)).max(),
                            condition_number=np.linalg.cond(matrix)))
    assert summary[0]['pairs_abs_r_above_0_7'] == 11
    assert summary[1]['pairs_abs_r_above_0_7'] == 4
    for filename in ("Arial.ttf", "Arial-Bold.ttf", "Arial-Italic.ttf", "Arial-BoldItalic.ttf"):
        font_path = ROOT / ".fonts/arial" / filename
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.linewidth": 1, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    width, height = 204.00984, 78
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    a = fig.add_axes([.08, .34, .235, .47])
    b = fig.add_axes([.43, .34, .17, .47])
    c = fig.add_axes([.76, .34, .18, .18 * width / height])
    train = folds.training_mean_fc_correlation.to_numpy()
    test = folds.held_out_fc_correlation.to_numpy()
    a.bar(np.arange(1, 8), test, width=.67, color="#527DA5", zorder=2)
    for i, value in enumerate(test, 1):
        a.text(i, value + .014, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    a.axhline(.58, ls="--", color="black", lw=1, zorder=3)
    a.set(xticks=np.arange(1, 8), xlim=(.4, 7.6), ylim=(0, .82),
          xlabel="LOOCV fold", ylabel="Correlation coefficient")
    limit = np.ceil(np.abs(weights).max() * 10) / 10
    im = b.imshow(weights, cmap="RdBu_r", norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    short_names = ["5-HT", "DA", "ACh", "GABA", "Glu", "CB", "Opioid", "Hist", "NE"]
    b.set(xticks=np.arange(7), xticklabels=np.arange(1, 8), yticks=np.arange(9), yticklabels=short_names,
          xlabel="LOOCV fold")
    b.tick_params(length=0)
    weight_ticks = np.linspace(-limit, limit, 5)
    cb = fig.colorbar(im, cax=fig.add_axes([.612, .34, .009, .47]), ticks=weight_ticks)
    cb.ax.set_title("Weight", fontsize=8, pad=5)
    cb.ax.tick_params(labelsize=9, length=2, pad=1)
    masked = np.ma.masked_where(np.triu(np.ones((9, 9)), 1).astype(bool), corr)
    imc = c.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    c.set(xticks=np.arange(9), xticklabels=short_names, yticks=np.arange(9), yticklabels=short_names)
    c.tick_params(length=0)
    plt.setp(c.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    correlation_ticks = np.linspace(-1, 1, 5)
    cc = fig.colorbar(imc, cax=fig.add_axes([.952, .34, .009, .47]), ticks=correlation_ticks)
    cc.ax.set_title("Pearson's r", fontsize=8, pad=5)
    cc.ax.tick_params(labelsize=9, length=2, pad=1)
    for ax in (b, c):
        for spine in ax.spines.values():
            spine.set_visible(False)
    for letter, x, y in [("a", .015, .865), ("b", .345, .865), ("c", .68, .865)]:
        fig.text(x, y, letter, fontsize=12, fontweight="bold", ha="left", va="bottom")
    fig.savefig(OUT / "neurochemical_validation.png", dpi=300, facecolor="white")
    plt.close(fig)
    folds.to_csv(OUT / "fold_results.csv", index=False)
    pd.DataFrame(weights, index=names, columns=np.arange(1, 8)).to_csv(OUT / "system_weights_by_fold.csv")
    pd.DataFrame(corr, index=names, columns=names).to_csv(OUT / "system_spatial_correlations.csv")
    pd.DataFrame(original_corr, index=RECEPTORS, columns=RECEPTORS).to_csv(OUT / "receptor_spatial_correlations.csv")
    pd.DataFrame(maps, columns=names).to_csv(OUT / "system_maps_80_regions.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "collinearity_summary.csv", index=False)
    legend = (
        "Neurochemical heterogeneity: cross-validation, parameter stability and spatial collinearity.\n"
        "a, Subject-wise leave-one-out validation across seven participants. Each bar shows the "
        "correlation between simulated FC and the held-out participant's empirical FC. "
        "Numbers above bars show correlations to two decimal places. The dashed line at 0.58 "
        "is the previously reported full-sample homogeneous-model reference, with global coupling "
        "optimized but without neurochemical heterogeneity weights. It is not a matched "
        "cross-validated baseline. Its rounded value is documented in the original model script "
        "(Stroke_simulation_whole_brain_model.py, homogeneous parameter fitting); the corresponding "
        "numerical sweep array was not located, so the precise unrounded value was not reverified.\n"
        "b, Nine fitted system weights across seven folds, shown on a shared zero-centered scale "
        "without row normalization; the colorbar is shown to the right. "
        "Stability is across training-subject splits, not repeated optimization seeds on identical data.\n"
        "c, Pearson spatial correlations among nine standardized system representative maps across "
        "80 regions. Multi-receptor maps use PC1, with its sign aligned to the first receptor in "
        "the original implementation; single-receptor systems use their standardized map. "
        "Weights are model-dependent fitted contributions, not excitatory/inhibitory effects.\n"
        "Abbreviations: 5-HT, serotonin; DA, dopamine; ACh, acetylcholine; GABA, "
        "gamma-aminobutyric acid; Glu, glutamate; CB, cannabinoid; Opioid, opioid; "
        "Hist, histamine; NE, norepinephrine.\n"
        "No significance tests. No cross-validated homogeneous-model comparison is shown.\n"
        f"Held-out r: {test.mean():.6f} ± {test.std(ddof=1):.6f} "
        f"(mean ± sample SD).\n"
        f"G: {folds.G.mean():.6f} ± {folds.G.std(ddof=1):.6f}.\n"
        f"Canvas: {width} x {height} mm; Arial; labels 10 pt, ticks 9 pt, "
        "annotations 8 pt, panel letters bold 12 pt; PNG output.\n")
    (OUT / "README_figure_legend.txt").write_text(legend, encoding="utf-8")
    print(legend)
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    args = parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")
    raw, standardized, maps = system_maps()
    if args.validate_only:
        load_backend()
        print("Validation passed: repository inputs and 7-subject HCP dataset are available.")
        return
    folds = run_loso(args.n_trials, args.seed, maps)
    plot_validation(folds, raw, standardized, maps)


if __name__ == "__main__":
    main()
