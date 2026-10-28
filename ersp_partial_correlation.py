"""Recompute ERSP features and five-covariate partial correlations."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/stroke_eeg_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stroke_eeg_matplotlib_cache")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", "/tmp/stroke_eeg_mne_home")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.io import loadmat


FREQUENCIES = ("alpha", "beta1", "beta2")
MI_CONDITIONS = ("Ahand", "Uhand")
ERSP_METRICS = ("ERSP_Contra", "ERSP_Ipsilateral", "Lateral_Index")
MST_METRICS = ("Diameter", "Eccentricity", "Leaf_number", "Tree_hierarchy")
EXCLUDED_SUBJECTS = {"sub-05", "sub-06", "sub-15", "sub-49"}

N_PERMUTATIONS = 9999
N_BOOTSTRAP = 5000
RANDOM_SEED = 2026
N_JOBS = 4
ALTERNATIVE = "two-sided"

PRIMARY_COVARIATES = ("Age", "Gender", "Duration", "ParalysisSide")
RECURRENCE_COVARIATES = PRIMARY_COVARIATES + ("IsFirstTime",)
SEVERITY_COVARIATES = RECURRENCE_COVARIATES + ("MBI", "mRS")
STRATIFIED_COVARIATES = ("Age", "Gender", "Duration", "IsFirstTime")
CONTINUOUS_OR_ORDINAL = {"Age", "Duration", "MBI", "mRS"}


def load_ersp_features(root: Path, subjects: pd.DataFrame) -> dict[str, np.ndarray]:
    try:
        from mne.time_frequency import tfr_array_multitaper
    except ImportError as exc:
        raise RuntimeError("ERSP reconstruction requires MNE-Python.") from exc

    n_subjects = len(subjects)

    erd_left = np.empty((n_subjects, 2, 24), dtype=float)
    erd_right = np.empty_like(erd_left)
    frequencies = np.arange(6.0, 30.0)
    for subject_number in range(1, n_subjects + 1):
        participant_id = f"sub-{subject_number:02d}"
        path = (
            root / "data_load" / "data_clean"
            / f"{participant_id}_task-motor-imagery_eeg.mat"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing uploaded EEG file: {path}")
        eeg = loadmat(path)["EEG_save"]["eeg"][0, 0]
        raw = np.asarray(eeg["rawdata"][0, 0], dtype=float)
        labels = np.asarray(eeg["label"][0, 0]).squeeze().astype(int)
        trigger = np.squeeze(np.where(raw[:, -1, :] == 2))[1, :].astype(int)


        continuous = np.concatenate(raw[:, [13, 14], :], axis=-1)
        global_onset = trigger + np.arange(len(trigger)) * raw.shape[-1]
        epochs = np.stack([
            continuous[:, onset - 1000 : onset + 2501] for onset in global_onset
        ])
        power = tfr_array_multitaper(
            epochs,
            sfreq=500.0,
            freqs=frequencies,
            n_cycles=frequencies,
            use_fft=True,
            output="power",
            decim=2,
            n_jobs=1,
            verbose=False,
        )
        times = -2.0 + np.arange(power.shape[-1]) * 2.0 / 500.0
        baseline = power[..., (times >= -1.0) & (times <= 0.0)].mean(
            axis=-1, keepdims=True
        )
        power_db = 10.0 * np.log10(power / baseline)
        active = power_db[..., (times >= 1.0) & (times < 3.0)].mean(axis=-1)
        if not np.array_equal(np.unique(labels), np.array([1, 2])):
            raise ValueError(f"Unexpected task labels in {path}: {np.unique(labels)}")
        erd_left[subject_number - 1] = active[labels == 1].mean(axis=0)
        erd_right[subject_number - 1] = active[labels == 2].mean(axis=0)
        print(f"ERSP {participant_id}: transformed {len(labels)} trials", flush=True)

    band_slices = (slice(2, 8), slice(8, 15), slice(15, None))
    left_band = np.stack(
        [erd_left[..., band].mean(axis=-1) for band in band_slices], axis=-1
    )
    right_band = np.stack(
        [erd_right[..., band].mean(axis=-1) for band in band_slices], axis=-1
    )
    values = {
        (condition, metric): np.empty((n_subjects, len(FREQUENCIES)), dtype=float)
        for condition in MI_CONDITIONS
        for metric in ERSP_METRICS
    }
    for index, side in enumerate(subjects["ParalysisSide"]):
        if side == "left":
            contra_ahand, ipsi_ahand = left_band[index, 1], left_band[index, 0]
            contra_uhand, ipsi_uhand = right_band[index, 0], right_band[index, 1]
        else:
            contra_uhand, ipsi_uhand = left_band[index, 1], left_band[index, 0]
            contra_ahand, ipsi_ahand = right_band[index, 0], right_band[index, 1]
        values[("Ahand", "ERSP_Contra")][index] = contra_ahand
        values[("Ahand", "ERSP_Ipsilateral")][index] = ipsi_ahand
        values[("Ahand", "Lateral_Index")][index] = contra_ahand - ipsi_ahand
        values[("Uhand", "ERSP_Contra")][index] = contra_uhand
        values[("Uhand", "ERSP_Ipsilateral")][index] = ipsi_uhand
        values[("Uhand", "Lateral_Index")][index] = contra_uhand - ipsi_uhand

    return {
        f"ERSP|{condition}|{frequency}|{metric}": values[(condition, metric)][
            :, frequency_index
        ]
        for condition in MI_CONDITIONS
        for frequency_index, frequency in enumerate(FREQUENCIES)
        for metric in ERSP_METRICS
    }


def covariate_design_matrix(
    clinical: pd.DataFrame, covariates: tuple[str, ...]
) -> tuple[np.ndarray, int]:
    parts = [np.ones((len(clinical), 1), dtype=float)]
    for covariate in covariates:
        if covariate in CONTINUOUS_OR_ORDINAL:
            parts.append(
                stats.rankdata(clinical[covariate].to_numpy(dtype=float)).reshape(-1, 1)
            )
        else:
            dummies = pd.get_dummies(
                clinical[covariate].astype(str), drop_first=True, dtype=float
            )
            if dummies.shape[1]:
                parts.append(dummies.to_numpy(dtype=float))
    design = np.column_stack(parts)
    return design, int(np.linalg.matrix_rank(design))


def partial_spearman_components(
    feature: np.ndarray, outcome: np.ndarray, design: np.ndarray, design_rank: int
) -> tuple[float, np.ndarray, np.ndarray, float, int]:
    feature_rank = stats.rankdata(np.asarray(feature, dtype=float))
    outcome_rank = stats.rankdata(np.asarray(outcome, dtype=float))
    feature_residual = feature_rank - design @ np.linalg.lstsq(
        design, feature_rank, rcond=None
    )[0]
    outcome_residual = outcome_rank - design @ np.linalg.lstsq(
        design, outcome_rank, rcond=None
    )[0]
    rho = float(np.corrcoef(feature_residual, outcome_residual)[0, 1])
    rho = float(np.clip(rho, -1, 1))
    degrees_freedom = len(feature_rank) - design_rank - 1
    statistic = rho * np.sqrt(degrees_freedom / max(1e-15, 1 - rho**2))
    p_two_sided = float(2 * stats.t.sf(abs(statistic), degrees_freedom))
    return rho, feature_residual, outcome_residual, p_two_sided, degrees_freedom


def partial_permutation_pvalue(
    feature_residual: np.ndarray,
    outcome_residual: np.ndarray,
    alternative: str = ALTERNATIVE,
) -> float:
    x = feature_residual - feature_residual.mean()
    y = outcome_residual - outcome_residual.mean()
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    observed = float(np.dot(x, y))
    rng = np.random.default_rng(RANDOM_SEED)
    count = 0
    batch_size = 1000
    for start in range(0, N_PERMUTATIONS, batch_size):
        size = min(batch_size, N_PERMUTATIONS - start)
        permuted = np.asarray([rng.permutation(x) for _ in range(size)])
        null_rho = permuted @ y
        if alternative == "less":
            count += int(np.count_nonzero(null_rho <= observed))
        elif alternative == "greater":
            count += int(np.count_nonzero(null_rho >= observed))
        else:
            count += int(np.count_nonzero(np.abs(null_rho) >= abs(observed)))
    return (count + 1) / (N_PERMUTATIONS + 1)


def partial_bootstrap_ci(
    feature: np.ndarray,
    outcome: np.ndarray,
    clinical: pd.DataFrame,
    covariates: tuple[str, ...],
) -> tuple[float, float]:
    feature = np.asarray(feature, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_rho = np.empty(N_BOOTSTRAP, dtype=float)
    for bootstrap_index in range(N_BOOTSTRAP):
        indices = rng.integers(0, len(feature), size=len(feature))
        sampled_clinical = clinical.iloc[indices].reset_index(drop=True)
        design, design_rank = covariate_design_matrix(sampled_clinical, covariates)
        try:
            rho, _, _, _, _ = partial_spearman_components(
                feature[indices], outcome[indices], design, design_rank
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            rho = np.nan
        bootstrap_rho[bootstrap_index] = rho
    bootstrap_rho = bootstrap_rho[np.isfinite(bootstrap_rho)]
    if not len(bootstrap_rho):
        return np.nan, np.nan
    return tuple(np.quantile(bootstrap_rho, [0.025, 0.975]))


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def analyze_feature(
    feature_name: str,
    feature: np.ndarray,
    outcome: np.ndarray,
    clinical: pd.DataFrame,
    covariates: tuple[str, ...],
    subgroup: str,
    adjustment: str,
) -> dict[str, object]:
    design, design_rank = covariate_design_matrix(clinical, covariates)
    rho, feature_residual, outcome_residual, p_parametric, degrees_freedom = (
        partial_spearman_components(feature, outcome, design, design_rank)
    )
    p_permutation = partial_permutation_pvalue(
        feature_residual, outcome_residual, alternative=ALTERNATIVE
    )
    ci_low, ci_high = partial_bootstrap_ci(feature, outcome, clinical, covariates)
    domain, condition, frequency, metric = feature_name.split("|", 3)
    family_size = len(ERSP_METRICS) if domain == "ERSP" else len(MST_METRICS)
    return {
        "subgroup": subgroup,
        "adjustment": adjustment,
        "Domain": domain,
        "MI": condition,
        "Frequency": frequency,
        "Metric": metric,
        "feature": feature_name,
        "N": len(feature),
        "covariates": ";".join(covariates),
        "Effect_size": "partial Spearman rho",
        "partial_spearman_rho": rho,
        "CI_level": 0.95,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "P_raw": p_permutation,
        "Alternative": ALTERNATIVE,
        "P_parametric_two_sided": p_parametric,
        "degrees_freedom": degrees_freedom,
        "FDR_method": "Benjamini-Hochberg",
        "FDR_family": f"{domain}|{condition}|{frequency}",
        "FDR_family_size": family_size,
        "N_permutations": N_PERMUTATIONS,
        "N_bootstrap": N_BOOTSTRAP,
        "Total_tests_ERSP_plus_MST": 42,
    }


def partial_correlation_table(
    x: pd.DataFrame,
    y: pd.Series,
    clinical: pd.DataFrame,
    covariates: tuple[str, ...],
    subgroup: str,
    adjustment: str,
) -> pd.DataFrame:
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(analyze_feature)(
            feature,
            x[feature].to_numpy(dtype=float),
            y.to_numpy(dtype=float),
            clinical,
            covariates,
            subgroup,
            adjustment,
        )
        for feature in x.columns
    )
    result = pd.DataFrame(rows)
    result["P_FDR_BH"] = np.nan
    family_columns = ["subgroup", "adjustment", "Domain", "MI", "Frequency"]
    for _, indices in result.groupby(family_columns).groups.items():
        result.loc[indices, "P_FDR_BH"] = benjamini_hochberg(result.loc[indices, "P_raw"])
    result["significant_raw_0.05"] = result["P_raw"] < 0.05
    result["significant_FDR_0.05"] = result["P_FDR_BH"] < 0.05
    return result.sort_values(
        ["subgroup", "adjustment", "Domain", "MI", "Frequency", "P_raw", "Metric"]
    ).reset_index(drop=True)


def find_project_root(script_dir: Path) -> Path:
    if not (script_dir / "data_load" / "subject.csv").exists():
        raise FileNotFoundError(
            f"Missing {script_dir / 'data_load' / 'subject.csv'}; "
            "run this script from a complete clone of the repository."
        )
    return script_dir


def run_domain_analysis(domain: str, script_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = find_project_root(script_dir)
    subjects = pd.read_csv(root / "data_load" / "subject.csv")
    feature_values = load_ersp_features(root, subjects)
    selected = pd.DataFrame(
        feature_values, index=subjects["Participant_ID"].astype(str)
    )
    outcome = pd.Series(
        subjects["NIHSS"].to_numpy(float), index=selected.index, name="NIHSS"
    )
    clinical = subjects.copy()
    clinical["Participant_ID"] = clinical["Participant_ID"].astype(str)
    clinical = clinical.set_index("Participant_ID")
    keep = ~selected.index.isin(EXCLUDED_SUBJECTS)
    selected, outcome, clinical = (
        selected.loc[keep], outcome.loc[keep], clinical.loc[keep]
    )
    if selected.shape != (46, 18):
        raise ValueError(f"Expected 46 x 18, found {selected.shape}.")
    required = set(RECURRENCE_COVARIATES)
    if selected.isna().any().any() or outcome.isna().any() or clinical[list(required)].isna().any().any():
        raise ValueError("Missing values detected in features, NIHSS, or covariates.")
    results = partial_correlation_table(
        selected,
        outcome,
        clinical,
        RECURRENCE_COVARIATES,
        subgroup="all",
        adjustment="primary_five_covariates",
    ).rename(columns={"P_raw": "P_two_sided", "P_FDR_BH": "P_FDR_BH_two_sided"})

    design, design_rank = covariate_design_matrix(clinical, RECURRENCE_COVARIATES)
    source_rows = []
    for feature in selected.columns:
        _, feature_residual, outcome_residual, _, _ = partial_spearman_components(
            selected[feature].to_numpy(float), outcome.to_numpy(float), design, design_rank
        )
        source_rows.append(
            pd.DataFrame({
                "Participant_ID": selected.index,
                "feature": feature,
                "NIHSS_rank_residual": outcome_residual,
                "feature_rank_residual": feature_residual,
            })
        )
    return results, pd.concat(source_rows, ignore_index=True)


def plot_domain_panels(
    domain: str,
    results: pd.DataFrame,
    source: pd.DataFrame,
    figure_dir: Path,
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stroke_partial_repro_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns

    font_dir = find_project_root(Path(__file__).resolve().parent) / ".fonts" / "arial"
    for font_file in font_dir.glob("*.ttf") if font_dir.exists() else ():
        font_manager.fontManager.addfont(str(font_file))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    labels = {
        "ERSP_Contra": "Contralateral ERSP",
        "ERSP_Ipsilateral": "Ipsilateral ERSP",
        "Lateral_Index": "Lateral index",
        "Diameter": "Diameter",
        "Eccentricity": "Eccentricity range",
        "Leaf_number": "Leaf number",
        "Tree_hierarchy": "Tree hierarchy",
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    for _, row in results.iterrows():
        feature = str(row["feature"])
        data = source[source["feature"].eq(feature)]
        fig, ax = plt.subplots(figsize=(48 / 25.4, 44 / 25.4), layout="constrained")
        sns.regplot(
            data=data,
            x="NIHSS_rank_residual",
            y="feature_rank_residual",
            color="royalblue",
            ci=95,
            n_boot=5000,
            seed=RANDOM_SEED,
            scatter_kws={"color": "royalblue", "s": 8},
            line_kws={"color": "royalblue", "linewidth": 1.5},
            ax=ax,
        )
        rho = float(row["partial_spearman_rho"])
        p_fdr = float(row["P_FDR_BH_two_sided"])
        if p_fdr < 0.001:
            p_text = "<0.001"
        elif 0.0495 <= p_fdr < 0.05:
            p_text = f"={p_fdr:.4f}"
        else:
            p_text = f"={p_fdr:.3f}"
        ax.text(
            .03 if rho >= 0 else .97,
            .97,
            f"r={rho:.2f}    p{p_text}",
            transform=ax.transAxes,
            ha="left" if rho >= 0 else "right",
            va="top",
            fontsize=8,
        )
        ax.set_xlabel("NIHSS rank residuals")
        ax.set_ylabel(f"{labels[str(row['Metric'])]}\nrank residuals")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(direction="in", length=3, width=1.0, pad=2)
        domain_name, condition, frequency, metric = feature.split("|", 3)
        hand = {"Ahand": "affected_hand", "Uhand": "unaffected_hand"}[condition]
        band = {"alpha": "alpha", "beta1": "low_beta", "beta2": "high_beta"}[frequency]
        slug = f"{domain_name.lower()}_{metric.lower()}_{band}_{hand}"
        fig.savefig(
            figure_dir / f"partial_regression_{slug}.png",
            dpi=600,
            facecolor="white",
        )
        plt.close(fig)


def run_and_export_domain(domain: str, script_dir: Path) -> None:
    results, source = run_domain_analysis(domain, script_dir)
    result_root = script_dir / "results" / f"{domain.lower()}_partial_correlation"
    figure_root = script_dir / "figures" / f"{domain.lower()}_partial_correlation"
    result_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(result_root / "partial_spearman_results.csv", index=False)
    source.to_csv(result_root / "partial_correlation_plot_source.csv", index=False)
    plot_domain_panels(domain, results, source, figure_root)
    print(f"Saved {len(results)} {domain} tests to {result_root}")
    print(f"Saved {len(results)} PNG panels to {figure_root}")


if __name__ == "__main__":
    run_and_export_domain("ERSP", Path(__file__).resolve().parent)
