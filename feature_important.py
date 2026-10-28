"""Run nested five-fold PLS prediction and full-cohort SHAP interpretation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stroke_pls_shap_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/stroke_pls_shap_numba")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", "/tmp/stroke_pls_shap_mne_home")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm, pearsonr, spearmanr
from scipy import stats
from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "results" / "pls_nested5"
FIGURE_DIR = SCRIPT_DIR / "figures" / "pls_nested5"
SHAP_OUTPUT_DIR = SCRIPT_DIR / "results" / "pls_shap"

SEED = 0
OUTER_SPLITS = 5
INNER_SPLITS = 5
TOP_K_CANDIDATES = list(range(1, 11))

FREQUENCIES = ("alpha", "beta1", "beta2")
MI_CONDITIONS = ("Ahand", "Uhand")
ERSP_METRICS = ("ERSP_Contra", "ERSP_Ipsilateral", "Lateral_Index")
MST_METRICS = ("Diameter", "Eccentricity", "Leaf_number", "Tree_hierarchy")
EXCLUDED_SUBJECTS = {"sub-05", "sub-06", "sub-15", "sub-49"}
SIDE_COVARIATE = "Clinical|ParalysisSide_right"

def mst_network(connectivity: np.ndarray) -> np.ndarray:
    graph = nx.from_numpy_array(connectivity)
    tree = nx.maximum_spanning_tree(graph, weight="weight")
    eccentricity = nx.eccentricity(tree)
    leaves = [node for node in tree.nodes if tree.degree(node) == 1]
    betweenness = nx.betweenness_centrality(tree)
    hierarchy = len(leaves) / (2 * (len(tree) - 1) * max(betweenness.values()))
    return np.asarray(
        [
            nx.diameter(tree),
            max(eccentricity.values()) - min(eccentricity.values()),
            len(leaves),
            hierarchy,
        ],
        dtype=float,
    )


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
        path = root / "data_load" / "data_clean" / (
            f"{participant_id}_task-motor-imagery_eeg.mat"
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
            epochs, sfreq=500.0, freqs=frequencies, n_cycles=frequencies,
            use_fft=True, output="power", decim=2, n_jobs=1, verbose=False,
        )
        times = -2.0 + np.arange(power.shape[-1]) * 2.0 / 500.0
        baseline = power[..., (times >= -1.0) & (times <= 0.0)].mean(
            axis=-1, keepdims=True
        )
        active = (10.0 * np.log10(power / baseline))[
            ..., (times >= 1.0) & (times < 3.0)
        ].mean(axis=-1)
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


def load_mst_features(root: Path, subjects: pd.DataFrame) -> dict[str, np.ndarray]:
    left = np.load(root / "data_load" / "ImCoh_data" / "imcoh_left.npy")
    right = np.load(root / "data_load" / "ImCoh_data" / "imcoh_right.npy")
    n_subjects = len(subjects)
    left_stats = np.empty((n_subjects, len(FREQUENCIES), len(MST_METRICS)), dtype=float)
    right_stats = np.empty_like(left_stats)

    for index in range(n_subjects):
        for frequency_index in range(len(FREQUENCIES)):
            left_stats[index, frequency_index] = mst_network(
                left[index, :, :, frequency_index]
            )
            right_stats[index, frequency_index] = mst_network(
                right[index, :, :, frequency_index]
            )

    affected = np.empty_like(left_stats)
    unaffected = np.empty_like(left_stats)
    for index, side in enumerate(subjects["ParalysisSide"]):
        if side == "left":
            affected[index], unaffected[index] = left_stats[index], right_stats[index]
        else:
            affected[index], unaffected[index] = right_stats[index], left_stats[index]

    condition_values = {"Ahand": affected, "Uhand": unaffected}
    return {
        f"MST|{condition}|{frequency}|{metric}": condition_values[condition][
            :, frequency_index, metric_index
        ]
        for condition in MI_CONDITIONS
        for frequency_index, frequency in enumerate(FREQUENCIES)
        for metric_index, metric in enumerate(MST_METRICS)
    }


def build_feature_table(root: Path) -> tuple[pd.DataFrame, pd.Series]:
    subjects = pd.read_csv(root / "data_load" / "subject.csv")
    features: dict[str, np.ndarray] = {}
    features.update(load_ersp_features(root, subjects))
    features.update(load_mst_features(root, subjects))
    table = pd.DataFrame(features, index=subjects["Participant_ID"].astype(str))
    outcome = pd.Series(
        subjects["NIHSS"].to_numpy(dtype=float), index=table.index, name="NIHSS"
    )
    keep = ~table.index.isin(EXCLUDED_SUBJECTS)
    table, outcome = table.loc[keep], outcome.loc[keep]
    if table.shape != (46, 42):
        raise ValueError(f"Expected a 46 x 42 feature table, found {table.shape}.")
    if table.isna().any().any() or outcome.isna().any():
        raise ValueError("Missing values found; fold-local imputation would be required.")
    return table, outcome


def rank_features(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for feature in x.columns:
        result = spearmanr(x[feature], y, nan_policy="omit")
        rows.append(
            {
                "feature": feature,
                "spearman_rho": float(result.statistic),
                "p_value_two_sided": float(result.pvalue),
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["_sort_abs_rho"] = ranking["spearman_rho"].abs().fillna(-np.inf)
    ranking = (
        ranking.sort_values(["_sort_abs_rho", "feature"], ascending=[False, True])
        .drop(columns="_sort_abs_rho")
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def fit_predict_pls(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    selected: list[str],
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = PLSRegression(n_components=n_components)
    model.fit(x_train[selected], y_train)
    predictions = np.asarray(model.predict(x_test[selected])).reshape(-1)
    coefficients = np.asarray(model.coef_).reshape(-1)
    return predictions, coefficients


def choose_topk_and_components(
    x: pd.DataFrame, y: pd.Series
) -> tuple[int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    fold_cache = []
    ranking_rows = []
    for inner_fold, (train_index, validation_index) in enumerate(splitter.split(x), start=1):
        x_train, x_validation = x.iloc[train_index], x.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
        ranking = rank_features(x_train, y_train)
        detailed = ranking.copy()
        detailed.insert(0, "inner_fold", inner_fold)
        for top_k in TOP_K_CANDIDATES:
            detailed[f"selected_at_K_{top_k}"] = detailed["rank"] <= top_k
        ranking_rows.append(detailed)
        fold_cache.append(
            (inner_fold, x_train, x_validation, y_train, y_validation, ranking)
        )

    score_rows = []
    for top_k in TOP_K_CANDIDATES:
        for n_components in range(1, top_k + 1):
            for (
                inner_fold,
                x_train,
                x_validation,
                y_train,
                y_validation,
                ranking,
            ) in fold_cache:
                selected = ranking.head(top_k)["feature"].tolist()
                prediction, _ = fit_predict_pls(
                    x_train, y_train, x_validation, selected, n_components
                )
                score_rows.append(
                    {
                        "inner_fold": inner_fold,
                        "top_k": top_k,
                        "n_components": n_components,
                        "R2": r2_score(y_validation, prediction),
                        "RMSE": np.sqrt(mean_squared_error(y_validation, prediction)),
                        "MAE": mean_absolute_error(y_validation, prediction),
                    }
                )

    fold_scores = pd.DataFrame(score_rows)
    candidate_summary = (
        fold_scores.groupby(["top_k", "n_components"], as_index=False)
        .agg(
            mean_inner_R2=("R2", "mean"),
            sd_inner_R2=("R2", "std"),
            mean_inner_RMSE=("RMSE", "mean"),
            mean_inner_MAE=("MAE", "mean"),
        )
        .sort_values(["top_k", "n_components"])
    )
    best = candidate_summary.sort_values(
        ["mean_inner_R2", "top_k", "n_components"], ascending=[False, True, True]
    ).iloc[0]
    return (
        int(best["top_k"]),
        int(best["n_components"]),
        candidate_summary,
        fold_scores,
        pd.concat(ranking_rows, ignore_index=True),
    )


def run_nested_five_fold(x: pd.DataFrame, y: pd.Series):
    outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    predictions, fold_metrics = [], []
    outer_rankings, inner_rankings = [], []
    tuning_summaries, tuning_fold_scores, coefficient_rows = [], [], []

    for outer_fold, (train_index, test_index) in enumerate(outer.split(x), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        best_k, best_components, tuning, inner_scores, inner_ranking = (
            choose_topk_and_components(x_train, y_train)
        )
        tuning.insert(0, "outer_fold", outer_fold)
        tuning_summaries.append(tuning)
        inner_scores.insert(0, "outer_fold", outer_fold)
        tuning_fold_scores.append(inner_scores)
        inner_ranking.insert(0, "outer_fold", outer_fold)
        inner_rankings.append(inner_ranking)

        ranking = rank_features(x_train, y_train)
        ranking["selected"] = ranking["rank"] <= best_k
        ranking.insert(0, "outer_fold", outer_fold)
        ranking.insert(1, "test_subjects", ";".join(x_test.index))
        ranking.insert(2, "chosen_top_k", best_k)
        outer_rankings.append(ranking)
        selected = ranking.loc[ranking["selected"], "feature"].tolist()

        prediction, coefficients = fit_predict_pls(
            x_train, y_train, x_test, selected, best_components
        )
        actual = y_test.to_numpy(dtype=float)
        fold_pearson = pearsonr(actual, prediction)
        for local_index, subject_position in enumerate(test_index):
            predictions.append(
                {
                    "outer_fold": outer_fold,
                    "Participant_ID": x.index[subject_position],
                    "NIHSS_actual": float(actual[local_index]),
                    "NIHSS_predicted": float(prediction[local_index]),
                    "top_k": best_k,
                    "n_components": best_components,
                }
            )
        fold_metrics.append(
            {
                "outer_fold": outer_fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "test_subjects": ";".join(x_test.index),
                "top_k": best_k,
                "n_components": best_components,
                "R2": r2_score(actual, prediction),
                "RMSE": np.sqrt(mean_squared_error(actual, prediction)),
                "MAE": mean_absolute_error(actual, prediction),
                "Pearson_r": float(fold_pearson.statistic),
                "Pearson_p_two_sided": float(fold_pearson.pvalue),
            }
        )
        coefficient_rows.extend(
            {
                "outer_fold": outer_fold,
                "feature": feature,
                "standardized_coefficient": float(value),
            }
            for feature, value in zip(selected, coefficients)
        )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(fold_metrics),
        pd.concat(outer_rankings, ignore_index=True),
        pd.concat(inner_rankings, ignore_index=True),
        pd.concat(tuning_summaries, ignore_index=True),
        pd.concat(tuning_fold_scores, ignore_index=True),
        pd.DataFrame(coefficient_rows),
    )


def summarize(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    actual = predictions["NIHSS_actual"].to_numpy(dtype=float)
    predicted = predictions["NIHSS_predicted"].to_numpy(dtype=float)
    pooled_pearson = pearsonr(actual, predicted)
    fisher_z = np.arctanh(np.clip(float(pooled_pearson.statistic), -0.999999, 0.999999))
    fisher_se = 1 / np.sqrt(len(actual) - 3)
    critical_value = norm.ppf(0.975)
    pearson_ci_low, pearson_ci_high = np.tanh(
        [fisher_z - critical_value * fisher_se, fisher_z + critical_value * fisher_se]
    )
    return pd.DataFrame(
        [
            {
                "N": len(predictions),
                "outer_cv": f"5-fold_shuffled_seed{SEED}",
                "ranking": "training_only_absolute_Spearman",
                "top_k_candidates": ";".join(map(str, TOP_K_CANDIDATES)),
                "mean_of_5_fold_R2": fold_metrics["R2"].mean(),
                "sd_of_5_fold_R2": fold_metrics["R2"].std(ddof=1),
                "mean_of_5_fold_RMSE": fold_metrics["RMSE"].mean(),
                "mean_of_5_fold_MAE": fold_metrics["MAE"].mean(),
                "pooled_OOF_R2": r2_score(actual, predicted),
                "pooled_OOF_RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "pooled_OOF_MAE": mean_absolute_error(actual, predicted),
                "pooled_OOF_Pearson_r": float(pooled_pearson.statistic),
                "pooled_OOF_Pearson_p_two_sided": float(pooled_pearson.pvalue),
                "pooled_OOF_Pearson_r_CI95_low": float(pearson_ci_low),
                "pooled_OOF_Pearson_r_CI95_high": float(pearson_ci_high),
            }
        ]
    )


def write_summary(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    selection_frequency: pd.DataFrame,
) -> None:
    lines = [
        "Nested five-fold PLS",
        "======================",
        "",
        "Design:",
        f"- Outer CV: shuffled {OUTER_SPLITS}-fold, random_state={SEED}.",
        f"- Inner CV: shuffled {INNER_SPLITS}-fold, random_state={SEED}.",
        "- Model: sklearn PLSRegression with default internal scaling.",
        "- Ranking: training-fold absolute Spearman rho, largest first.",
        f"- Candidate K: {TOP_K_CANDIDATES}.",
        "- Candidate components: 1 to K.",
        "- Joint tuning: highest mean inner-fold R2; ties favor simpler models.",
        "- Ranking and tuning are repeated inside every relevant training split.",
        "- Pooled and fold-wise Pearson correlations compare actual and OOF-predicted NIHSS.",
        "",
        "Overall metrics:",
        metrics.to_string(index=False),
        "",
        "Outer-fold metrics:",
        fold_metrics.to_string(index=False),
        "",
        "Feature-selection frequency across outer training folds:",
        selection_frequency.to_string(index=False),
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_nested_cv_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x, y = build_feature_table(ROOT)
    (
        predictions,
        fold_metrics,
        outer_rankings,
        inner_rankings,
        tuning,
        tuning_fold_scores,
        coefficients,
    ) = run_nested_five_fold(x, y)
    metrics = summarize(predictions, fold_metrics)

    selected_outer = outer_rankings.loc[outer_rankings["selected"]]
    selection_frequency = (
        selected_outer.groupby("feature", as_index=False)
        .agg(selection_count=("outer_fold", "nunique"), mean_selected_rank=("rank", "mean"))
        .set_index("feature")
        .reindex(x.columns)
        .rename_axis("feature")
        .reset_index()
        .assign(selection_count=lambda frame: frame["selection_count"].fillna(0).astype(int))
        .assign(selection_frequency=lambda frame: frame["selection_count"] / OUTER_SPLITS)
        .sort_values(
            ["selection_frequency", "mean_selected_rank", "feature"],
            ascending=[False, True, True],
        )
    )
    predictions.to_csv(OUTPUT_DIR / "outer_fold_predictions.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "outer_fold_metrics.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)
    outer_rankings.to_csv(OUTPUT_DIR / "outer_fold_feature_rankings.csv", index=False)
    inner_rankings.to_csv(OUTPUT_DIR / "inner_fold_feature_rankings.csv", index=False)
    tuning.to_csv(OUTPUT_DIR / "inner_cv_topk_component_scores.csv", index=False)
    tuning_fold_scores.to_csv(OUTPUT_DIR / "inner_cv_fold_scores.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "outer_fold_coefficients.csv", index=False)
    selection_frequency.to_csv(OUTPUT_DIR / "feature_selection_frequency.csv", index=False)
    write_summary(metrics, fold_metrics, selection_frequency)

    print(metrics.to_string(index=False))
    print(f"Results written to {OUTPUT_DIR}")
    print(f"Figures written to {FIGURE_DIR}")


def shap_feature_label(feature: str) -> str:
    clinical_labels = {
        "Clinical|Age": "Age",
        "Clinical|Gender_female": "Sex",
        "Clinical|Duration": "Post-stroke duration",
        "Clinical|RecurrentStroke": "Stroke history",
        SIDE_COVARIATE: "Paralysis side",
    }
    if feature in clinical_labels:
        return clinical_labels[feature]
    _, hand, band, metric = feature.split("|")
    hand = {"Ahand": "A", "Uhand": "U"}[hand]
    band = {"alpha": "a", "beta1": "l", "beta2": "h"}[band]
    metric = {
        "ERSP_Contra": "Contralateral ERSP",
        "ERSP_Ipsilateral": "Ipsilateral ERSP",
        "Lateral_Index": "Lateral Index",
        "Leaf_number": "Leaf Number",
        "Tree_hierarchy": "Tree Hierarchy",
        "Eccentricity": "Eccentricity Range",
        "Diameter": "Diameter",
    }[metric]
    return f"{metric} ({hand}{band})"


def tune_full_cohort_with_covariates(
    x: pd.DataFrame, y: pd.Series, covariates: pd.DataFrame
):
    cache, ranking_rows, score_rows = [], [], []
    splitter = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train, validation) in enumerate(splitter.split(x), start=1):
        ranking = rank_features(x.iloc[train], y.iloc[train])
        ranking_rows.append(ranking.assign(inner_fold=fold))
        cache.append(
            (
                fold,
                x.iloc[train].join(covariates.iloc[train]),
                x.iloc[validation].join(covariates.iloc[validation]),
                y.iloc[train],
                y.iloc[validation],
                ranking,
            )
        )
    for top_k in TOP_K_CANDIDATES:
        for n_components in range(1, top_k + covariates.shape[1] + 1):
            for fold, train_x, validation_x, train_y, validation_y, ranking in cache:
                selected = (
                    ranking.head(top_k).feature.tolist()
                    + covariates.columns.tolist()
                )
                prediction, _ = fit_predict_pls(
                    train_x,
                    train_y,
                    validation_x,
                    selected,
                    n_components,
                )
                score_rows.append(
                    {
                        "inner_fold": fold,
                        "top_k": top_k,
                        "n_components": n_components,
                        "n_total_predictors": top_k + covariates.shape[1],
                        "mandatory_covariates": ";".join(covariates.columns),
                        "all_covariates_forced_included": True,
                        "R2": r2_score(validation_y, prediction),
                        "RMSE": np.sqrt(
                            mean_squared_error(validation_y, prediction)
                        ),
                        "MAE": mean_absolute_error(validation_y, prediction),
                    }
                )
    scores = pd.DataFrame(score_rows)
    tuning = scores.groupby(["top_k", "n_components"], as_index=False).agg(
        mean_inner_R2=("R2", "mean"),
        sd_inner_R2=("R2", "std"),
        mean_inner_RMSE=("RMSE", "mean"),
        mean_inner_MAE=("MAE", "mean"),
    )
    best = tuning.sort_values(
        ["mean_inner_R2", "top_k", "n_components"],
        ascending=[False, True, True],
    ).iloc[0]
    return (
        int(best.top_k),
        int(best.n_components),
        tuning,
        scores,
        pd.concat(ranking_rows, ignore_index=True),
    )


def run_full_cohort_shap(
    include_paralysis_side: bool = False,
    include_clinical_covariates: bool = False,
) -> None:
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "The SHAP analysis requires the optional 'shap' package. "
            "Install it in the active environment before running "
            "--analysis shap or --analysis all."
        ) from error

    include_side = include_paralysis_side or include_clinical_covariates
    output_dir = SHAP_OUTPUT_DIR
    if include_side:
        suffix = (
            "with_clinical" if include_clinical_covariates else "with_side"
        )
        output_dir = output_dir.with_name(f"pls_shap_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print(
        "Building the same 46 x 42 feature table as the nested-CV analysis...",
        flush=True,
    )
    x, y = build_feature_table(ROOT)
    assert x.shape == (46, 42) and x.index.equals(y.index)
    assert np.isfinite(x.to_numpy()).all() and np.isfinite(y.to_numpy()).all()

    covariates = pd.DataFrame(index=x.index)
    clinical = None
    if include_side:
        clinical = (
            pd.read_csv(ROOT / "data_load" / "subject.csv")
            .set_index("Participant_ID")
            .loc[x.index]
        )
        assert clinical.ParalysisSide.isin(["left", "right"]).all()
        covariates[SIDE_COVARIATE] = clinical.ParalysisSide.eq("right").astype(float)
        if include_clinical_covariates:
            assert clinical.Gender.isin(["male", "female"]).all()
            assert clinical.IsFirstTime.isin(["yes", "no"]).all()
            covariates["Clinical|Age"] = clinical.Age.astype(float)
            covariates["Clinical|Gender_female"] = clinical.Gender.eq(
                "female"
            ).astype(float)
            covariates["Clinical|Duration"] = clinical.Duration.astype(float)
            covariates["Clinical|RecurrentStroke"] = clinical.IsFirstTime.eq(
                "no"
            ).astype(float)
        assert np.isfinite(covariates.to_numpy()).all()
        top_k, n_components, tuning, scores, rankings = (
            tune_full_cohort_with_covariates(x, y, covariates)
        )
    else:
        top_k, n_components, tuning, scores, rankings = (
            choose_topk_and_components(x, y)
        )

    full_ranking = rank_features(x, y)
    full_ranking["selected"] = full_ranking["rank"] <= top_k
    selected = full_ranking.head(top_k).feature.tolist()
    full_ranking["selection_role"] = "EEG_candidate"
    if include_side:
        x = x.join(covariates)
        selected.extend(covariates.columns)
        full_ranking = pd.concat(
            [
                full_ranking,
                pd.DataFrame(
                    [
                        {
                            "feature": feature,
                            "selected": True,
                            "selection_role": "mandatory_covariate",
                        }
                        for feature in covariates.columns
                    ]
                ),
            ],
            ignore_index=True,
        )

    selected_x = x[selected]
    model = PLSRegression(n_components=n_components).fit(selected_x, y)
    prediction = model.predict(selected_x).reshape(-1)

    center = selected_x.mean(axis=0).to_numpy()
    step = selected_x.std(axis=0, ddof=1).to_numpy()
    assert np.all(step > 0)
    base = float(
        model.predict(pd.DataFrame([center], columns=selected)).item()
    )
    probes = pd.DataFrame(center + np.diag(step), columns=selected)
    coefficients = (model.predict(probes).reshape(-1) - base) / step
    intercept = base - center @ coefficients
    affine_prediction = selected_x.to_numpy() @ coefficients + intercept
    affine_error = float(np.max(np.abs(affine_prediction - prediction)))
    np.testing.assert_allclose(
        affine_prediction, prediction, atol=1e-9, rtol=1e-9
    )

    masker = shap.maskers.Independent(selected_x, max_samples=len(selected_x))
    explainer = shap.LinearExplainer((coefficients, intercept), masker)
    explanation = explainer(selected_x)
    reconstruction = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(
        reconstruction, prediction, atol=1e-9, rtol=1e-9
    )
    np.testing.assert_allclose(
        explanation.values,
        (selected_x.to_numpy() - center) * coefficients,
        atol=1e-9,
        rtol=1e-9,
    )

    importance = pd.DataFrame(
        {
            "feature": selected,
            "mean_abs_SHAP": np.abs(explanation.values).mean(axis=0),
            "raw_unit_coefficient": coefficients,
            "background_mean": center,
        }
    ).sort_values("mean_abs_SHAP", ascending=False, kind="stable")
    importance.insert(0, "importance_rank", np.arange(1, len(selected) + 1))
    all_features = full_ranking.merge(importance, on="feature", how="left")
    all_features["mean_abs_SHAP"] = all_features["mean_abs_SHAP"].fillna(0)
    all_features["raw_unit_coefficient"] = all_features[
        "raw_unit_coefficient"
    ].fillna(0)
    all_features["attribution_status"] = np.where(
        all_features.selected,
        "included_in_final_model",
        "not_selected_zero_model_contribution",
    )

    x.assign(NIHSS=y).to_csv(
        output_dir / "full_cohort_source_data.csv", index_label="Participant_ID"
    )
    for frame, name in (
        (tuning, "tuning_candidate_scores"),
        (scores, "tuning_fold_scores"),
        (rankings, "tuning_fold_rankings"),
        (full_ranking, "full_data_feature_ranking"),
        (importance, "shap_importance_selected"),
        (all_features, f"all{len(full_ranking)}_feature_summary"),
    ):
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    pd.DataFrame(
        explanation.values, index=x.index, columns=selected
    ).to_csv(output_dir / "shap_values.csv", index_label="Participant_ID")
    pd.DataFrame(
        {
            "NIHSS_observed": y,
            "final_model_fitted_NIHSS": prediction,
            "SHAP_base_value": explanation.base_values,
            "SHAP_reconstructed_prediction": reconstruction,
        }
    ).to_csv(
        output_dir / "fitted_predictions_and_shap_check.csv",
        index_label="Participant_ID",
    )
    selected_x.corr().to_csv(
        output_dir / "selected_feature_pearson_correlations.csv"
    )
    joblib.dump(
        {
            "model": model,
            "selected_features": selected,
            "seed": SEED,
            "background": selected_x,
            "raw_coefficients": coefficients,
            "raw_intercept": intercept,
        },
        output_dir / "final_pls_model.joblib",
    )

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {
        "analysis": "full_cohort_pls_shap",
        "n_subjects": 46,
        "n_candidate_features": 42,
        "seed": SEED,
        "top_k": top_k,
        "n_components": n_components,
        "background_n": len(selected_x),
        "mandatory_covariates": covariates.columns.tolist(),
        "clinical_encoding": (
            {
                "ParalysisSide": "left=0, right=1",
                "Gender": "male=0, female=1",
                "IsFirstTime": "yes=0, no=1 (recurrent)",
                "Age_and_Duration": (
                    "original numeric units; PLS internally scales inputs"
                ),
            }
            if include_clinical_covariates
            else None
        ),
        "n_total_predictors": len(selected),
        "side_coding": {"left": 0, "right": 1} if include_side else None,
        "side_counts": (
            clinical.ParalysisSide.value_counts().to_dict()
            if clinical is not None
            else None
        ),
        "base_value_NIHSS": base,
        "selected_features": selected,
        "max_affine_prediction_error": affine_error,
        "max_SHAP_additivity_error": float(
            np.max(np.abs(reconstruction - prediction))
        ),
        "versions": {
            "sklearn": sklearn.__version__,
            "shap": shap.__version__,
            "numpy": np.__version__,
        },
        "combined_script_sha256": script_hash,
        "data_sha256": hashlib.sha256(
            (output_dir / "full_cohort_source_data.csv").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    mode_flag = (
        " --include-clinical-covariates"
        if include_clinical_covariates
        else " --include-paralysis-side"
        if include_side
        else ""
    )
    (output_dir / "README.md").write_text(
        "# Exploratory final-model SHAP analysis\n\n"
        f"All 46 participants; 42 candidates; seed={SEED}; selected K={top_k}; "
        f"PLS components={n_components}. Parameters were selected by five-fold "
        "CV with training-fold absolute-Spearman EEG ranking. The final ranking "
        "and PLS fit use all 46 participants.\n\n"
        "This SHAP analysis explains the fitted full-cohort model. It is not an "
        "out-of-sample performance estimate, inferential test, or causal effect. "
        "Generalization performance is reported only by the nested outer CV.\n\n"
        "Run `python feature_important.py --analysis shap"
        f"{mode_flag}` from the project directory.\n",
        encoding="utf-8",
    )

    if include_side:
        baseline = SHAP_OUTPUT_DIR / "shap_importance_selected.csv"
        if baseline.exists():
            previous = pd.read_csv(baseline)
            comparison = previous.merge(
                importance,
                on="feature",
                how="outer",
                suffixes=("_EEG_only", "_with_side"),
            )
            comparison.to_csv(
                output_dir / "comparison_with_EEG_only.csv", index=False
            )

    print(json.dumps(metadata, indent=2), flush=True)
    print(importance.to_string(index=False), flush=True)


def plot_performance_and_shap() -> None:
    import shap

    predictions = pd.read_csv(OUTPUT_DIR / "outer_fold_predictions.csv")
    fold_metrics = pd.read_csv(OUTPUT_DIR / "outer_fold_metrics.csv")
    importance = pd.read_csv(SHAP_OUTPUT_DIR / "shap_importance_selected.csv")
    shap_values = pd.read_csv(SHAP_OUTPUT_DIR / "shap_values.csv", index_col="Participant_ID")
    source = pd.read_csv(SHAP_OUTPUT_DIR / "full_cohort_source_data.csv", index_col="Participant_ID")
    selected = shap_values.columns.tolist()
    if set(selected) != set(importance["feature"]):
        raise ValueError("SHAP feature names do not match the importance table.")

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
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })
    fig_b, ax_b = plt.subplots(figsize=(54 / 25.4, 50 / 25.4))
    fig_b.subplots_adjust(left=.34, right=.95, bottom=.22, top=.84)
    fig_c, ax_c = plt.subplots(figsize=(68 / 25.4, 50 / 25.4))
    fig_c.subplots_adjust(left=.22, right=.96, bottom=.22, top=.84)
    fig_d, ax_d = plt.subplots(figsize=(100 / 25.4, 56 / 25.4))
    fig_d.subplots_adjust(left=.38, right=.96, bottom=.22, top=.90)
    fig_e, ax_e = plt.subplots(figsize=(100 / 25.4, 56 / 25.4))
    fig_e.subplots_adjust(left=.16, right=.80, bottom=.22, top=.90)

    observed = predictions["NIHSS_actual"].to_numpy(float)
    predicted = predictions["NIHSS_predicted"].to_numpy(float)
    fit = stats.linregress(observed, predicted)
    x_line = np.linspace(0, 12, 250)
    fitted = fit.intercept + fit.slope * x_line
    residuals = predicted - (fit.intercept + fit.slope * observed)
    residual_se = np.sqrt(np.sum(residuals**2) / (len(observed) - 2))
    centered_ss = np.sum((observed - observed.mean()) ** 2)
    mean_se = residual_se * np.sqrt(
        1 / len(observed) + (x_line - observed.mean()) ** 2 / centered_ss
    )
    critical_t = stats.t.ppf(.975, df=len(observed) - 2)
    ax_b.fill_between(
        x_line, fitted - critical_t * mean_se, fitted + critical_t * mean_se,
        color="royalblue", alpha=.14, linewidth=0,
    )
    ax_b.plot(x_line, fitted, color="royalblue", linewidth=1.5)
    ax_b.scatter(observed, predicted, s=8, color="royalblue", edgecolor="none", zorder=3)
    correlation = pearsonr(observed, predicted)
    ax_b.text(
        .04, .96,
        f"r = {correlation.statistic:.2f}\np = {correlation.pvalue:.4f}",
        transform=ax_b.transAxes, ha="left", va="top", fontsize=8,
        linespacing=1.22,
    )
    ax_b.set(
        xlabel="Observed NIHSS", ylabel="OOF-predicted NIHSS",
        xlim=(0, 12), ylim=(0, 12),
    )
    ticks = np.array([0, 4, 8, 12])
    ax_b.set_xticks(ticks)
    ax_b.set_yticks(ticks, labels=("", "4", "8", "12"))
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.tick_params(direction="in", length=3, width=1.0, pad=2)

    metric_columns = ["R2", "RMSE", "MAE", "Pearson_r"]
    metric_labels = ["R²", "RMSE", "MAE", "r"]
    colors = ["royalblue", "#D98C3F", "#42949E", "#8B6FAE"]
    values = [fold_metrics[column].to_numpy(float) for column in metric_columns]
    positions = np.arange(4)
    box = ax_c.boxplot(
        values, positions=positions, widths=.44, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#272727", "linewidth": 1.15},
        whiskerprops={"color": "#272727", "linewidth": .8},
        capprops={"color": "#272727", "linewidth": .8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color, alpha=.38, linewidth=1.0)
    rng = np.random.default_rng(19)
    for position, metric_values, color in zip(positions, values, colors):
        ax_c.scatter(
            position + rng.uniform(-.10, .10, len(metric_values)), metric_values,
            s=16, color=color, edgecolor="white", linewidth=.45, zorder=3,
        )
        ax_c.scatter(
            position, metric_values.mean(), marker="D", s=22, color=color,
            edgecolor="white", linewidth=.5, zorder=4,
        )
    ax_c.set(
        xticks=positions, xticklabels=metric_labels, ylabel="Metric value",
        ylim=(-.28, 3.38), yticks=[0, 1, 2, 3],
    )
    ax_c.tick_params(axis="x", length=0, pad=3)
    ax_c.tick_params(axis="y", direction="in", length=3, width=1.0, pad=2)

    explanation = shap.Explanation(
        values=shap_values[selected].to_numpy(),
        data=source[selected].to_numpy(),
        feature_names=[shap_feature_label(feature) for feature in selected],
    )
    order = np.argsort(-np.abs(explanation.values).mean(axis=0), kind="stable")
    shap.plots.bar(
        explanation, max_display=len(selected), order=order, ax=ax_d, show=False
    )
    for annotation in ax_d.texts:
        annotation.set_fontsize(8)
    ax_d.set_xlabel("mean(|SHAP value|)", fontsize=10)
    ax_d.tick_params(axis="both", labelsize=9)

    axes_before = set(fig_e.axes)
    np.random.seed(SEED)
    shap.plots.beeswarm(
        explanation, max_display=len(selected), order=order, ax=ax_e,
        show=False, plot_size=None, s=16, alpha=1.0,
        color=shap.plots.colors.red_blue, color_bar_label="Feature value",
        group_remaining_features=False,
    )
    ax_e.set_xlabel("SHAP value", fontsize=10)
    ax_e.set_yticklabels([])
    ax_e.tick_params(axis="both", labelsize=9)
    for colorbar_axis in set(fig_e.axes) - axes_before:
        colorbar_axis.tick_params(labelsize=9)
        colorbar_axis.yaxis.label.set_size(10)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "b": FIGURE_DIR / "panel_b_oof_predictions.png",
        "c": FIGURE_DIR / "panel_c_fold_metrics.png",
        "d": FIGURE_DIR / "panel_d_shap_importance.png",
        "e": FIGURE_DIR / "panel_e_shap_values.png",
    }
    for label, figure in zip("bcde", (fig_b, fig_c, fig_d, fig_e)):
        figure.text(
            .015, .985, label, fontsize=12, fontweight="bold",
            ha="left", va="top",
        )
        figure.savefig(outputs[label], dpi=600, facecolor="white")
        plt.close(figure)
    print(f"Saved panels b-e to {FIGURE_DIR}")


def parse_cli_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=("nested", "shap", "all"),
        default="all",
        help="Run nested CV, full-cohort SHAP, or both (default).",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--include-paralysis-side", action="store_true")
    modes.add_argument("--include-clinical-covariates", action="store_true")
    args = parser.parse_args(argv)
    if args.analysis == "nested" and (
        args.include_paralysis_side or args.include_clinical_covariates
    ):
        parser.error(
            "Clinical-covariate flags apply only to --analysis shap or all."
        )
    return args


def main(argv=None) -> None:
    args = parse_cli_args(argv)
    if args.analysis in ("nested", "all"):
        run_nested_cv_analysis()
    if args.analysis in ("shap", "all"):
        run_full_cohort_shap(
            include_paralysis_side=args.include_paralysis_side,
            include_clinical_covariates=args.include_clinical_covariates,
        )
    if args.analysis == "all":
        plot_performance_and_shap()


if __name__ == "__main__":
    main()
