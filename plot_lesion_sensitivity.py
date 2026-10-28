"""Run lesion-model sensitivity simulations and plot panels b--e."""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stroke_eeg_mst_sensitivity_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/stroke_eeg_mst_sensitivity_numba")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", "/tmp/stroke_eeg_mst_sensitivity_mne")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/stroke_eeg_mst_sensitivity_xdg")

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "lesion_sensitivity"
OUTPUT_DIR = ROOT / "figures/lesion_sensitivity"

MM_TO_INCH = 1 / 25.4
FIGURE_WIDTH_MM = 204.00984


FIGURE_HEIGHT_MM = 270.0
BLUE = "royalblue"
RANDOM_SEED = 2026

METRICS = (
    ("diameter", "Diameter", "Diameter"),
    ("eccentricity_range", "Eccentricity range", "Eccentricity"),
    ("leaf_number", "Leaf number", "Leaf_number"),
    ("tree_hierarchy", "Tree hierarchy", "Tree_hierarchy"),
)

STRATEGIES = (
    ("random_scattered", "Random node deletion", "c"),
    ("hub_centered_strength", "Targeted hub deletion", "d"),
    ("random_centered_contiguous", "Anatomically constrained deletion", "e"),
)

ATTENUATION_COLORS = {
    0.25: "#59A14F",
    0.50: "#F28E2B",
    0.75: "#E15759",
    1.00: BLUE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--time-s", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=10)
    return parser.parse_args()


def lesion_counts() -> list[int]:
    return list(range(9))


def anatomical_sequence(candidates: np.ndarray, dmat: np.ndarray, seed: int) -> np.ndarray:
    chosen, remaining = [int(seed)], set(map(int, candidates)) - {int(seed)}
    while remaining:
        scored = []
        for node in sorted(remaining):
            distances = np.asarray(dmat[node, chosen], float)
            usable = distances[np.isfinite(distances) & (distances > 0)]
            scored.append((float(usable.mean()) if usable.size else np.inf, node))
        score, node = min(scored)
        if not np.isfinite(score):
            raise ValueError("The HCP distance matrix cannot define lesion growth.")
        chosen.append(node)
        remaining.remove(node)
    return np.asarray(chosen, int)


def build_sequences(cmat: np.ndarray, dmat: np.ndarray, seed: int) -> dict[str, list[np.ndarray]]:
    candidates = np.arange(1, cmat.shape[0], 2, dtype=int)
    legacy = random.Random(seed)
    random_sequences = []
    for _ in range(20):
        sequence = list(map(int, candidates))
        legacy.shuffle(sequence)
        random_sequences.append(np.asarray(sequence, int))
    rng = np.random.default_rng(seed)
    anatomical_seeds = rng.choice(candidates, size=20, replace=False)
    anatomical = [anatomical_sequence(candidates, dmat, node) for node in anatomical_seeds]
    graph = nx.from_numpy_array(cmat)
    strength = dict(graph.degree(weight="weight"))
    hubs = sorted(candidates, key=lambda node: (-strength[int(node)], int(node)))[:5]
    hub_centered = [anatomical_sequence(candidates, dmat, node) for node in hubs]
    return {
        "random_scattered": random_sequences,
        "hub_centered_strength": hub_centered,
        "random_centered_contiguous": anatomical,
    }


def attenuate_nodes(cmat: np.ndarray, nodes: np.ndarray, fraction: float) -> np.ndarray:
    selected = np.zeros(cmat.shape[0], dtype=bool)
    selected[np.asarray(nodes, int)] = True
    incident = selected[:, None] | selected[None, :]
    output = np.array(cmat, dtype=float, copy=True)
    output[incident] *= 1.0 - fraction
    return output


def graph_metrics(matrix: np.ndarray) -> dict[str, float]:
    tree = nx.maximum_spanning_tree(nx.from_numpy_array(matrix), weight="weight")
    eccentricity = np.asarray(list(nx.eccentricity(tree).values()), float)
    leaves = sum(degree == 1 for _, degree in tree.degree())
    maximum = max(nx.betweenness_centrality(tree).values())
    return {
        "diameter": float(nx.diameter(tree)),
        "eccentricity_range": float(np.ptp(eccentricity)),
        "leaf_number": float(leaves),
        "tree_hierarchy": float(leaves / (2 * (len(tree) - 1) * maximum)),
    }


def load_model_backend():
    try:
        import Stroke_simulation_whole_brain_model as original
        from neurolib.utils.loadData import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the model dependencies from requirements.txt; the retained "
            "Stroke_simulation_whole_brain_model.py must be beside this script."
        ) from exc
    original.ds = Dataset("hcp")
    return original


def simulate_fc(original, cmat: np.ndarray, time_s: float) -> np.ndarray:
    model = original.model_para_set(cmat, time_s=time_s)
    duration_ms = int(model.params["duration"])
    time = np.arange(.0001, duration_ms / 1000 + .0001, .0001)
    drive = original.MI_stimulus_function(time)
    stimulus = np.zeros((cmat.shape[0], duration_ms * 10), dtype=float)
    stimulus[1], stimulus[55] = drive, drive
    model.params["exc_ext"] = stimulus
    model.run(chunkwise=False, bold=False)
    matrix = np.asarray(original.plot_fc_band(model), dtype=float)
    if matrix.shape != (29, 29) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid simulated FC matrix: {matrix.shape}")
    return matrix


def run_simulations(time_s: float, seed: int) -> pd.DataFrame:
    os.chdir(ROOT)
    cmat_path = ROOT / "data_load/model_data/curr_cmat.npy"
    if not cmat_path.exists():
        raise FileNotFoundError(f"Missing uploaded fitted matrix: {cmat_path}")
    cmat = np.load(cmat_path)
    if cmat.shape != (80, 80) or not np.allclose(cmat, cmat.T):
        raise ValueError(f"Unexpected fitted matrix: {cmat.shape}")
    original = load_model_backend()
    sequences = build_sequences(cmat, np.asarray(original.ds.Dmat), seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULT_DIR / "lesion_sensitivity_results.csv"
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    completed = {
        (row["strategy"], int(row["sequence_id"]), int(row["removed_nodes"]),
         round(float(row["damage_fraction"]), 2)) for row in rows
    }
    baseline_metrics = None
    total = sum(len(value) for value in sequences.values()) * 4 * len(lesion_counts())
    for strategy, strategy_sequences in sequences.items():
        for sequence_id, sequence in enumerate(strategy_sequences):
            for fraction in (.25, .50, .75, 1.0):
                for count in lesion_counts():
                    key = (strategy, sequence_id, count, round(fraction, 2))
                    if key in completed:
                        continue
                    if count == 0 and baseline_metrics is not None:
                        metrics = baseline_metrics
                    else:
                        altered = cmat if count == 0 else attenuate_nodes(cmat, sequence[:count], fraction)
                        metrics = graph_metrics(simulate_fc(original, altered, time_s))
                        if count == 0:
                            baseline_metrics = metrics
                    row = {
                        "strategy": strategy, "sequence_id": sequence_id,
                        "removed_nodes": count, "lesion_extent_percent": 2.5 * count,
                        "damage_fraction": fraction, "retained_fraction": 1.0 - fraction,
                        "selected_nodes_zero_based": ";".join(map(str, sequence[:count])),
                        **metrics,
                    }
                    rows.append(row)
                    completed.add(key)
                    pd.DataFrame(rows).sort_values(
                        ["strategy", "sequence_id", "damage_fraction", "removed_nodes"]
                    ).to_csv(checkpoint, index=False)
                    print(f"[{len(completed)}/{total}] {strategy}, seq={sequence_id}, "
                          f"attenuation={fraction:.2f}, nodes={count}", flush=True)
    return pd.DataFrame(rows)


def configure_style() -> None:
    for font_file in (ROOT / ".fonts/arial").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_file))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return np.nan
    if np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return np.nan
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


def sequence_correlations(frame: pd.DataFrame, metric: str) -> np.ndarray:
    values = []
    for _, sequence in frame.groupby("sequence_id"):
        sequence = sequence.sort_values("lesion_extent_percent")
        values.append(
            finite_spearman(
                sequence["lesion_extent_percent"].to_numpy(float),
                sequence[metric].to_numpy(float),
            )
        )
    return np.asarray(values, dtype=float)


def annotate_empirical(ax: plt.Axes, result: pd.Series) -> None:
    coefficient = float(result["Partial_Spearman_r"])
    x = 0.95 if coefficient < 0 else 0.05
    alignment = "right" if coefficient < 0 else "left"
    ax.text(
        x,
        0.98,
        f"partial r={coefficient:.2f}\n"
        f"p$_{{FDR}}$={float(result['P_FDR_BH_two_sided_displayed']):.3g}",
        transform=ax.transAxes,
        ha=alignment,
        va="top",
        fontsize=10,
    )


def annotate_simulation(ax: plt.Axes, correlations: np.ndarray) -> None:
    finite = correlations[np.isfinite(correlations)]
    ax.text(
        0.05,
        0.98,
        f"r={finite.mean():.2f} ± {finite.std(ddof=1):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )


def add_panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str) -> None:
    position = ax.get_position()
    fig.text(
        position.x0 - 0.062,
        position.y1 + 0.012,
        letter,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_empirical_row(
    axes: list[plt.Axes], partial_source: pd.DataFrame, partial_index: pd.DataFrame
) -> list[pd.DataFrame]:
    source = []
    for ax, (metric, title, partial_metric) in zip(axes, METRICS):
        data = partial_source.loc[
            partial_source["Domain"].eq("MST")
            & partial_source["MI"].eq("Ahand")
            & partial_source["Frequency"].eq("beta1")
            & partial_source["Metric"].eq(partial_metric)
        ].copy()
        result = partial_index.loc[
            partial_index["Domain"].eq("MST")
            & partial_index["MI"].eq("Ahand")
            & partial_index["Frequency"].eq("beta1")
            & partial_index["Metric"].eq(partial_metric)
        ]
        if len(data) != 46 or data["Participant_ID"].nunique() != 46:
            raise ValueError(f"Expected 46 partial-correlation points for {partial_metric}.")
        if len(result) != 1:
            raise ValueError(f"Expected one partial-correlation row for {partial_metric}.")
        plotted_r = float(
            np.corrcoef(data["Adjusted_NIHSS_rank"], data["Adjusted_feature_rank"])[0, 1]
        )
        if not np.isclose(plotted_r, float(result.iloc[0]["Partial_Spearman_r"]), atol=1e-12):
            raise ValueError(f"Residual plot does not reproduce partial r for {partial_metric}.")
        sns.regplot(
            data=data,
            x="Adjusted_NIHSS_rank",
            y="Adjusted_feature_rank",
            ax=ax,
            ci=95,
            n_boot=5000,
            seed=RANDOM_SEED,
            color=BLUE,
            scatter_kws={"color": BLUE, "s": 8},
            line_kws={"color": BLUE, "linewidth": 1.5},
        )
        annotate_empirical(ax, result.iloc[0])
        ax.set(xlabel="NIHSS rank residuals", ylabel=f"{title}\nrank residuals", title="")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(direction="in", length=3, width=1.0, pad=2)
        source.append(
            data.assign(panel="a", domain="empirical_partial", metric=metric)
        )
    return source


def plot_original_simulation_row(
    axes: list[plt.Axes], simulation: pd.DataFrame
) -> tuple[list[pd.DataFrame], list[dict[str, float]]]:
    source = []
    results = []
    for ax, (metric, title, _) in zip(axes, METRICS):
        sns.lineplot(
            data=simulation,
            x="lesion_extent_percent",
            y=metric,
            errorbar="se",
            ax=ax,
            color=BLUE,
        )
        correlations = sequence_correlations(simulation, metric)
        ax.set(xlabel="Damaged brain regions (%)", ylabel=title, title="")
        ax.set_xlim(-1.0, 21.0)
        ax.set_xticks((0, 10, 20))
        ax.yaxis.set_major_locator(MaxNLocator(integer=metric != "tree_hierarchy", nbins=4))
        ax.tick_params(direction="in", length=3, width=1.0, pad=2)
        source.append(
            simulation[["sequence_id", "lesion_extent_percent", metric]]
            .rename(columns={metric: "metric_value"})
            .assign(panel="b", domain="original_random_deletion", metric=metric)
        )
        finite = correlations[np.isfinite(correlations)]
        results.append(
            {
                "panel": "b",
                "strategy": "random_scattered",
                "attenuation_percent": 100,
                "metric": metric,
                "n_sequences": len(finite),
                "mean_spearman_rho": finite.mean(),
                "sd_spearman_rho": finite.std(ddof=1),
            }
        )
    return source, results


def summarize_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, fraction), group in frame.groupby(["strategy", "damage_fraction"]):
        panel = next(item[2] for item in STRATEGIES if item[0] == strategy)
        for metric, _, _ in METRICS:
            correlations = sequence_correlations(group, metric)
            finite = correlations[np.isfinite(correlations)]
            rows.append(
                {
                    "panel": panel,
                    "strategy": strategy,
                    "attenuation_percent": int(round(100 * fraction)),
                    "metric": metric,
                    "n_sequences": len(finite),
                    "mean_spearman_rho": finite.mean(),
                    "sd_spearman_rho": finite.std(ddof=1),
                    "median_spearman_rho": np.median(finite),
                    "negative_sequence_fraction": np.mean(finite < 0),
                }
            )
    return pd.DataFrame(rows)


def plot_strategy_row(
    axes: list[plt.Axes], sensitivity: pd.DataFrame, strategy: str, panel: str
) -> pd.DataFrame:
    selected = sensitivity.loc[sensitivity["strategy"].eq(strategy)].copy()
    summary_frames = []
    for ax, (metric, title, _) in zip(axes, METRICS):
        for fraction in ATTENUATION_COLORS:
            values = selected.loc[np.isclose(selected["damage_fraction"], fraction)]
            trajectory = (
                values.groupby("lesion_extent_percent", as_index=False)[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            trajectory["se"] = trajectory["std"].fillna(0) / np.sqrt(trajectory["count"])
            trajectory["strategy"] = strategy
            trajectory["damage_fraction"] = fraction
            trajectory["metric"] = metric
            trajectory["panel"] = panel
            summary_frames.append(trajectory)
            x = trajectory["lesion_extent_percent"].to_numpy(float)
            mean = trajectory["mean"].to_numpy(float)
            se = trajectory["se"].to_numpy(float)
            color = ATTENUATION_COLORS[fraction]
            ax.plot(x, mean, color=color, linewidth=1.4)
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, linewidth=0)
        ax.set_title("")
        ax.set_xlim(-1.0, 21.0)
        ax.set_xticks((0, 10, 20))
        ax.set_xlabel("Damaged brain regions (%)")
        ax.set_ylabel(title)
        ax.yaxis.set_major_locator(
            MaxNLocator(integer=metric != "tree_hierarchy", nbins=4)
        )
        ax.tick_params(direction="in", length=3, width=1.0, pad=2)
    return pd.concat(summary_frames, ignore_index=True)


def validate_inputs(
    partial_source: pd.DataFrame, original: pd.DataFrame, sensitivity: pd.DataFrame
) -> None:
    selected_partial = partial_source.loc[
        partial_source["Domain"].eq("MST")
        & partial_source["MI"].eq("Ahand")
        & partial_source["Frequency"].eq("beta1")
    ]
    if len(selected_partial) != 4 * 46:
        raise ValueError(
            f"Expected 184 rows for the four empirical partial plots; found {len(selected_partial)}."
        )
    if original["sequence_id"].nunique() != 20:
        raise ValueError("Expected 20 original random-deletion sequences.")
    expected = {
        "random_scattered": 20,
        "hub_centered_strength": 5,
        "random_centered_contiguous": 20,
    }
    for strategy, n_expected in expected.items():
        n_found = sensitivity.loc[sensitivity["strategy"].eq(strategy), "sequence_id"].nunique()
        if n_found != n_expected:
            raise ValueError(f"Expected {n_expected} sequences for {strategy}; found {n_found}.")
    required_metrics = {item[0] for item in METRICS}
    if not required_metrics.issubset(sensitivity.columns):
        raise ValueError(
            f"Sensitivity data are missing MST metrics: {sorted(required_metrics - set(sensitivity.columns))}"
        )
    original_leaf = original[["sequence_id", "removed_nodes", "leaf_number"]]
    sensitivity_leaf = sensitivity.loc[
        sensitivity["strategy"].eq("random_scattered")
        & np.isclose(sensitivity["damage_fraction"], 1.0),
        ["sequence_id", "removed_nodes", "leaf_number"],
    ]
    matched = original_leaf.merge(
        sensitivity_leaf,
        on=["sequence_id", "removed_nodes"],
        suffixes=("_original", "_sensitivity"),
        validate="one_to_one",
    )
    if len(matched) != len(original_leaf) or not np.allclose(
        matched["leaf_number_original"], matched["leaf_number_sensitivity"]
    ):
        raise ValueError("Original and sensitivity random-deletion results do not match.")


def main_b_to_e() -> None:
    args = parse_args()
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        os.chdir(ROOT)
        cmat = np.load(ROOT / "data_load/model_data/curr_cmat.npy")
        backend = load_model_backend()
        sequences = build_sequences(cmat, np.asarray(backend.ds.Dmat), args.seed)
        assert {key: len(value) for key, value in sequences.items()} == {
            "random_scattered": 20,
            "hub_centered_strength": 5,
            "random_centered_contiguous": 20,
        }
        print("Validation passed: model inputs and all lesion sequences are available.")
        return
    sensitivity = run_simulations(args.time_s, args.seed)
    original = sensitivity.loc[
        sensitivity["strategy"].eq("random_scattered")
        & np.isclose(sensitivity["damage_fraction"], 1.0)
    ].copy()
    if original["sequence_id"].nunique() != 20:
        raise ValueError("Expected 20 original random-deletion sequences.")
    expected = {"random_scattered": 20, "hub_centered_strength": 5,
                "random_centered_contiguous": 20}
    for strategy, count in expected.items():
        found = sensitivity.loc[sensitivity["strategy"].eq(strategy), "sequence_id"].nunique()
        if found != count:
            raise ValueError(f"Expected {count} sequences for {strategy}; found {found}.")

    width_mm, height_mm = 180.0, 180.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    outer = fig.add_gridspec(4, 1, left=.09, right=.98, bottom=.14, top=.98, hspace=.68)
    rows = [outer[i].subgridspec(1, 4, wspace=.40) for i in range(4)]
    axes_rows = [[fig.add_subplot(row[0, i]) for i in range(4)] for row in rows]

    plot_original_simulation_row(axes_rows[0], original)
    trajectory_frames = []
    for axes, (strategy, title, panel) in zip(axes_rows[1:], STRATEGIES):
        trajectory_frames.append(plot_strategy_row(axes, sensitivity, strategy, panel))
        top = axes[0].get_position().y1
        fig.text(.535, top + .012, title, ha="center", va="bottom", fontsize=8)
    trajectories = pd.concat(trajectory_frames, ignore_index=True)

    for column, (metric, _, _) in enumerate(METRICS):
        subset = trajectories.loc[trajectories["metric"].eq(metric)]
        low = float(np.min(subset["mean"] - subset["se"]))
        high = float(np.max(subset["mean"] + subset["se"]))
        padding = .08 * max(high - low, 1e-6)
        for axes in axes_rows:
            axes[column].set_ylim(low - padding, high + padding)
    for axes, letter in zip(axes_rows, "bcde"):
        add_panel_letter(fig, axes[0], letter)
    for axes in axes_rows:
        for axis in axes:
            axis.set_xlabel("")
    fig.text(.535, .085, "Damaged brain regions (%)", ha="center", va="center", fontsize=9)

    handles = [Line2D([], [], color=ATTENUATION_COLORS[f], linewidth=1.5,
                      label=f"{100*f:.0f}%") for f in ATTENUATION_COLORS]
    fig.legend(handles=handles, title="Attenuation", loc="lower center",
               bbox_to_anchor=(.54, .012), ncol=4, fontsize=7, title_fontsize=7)
    output = OUTPUT_DIR / "lesion_sensitivity_panels_b_to_e.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    trajectories.to_csv(OUTPUT_DIR / "trajectory_summary.csv", index=False)
    print(f"Saved lesion-sensitivity figure to {output}")


if __name__ == "__main__":
    main_b_to_e()
