"""Recompute functional-connectivity bootstrap and time-window sensitivity analyses."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stroke_fc_sensitivity_mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_load"
OUT = ROOT / "figures" / "fc_sensitivity_bootstrap_windows"
RESULTS = ROOT / "results" / "fc_sensitivity_bootstrap_windows"
EXCLUDED = {"sub-05", "sub-06", "sub-15", "sub-49"}
FS, NW, N_TAPERS = 500.0, 3.0, 5
N_BOOTSTRAP, SEED = 1000, 2026
BAND = (14.0, 20.0)
WINDOWS = {
    "primary_1_3s": (1.0, 3.0),
    "alternative_0_3s": (0.0, 3.0),
    "alternative_1_4s": (1.0, 4.0),
    "alternative_0_4s": (0.0, 4.0),
}
WINDOW_LABELS = dict(zip(WINDOWS, ("1–3 s", "0–3 s", "1–4 s", "0–4 s")))
WINDOW_COLORS = dict(zip(WINDOWS, ("#4C78A8", "#4C9A8A", "#D08A45", "#8A6FB0")))
METRICS = ("Diameter", "Eccentricity", "Leaf number", "Tree hierarchy")
TEXT, BOOT_INTERVAL, BOOT_MEDIAN, OBSERVED = "#20262B", "#8EB4C6", "#245F7C", "#BC3E4A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser.parse_args()


def load_clean_trials(participant_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA_DIR / "data_clean" / f"{participant_id}_task-motor-imagery_eeg.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing uploaded EEG file: {path}")
    eeg = loadmat(path, squeeze_me=True, struct_as_record=False)["EEG_save"].eeg
    raw = np.asarray(eeg.rawdata, dtype=float)
    labels = np.asarray(eeg.label).astype(int).ravel()
    trigger_mask = raw[:, -1, :] == 2
    if not np.all(trigger_mask.sum(axis=1) >= 1):
        raise ValueError(f"Missing MI-onset trigger in {path}")
    return raw[:, :29, :], labels, np.argmax(trigger_mask, axis=1)


def extract_window(raw: np.ndarray, onset: np.ndarray, start_s: float, stop_s: float) -> np.ndarray:
    n_samples = int(round((stop_s - start_s) * FS))
    offset = int(round(start_s * FS))
    output = np.empty((len(raw), raw.shape[1], n_samples), dtype=float)
    for trial, trial_onset in enumerate(onset):
        first, last = int(trial_onset + offset), int(trial_onset + offset + n_samples)
        if first < 0 or last > raw.shape[-1]:
            raise ValueError(f"Window {start_s}--{stop_s} s exceeds trial bounds")
        output[trial] = raw[trial, :, first:last]
    return output


def trial_cross_spectra(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        from spectral_connectivity import Multitaper
    except ImportError as exc:
        raise RuntimeError("Install spectral-connectivity from requirements.txt.") from exc
    duration = trials.shape[-1] / FS
    estimator = Multitaper(
        np.transpose(trials, (2, 0, 1)), sampling_frequency=FS,
        time_halfbandwidth_product=NW, detrend_type="constant",
        time_window_duration=duration, time_window_step=duration,
        n_tapers=N_TAPERS, n_fft_samples=trials.shape[-1], is_low_bias=True,
    )
    fourier = np.asarray(estimator.fft())[0]
    frequencies = np.asarray(estimator.frequencies)
    keep = (frequencies >= 0.0) & (frequencies <= BAND[1])
    fourier = fourier[:, :, keep, :]
    spectra = fourier[..., :, None] * fourier[..., None, :].conjugate()
    return frequencies[keep], spectra.mean(axis=1), fourier.shape[1]


def fc_from_mean(mean_csm: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    use = (frequencies >= BAND[0]) & (frequencies <= BAND[1])
    csm = mean_csm[..., use, :, :]
    power = np.real(np.diagonal(csm, axis1=-2, axis2=-1))
    denominator = np.sqrt(power[..., :, None] * power[..., None, :])
    denominator = np.maximum(denominator, np.finfo(float).eps)
    matrices = np.clip(np.abs(np.imag(csm)) / denominator, 0.0, 1.0).mean(axis=-3)
    diagonal = np.arange(matrices.shape[-1])
    matrices[..., diagonal, diagonal] = 0.0
    return matrices


def fc_from_trials(trial_csm: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    return fc_from_mean(trial_csm.mean(axis=0), frequencies)


def edge_vector(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[np.triu_indices(matrix.shape[-1], 1)]


def mst_metrics(matrix: np.ndarray) -> np.ndarray:
    tree = nx.maximum_spanning_tree(nx.from_numpy_array(matrix), weight="weight")
    eccentricity = nx.eccentricity(tree)
    leaves = sum(degree == 1 for _, degree in tree.degree())
    max_between = max(nx.betweenness_centrality(tree).values())
    return np.asarray([
        nx.diameter(tree), max(eccentricity.values()) - min(eccentricity.values()),
        leaves, leaves / (2 * (len(tree) - 1) * max_between),
    ], dtype=float)


def consume_legacy_alpha_draws(rng: np.random.Generator, n_trials: int, n: int) -> None:
    for start in range(0, n, 20):
        size = min(20, n - start)
        rng.integers(0, n_trials, size=(size, n_trials))
    for _ in range(100):
        rng.permutation(n_trials)


def bootstrap_primary(trial_csm, frequencies, rng, n_bootstrap):
    full_fc = fc_from_trials(trial_csm, frequencies)
    full_edges, observed = edge_vector(full_fc), mst_metrics(full_fc)
    correlations = np.empty(n_bootstrap)
    samples = np.empty((n_bootstrap, len(METRICS)))
    n_trials = len(trial_csm)
    for start in range(0, n_bootstrap, 20):
        stop = min(start + 20, n_bootstrap)
        sampled = rng.integers(0, n_trials, size=(stop - start, n_trials))
        weights = np.stack([np.bincount(row, minlength=n_trials) / n_trials for row in sampled])
        matrices = fc_from_mean(np.tensordot(weights, trial_csm, axes=(1, 0)), frequencies)
        for offset, matrix in enumerate(matrices):
            index = start + offset
            correlations[index] = stats.pearsonr(full_edges, edge_vector(matrix)).statistic
            samples[index] = mst_metrics(matrix)
    fc_summary = {
        "FC_edge_r_median": np.median(correlations),
        "FC_edge_r_CI_low": np.quantile(correlations, .025),
        "FC_edge_r_CI_high": np.quantile(correlations, .975),
    }
    metric_rows = [{
        "Metric": metric, "Observed": observed[index],
        "Bootstrap_median": np.median(samples[:, index]),
        "CI_low": np.quantile(samples[:, index], .025),
        "CI_high": np.quantile(samples[:, index], .975),
    } for index, metric in enumerate(METRICS)]
    return fc_summary, metric_rows


def analyze_subject(position: int, subject: dict, n_bootstrap: int) -> tuple:
    participant = subject["Participant_ID"]
    raw, labels, onset = load_clean_trials(participant)
    affected_label = 1 if subject["ParalysisSide"] == "left" else 2
    selected = labels == affected_label
    raw, onset = raw[selected], onset[selected]
    window_fc, window_metric_rows, audit_rows, primary = {}, [], [], None
    for name, (start, stop) in WINDOWS.items():
        trials = extract_window(raw, onset, start, stop)
        frequencies, spectra, n_tapers = trial_cross_spectra(trials)
        matrix = fc_from_trials(spectra, frequencies)
        window_fc[name] = matrix
        for metric, value in zip(METRICS, mst_metrics(matrix)):
            window_metric_rows.append({
                "Participant_ID": participant, "MI": "Ahand", "Frequency": "beta1",
                "Window": name, "Metric": metric, "Value": float(value),
            })
        audit_rows.append({
            "Participant_ID": participant, "Window": name, "Start_s": start,
            "Stop_s": stop, "N_trials": int(selected.sum()), "N_samples": trials.shape[-1],
            "N_tapers": int(n_tapers), "N_FFT": trials.shape[-1],
            "Frequency_bin_spacing_Hz": FS / trials.shape[-1],
            "Full_smoothing_bandwidth_Hz": 2 * NW / (stop - start),
        })
        if name == "primary_1_3s":
            primary = (frequencies, spectra, n_tapers)
    primary_edges = edge_vector(window_fc["primary_1_3s"])
    window_rows = [{
        "Participant_ID": participant, "MI": "Ahand", "Frequency": "beta1",
        "Alternative_window": name, "Comparison": f"primary_1_3s_vs_{name}",
        "FC_edge_r": stats.pearsonr(primary_edges, edge_vector(window_fc[name])).statistic,
    } for name in list(WINDOWS)[1:]]
    frequencies, spectra, n_tapers = primary
    rng = np.random.default_rng(SEED + 100 * position)
    consume_legacy_alpha_draws(rng, len(spectra), n_bootstrap)
    fc_summary, metric_rows = bootstrap_primary(spectra, frequencies, rng, n_bootstrap)
    bootstrap_row = {
        "Participant_ID": participant, "MI": "Ahand", "Frequency": "beta1",
        "N_trials": len(spectra), "N_tapers": n_tapers,
        "N_bootstrap": n_bootstrap, **fc_summary,
    }
    for row in metric_rows:
        row.update({"Participant_ID": participant, "MI": "Ahand", "Frequency": "beta1"})
    return position, bootstrap_row, metric_rows, window_rows, window_metric_rows, audit_rows


def recompute(n_bootstrap: int, workers: int) -> tuple[pd.DataFrame, ...]:
    subjects = pd.read_csv(DATA_DIR / "subject.csv")
    subjects = subjects[~subjects.Participant_ID.isin(EXCLUDED)].reset_index(drop=True)
    records, collected = subjects.to_dict("records"), []
    if workers == 1:
        for position, record in enumerate(records):
            collected.append(analyze_subject(position, record, n_bootstrap))
            print(f"{record['Participant_ID']}: FC/MST robustness complete", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_subject, i, record, n_bootstrap): record["Participant_ID"]
                       for i, record in enumerate(records)}
            for future in as_completed(futures):
                collected.append(future.result())
                print(f"{futures[future]}: FC/MST robustness complete", flush=True)
    collected.sort(key=lambda item: item[0])
    bootstrap = pd.DataFrame([item[1] for item in collected])
    bootstrap_mst = pd.DataFrame([row for item in collected for row in item[2]])
    window_fc = pd.DataFrame([row for item in collected for row in item[3]])
    window_mst = pd.DataFrame([row for item in collected for row in item[4]])
    audit = pd.DataFrame([row for item in collected for row in item[5]])
    if (len(bootstrap), len(bootstrap_mst), len(window_fc), len(window_mst)) != (46, 184, 138, 736):
        raise RuntimeError("Unexpected output dimensions; expected 46 retained participants")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "trial_bootstrap_fc.csv": bootstrap, "trial_bootstrap_mst.csv": bootstrap_mst,
        "window_fc_comparisons.csv": window_fc, "window_mst_values.csv": window_mst,
        "spectral_parameter_audit.csv": audit,
    }.items():
        table.to_csv(RESULTS / name, index=False)
    return bootstrap, window_fc, bootstrap_mst, window_mst


def configure_style() -> None:
    font_dir = ROOT / ".fonts" / "arial"
    for path in font_dir.glob("*.ttf") if font_dir.exists() else ():
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "axes.titleweight": "bold", "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 1, "xtick.direction": "in", "ytick.direction": "in",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def density_jitter(values, rng, width=.16):
    values = np.asarray(values, float)
    bandwidth = max(float(np.std(values)) * .25, .018)
    density = np.exp(-.5 * ((values[:, None] - values[None, :]) / bandwidth) ** 2).sum(1)
    return rng.uniform(-1, 1, len(values)) * width * (.22 + .78 * density / density.max())


def draw_summary(axis, x, values, color, rng):
    values = np.asarray(values, float)
    axis.scatter(x + density_jitter(values, rng), values, s=11, color=color,
                 alpha=.82, edgecolor="white", linewidth=.22, zorder=2)
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    axis.vlines(x, q1, q3, color=TEXT, linewidth=.9, zorder=4)
    axis.hlines([q1, q3], x-.065, x+.065, color=TEXT, linewidth=.8, zorder=4)
    axis.hlines(median, x-.22, x+.22, color=TEXT, linewidth=1.25, zorder=5)


def plot_fc(bootstrap, window_fc):
    mm = 1 / 25.4
    fig = plt.figure(figsize=(204.00984 * mm, 72 * mm))
    grid = fig.add_gridspec(1, 2, left=.205, right=.795, bottom=.22, top=.97,
                           width_ratios=(1, 2.25), wspace=.28)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    rng = np.random.default_rng(20261012)
    draw_summary(axes[0], 0, bootstrap.FC_edge_r_median, WINDOW_COLORS["primary_1_3s"], rng)
    axes[0].set(xticks=[0], xticklabels=["Trial bootstrap"], xlim=(-.55, .55),
                ylabel="FC matrix correlation (Pearson r)")
    for x, name in enumerate(list(WINDOWS)[1:]):
        draw_summary(axes[1], x, window_fc.loc[window_fc.Alternative_window.eq(name), "FC_edge_r"],
                     WINDOW_COLORS[name], rng)
    axes[1].set(xticks=range(3), xticklabels=[WINDOW_LABELS[n] for n in list(WINDOWS)[1:]],
                xlim=(-.52, 2.52), xlabel="Alternative MI window")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    for axis in axes:
        axis.set_ylim(.4, 1.01)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fc_reliability_ahand_low_beta.png", dpi=600, facecolor="white")
    plt.close(fig)


def bootstrap_panel(fig, spec, data):
    grid = spec.subgridspec(1, 4, wspace=.20)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    participants = sorted(data.Participant_ID.unique(), key=lambda value: int(value[-2:]))
    ymap = {participant: len(participants)-1-i for i, participant in enumerate(participants)}
    for axis, metric in zip(axes, METRICS):
        selected = data[data.Metric.eq(metric)].copy()
        selected["y"] = selected.Participant_ID.map(ymap)
        for y in range(len(participants)):
            if y % 2 == 0:
                axis.axhspan(y-.5, y+.5, color="#F5F7F8", zorder=0)
        axis.hlines(selected.y, selected.CI_low, selected.CI_high, color=BOOT_INTERVAL, lw=1.05)
        axis.scatter(selected.Bootstrap_median, selected.y, s=7, color=BOOT_MEDIAN,
                     edgecolor="white", linewidth=.18, zorder=3)
        axis.scatter(selected.Observed, selected.y, s=10, marker="D", color=OBSERVED,
                     edgecolor="white", linewidth=.18, zorder=4)
        axis.set_title(metric)
        axis.set_ylim(-.8, len(participants)-.2)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=metric != "Tree hierarchy"))
    axes[0].set_yticks([ymap[p] for p in participants], participants)
    axes[0].tick_params(axis="y", length=0, pad=1.5, labelsize=5.4)
    axes[0].set_ylabel("Participant")
    for axis in axes[1:]:
        axis.tick_params(axis="y", left=False, labelleft=False)
    return axes


def window_panel(fig, spec, data):
    grid = spec.subgridspec(1, 4, wspace=.36)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    rng = np.random.default_rng(20261011)
    for axis, metric in zip(axes, METRICS):
        selected = data[data.Metric.eq(metric)]
        for x, name in enumerate(WINDOWS):
            values = selected.loc[selected.Window.eq(name), "Value"].to_numpy(float)
            axis.scatter(x + density_jitter(values, rng), values, s=8,
                         color=WINDOW_COLORS[name], alpha=.76, edgecolor="white", linewidth=.18)
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            axis.vlines(x, q1, q3, color=TEXT, lw=.85)
            axis.hlines([q1, q3], x-.06, x+.06, color=TEXT, lw=.75)
            axis.hlines(median, x-.22, x+.22, color=TEXT, lw=1.15)
        axis.set_ylabel(metric)
        axis.set_xticks(range(4), [WINDOW_LABELS[name].replace(" s", "") for name in WINDOWS])
        axis.set_xlim(-.58, 3.58)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=metric != "Tree hierarchy"))
    return axes


def plot_mst(bootstrap_mst, window_mst):
    mm = 1 / 25.4
    fig = plt.figure(figsize=(204.00984 * mm, 210 * mm))
    outer = fig.add_gridspec(2, 1, left=.105, right=.992, bottom=.105, top=.955,
                            height_ratios=(3.08, 1.18), hspace=.32)
    top, bottom = bootstrap_panel(fig, outer[0], bootstrap_mst), window_panel(fig, outer[1], window_mst)
    fig.canvas.draw()
    for axes, letter in ((top, "a"), (bottom, "b")):
        fig.text(.018, max(axis.get_position().y1 for axis in axes) + 3.5/210,
                 letter, fontsize=12, fontweight="bold", va="bottom")
    handles = [
        Line2D([0], [0], color=BOOT_INTERVAL, lw=1.25, label="Individual 95% bootstrap interval"),
        Line2D([0], [0], marker="o", ls="none", color=BOOT_MEDIAN, markersize=4.2,
               label="Bootstrap median"),
        Line2D([0], [0], marker="D", ls="none", color=OBSERVED, markersize=4.2,
               label="Observed full-trial value"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(.55, min(axis.get_position().y0 for axis in top)-.014),
               ncol=3, frameon=False)
    fig.text(.55, min(axis.get_position().y0 for axis in bottom)-.050,
             "MI analysis window (s)", ha="center", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mst_reliability_ahand_low_beta.png", dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    configure_style()
    bootstrap, window_fc, bootstrap_mst, window_mst = recompute(
        args.n_bootstrap, max(1, args.workers)
    )
    plot_fc(bootstrap, window_fc)
    plot_mst(bootstrap_mst, window_mst)
    print(f"Saved PNG figures to {OUT}")


if __name__ == "__main__":
    main()
