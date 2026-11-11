"""Figure 4 and Supplementary Figure 7: AWW detection-time distributions for
CTT = 1, 2, 3 consecutive positives at FPR 0.8% and 4.0%, testing every day
(Figure 4) or every 3 days (Supp. Fig. 7).

Reproduces simulate_independent_sliding_window / plotter_comparison from
figure4.ipynb, with p_det = 0.16 x 10% sampling, a fixed random seed, and the
underlying per-trial data saved alongside each figure.

Model (as in the notebook): tests are taken on days 0, f, 2f, ... (f = test
frequency). For CTT = n, a window starting on test day t needs n positive
tests on days t, t+f, ..., t+(n-1)f; each window and each n draws fresh,
independent test outcomes. A test on day d is a true positive with
probability 1 - (1 - p)^N, N ~ Poisson(mu_d) imported detectable cases, and
otherwise a false positive with probability alpha. Detection is the first
successful window; it is a true detection if any test in that window was a
true positive, and is reported retest_delay days after the window's last test.

Because windows are independent, each trial's first success is drawn directly
from the per-window success probabilities (same distribution as the
notebook's day-by-day loop, much faster).

Usage (from global_model/):
    python figure4_sim.py
Outputs in global_model/figure4_outputs/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CSV_PATH = "pgfgleam/all_results/global/daily_imports_sensitivity.csv"
OUT_DIR = "figure4_outputs"
COUNTRY = "Singapore"
R0, GEN_TIME = 2.0, 4.0
P_DET = 0.16 * 0.10          # 16% per-sample detection x 10% random sampling
FPRS = [0.008, 0.04]
TEST_FREQS = [1, 3]          # 1 = Figure 4 (daily), 3 = Supp. Fig. 7 (every 3 days)
FREQ_COMPARISON = [1, 3, 7]  # FPR 0, CTT=1: daily vs every 3 vs every 7 days
RETEST_DELAY = 3
NUM_TRIALS = 250_000
MAX_CONSECUTIVE = 4
SEED = 20261111


def set_figure_format():
    """Same rcParams as data_processing.FigureFormat (copied to avoid importing
    pgfgleam/scipy/seaborn just for fonts)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif",
                       "New Century Schoolbook", "Century Schoolbook L", "Utopia",
                       "Palatino", "Charter", "Bookman"],
        "mathtext.fontset": "cm",
        "font.weight": "normal",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.title_fontsize": 13,
        "figure.titlesize": 18,
    })


def simulate_independent_sliding_window(mu_trajectory, p, alpha, rng, num_trials=NUM_TRIALS,
                                        retest_delay=RETEST_DELAY, test_freq=1,
                                        max_consecutive=MAX_CONSECUTIVE):
    """Returns {n: DataFrame(trial, confirm_day, is_true)} for n = 1..max_consecutive.
    Trials with no detection within the trajectory are omitted, as in the notebook."""
    mu = np.asarray(mu_trajectory, dtype=float)
    p_tp = 1.0 - np.exp(-mu * p)                 # E[1 - (1-p)^N], N ~ Poisson(mu)
    p_fp = (1.0 - p_tp) * alpha                  # no true positive, false positive
    p_pos = p_tp + p_fp
    starts_all = np.arange(0, len(mu), test_freq)
    results = {}
    for n in range(1, max_consecutive + 1):
        starts = starts_all[starts_all + (n - 1) * test_freq < len(mu)]
        days = starts[:, None] + test_freq * np.arange(n)[None, :]     # (windows, n)
        p_window = p_pos[days].prod(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_all_fp = np.where(p_window > 0, (p_fp[days] / p_pos[days]).prod(axis=1), 0.0)
        success = rng.random((num_trials, len(starts))) < p_window[None, :]
        found = success.any(axis=1)
        first = success.argmax(axis=1)[found]
        trial = np.flatnonzero(found)
        is_true = rng.random(len(first)) >= p_all_fp[first]
        results[n] = pd.DataFrame({
            "trial": trial,
            "confirm_day": days[first, -1] + retest_delay,
            "is_true": is_true,
        })
    return results


def plot_split_histogram(ax, detection_days, is_true_positive, bins, color, label, xlim):
    """Plots TP (colored) and FP (white) split bars (as in figure4.ipynb)."""
    days = np.asarray(detection_days)
    truth = np.asarray(is_true_positive, dtype=bool)
    counts, bin_edges = np.histogram(days, bins=bins)
    tp_counts = np.histogram(days[truth], bins=bins)[0]
    bin_width = bin_edges[1] - bin_edges[0]
    total_area = np.sum(counts) * bin_width
    if total_area == 0:
        return Patch(), Patch()
    tp_density = tp_counts / total_area
    fp_density = (counts - tp_counts) / total_area
    for i in range(len(bin_edges) - 1):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        if (tp_density[i] + fp_density[i]) > 0:
            ax.bar(bin_center, tp_density[i], width=bin_width * 0.8, color=color, alpha=0.7, edgecolor=color)
            ax.bar(bin_center, fp_density[i], width=bin_width * 0.8, bottom=tp_density[i],
                   color="white", edgecolor=color, linewidth=1)
    return (Patch(facecolor=color, alpha=0.7, label=f"{label} (TP)"),
            Patch(facecolor="white", edgecolor=color, label=f"{label} (FP)"))


def plot_comparison(res, perfect, fpr, test_freq, out_prefix):
    fig, axes = plt.subplots(3, 1, figsize=(15, 16))
    xlim = {"Ireland": (0, 60), "Singapore": (0, 80)}.get(COUNTRY, (0, 90))
    if test_freq == 1:
        xlim = (0, 70)
    bins = np.arange(0, 100)
    colors = {1: "red", 2: "gray", 3: "blue"}
    labels = {1: "First", 2: "Two", 3: "Three"}
    for ax, stage in zip(axes, [1, 2, 3]):
        ax.tick_params(axis="both", which="major", labelsize=18)
        plot_split_histogram(ax, res[stage].confirm_day, res[stage].is_true, bins,
                             colors[stage], f"{labels[stage]} Consecutive", xlim)
        ax.hist(perfect[1].confirm_day, bins=bins, density=True, histtype="step",
                color="black", linestyle="--", linewidth=2)
        ax.set_xlim(xlim)
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=200)
    plt.close(fig)


def plot_frequency_comparison(by_freq, out_prefix):
    """FPR 0, CTT=1: detection-time distributions for each testing frequency,
    stacked, with a vertical line at the mean detection day."""
    freqs = list(by_freq)
    fig, axes = plt.subplots(len(freqs), 1, figsize=(15, 5.3 * len(freqs)), sharex=True, sharey=True)
    colors = {1: "red", 3: "darkorange", 7: "purple"}
    for ax, f in zip(axes, freqs):
        days = by_freq[f].confirm_day
        ax.tick_params(axis="both", which="major", labelsize=18)
        # One bin per testing interval, centred on the possible detection days
        # (test day + retest delay), so densities are per day and comparable.
        bins = np.arange(RETEST_DELAY - f / 2, 100 + f, f)
        plot_split_histogram(ax, days, by_freq[f].is_true, bins, colors.get(f, "grey"), "", (0, 90))
        m = days.mean()
        ax.axvline(m, color="black", linestyle="--", linewidth=2)
        ax.text(m + 0.8, 0.95, f"mean = {m:.1f} d", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=18,
                bbox=dict(facecolor="white", edgecolor="none", pad=2))
        label = "Daily" if f == 1 else f"Every {f} days"
        ax.text(0.01, 0.95, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=20, fontweight="bold")
        ax.set_xlim(0, 90)
    axes[-1].set_xlabel("Detection day", fontsize=20)
    axes[0].set_ylim(0, 1.2 * max(ax.get_ylim()[1] for ax in axes))  # headroom for labels
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=200)
    plt.close(fig)


def main():
    set_figure_format()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df = df[(df.R0 == R0) & (df.generation_time == GEN_TIME) & (df.outbreak_country == COUNTRY)]
    mu_traj = df.sort_values("time").daily_detectable_imports.fillna(0).values
    rng = np.random.default_rng(SEED)

    summary = []
    for test_freq in TEST_FREQS:
        perfect = simulate_independent_sliding_window(mu_traj, P_DET, 0.0, rng, test_freq=test_freq)
        for fpr in FPRS:
            res = simulate_independent_sliding_window(mu_traj, P_DET, fpr, rng, test_freq=test_freq)
            tag = f"{COUNTRY}_delay{RETEST_DELAY}_freq{test_freq}_{fpr}"
            plot_comparison(res, perfect, fpr, test_freq, os.path.join(OUT_DIR, tag))
            rows = [d.assign(ctt=n) for n, d in res.items()]
            rows.append(perfect[1].assign(ctt=0))   # ctt=0: CTT=1 with FPR 0 (dashed reference)
            pd.concat(rows).to_csv(os.path.join(OUT_DIR, f"{tag}_trials.csv.gz"), index=False)
            print(f"\n--- {COUNTRY} | FPR={fpr} | test_freq={test_freq} | p_det={P_DET:.4f} ---")
            ref = [(0, perfect[1])] if fpr == FPRS[0] else []
            for n, d in ref + list(res.items()):
                rec = {"country": COUNTRY, "fpr": 0.0 if n == 0 else fpr, "test_freq": test_freq,
                       "ctt": 1 if n == 0 else n, "reference_no_fpr": n == 0,
                       "n_detected": len(d), "pct_detected": 100 * len(d) / NUM_TRIALS,
                       "ppv_pct": 100 * d.is_true.mean() if len(d) else np.nan,
                       "mean_confirm_day": d.confirm_day.mean() if len(d) else np.nan,
                       "median_confirm_day": d.confirm_day.median() if len(d) else np.nan}
                summary.append(rec)
                if n:
                    print(f"CTT={n}: N={rec['n_detected']} | PPV={rec['ppv_pct']:.1f}% | "
                          f"mean day={rec['mean_confirm_day']:.1f} | median day={rec['median_confirm_day']:.0f}")
    # FPR 0, CTT=1: compare testing frequencies
    by_freq = {f: simulate_independent_sliding_window(mu_traj, P_DET, 0.0, rng, test_freq=f,
                                                      max_consecutive=1)[1]
               for f in FREQ_COMPARISON}
    tag = f"{COUNTRY}_delay{RETEST_DELAY}_fpr0_ctt1_freq_comparison"
    plot_frequency_comparison(by_freq, os.path.join(OUT_DIR, tag))
    pd.concat([d.assign(test_freq=f) for f, d in by_freq.items()]).to_csv(
        os.path.join(OUT_DIR, f"{tag}_trials.csv.gz"), index=False)
    print(f"\n--- {COUNTRY} | FPR=0 | CTT=1 | testing frequency comparison ---")
    for f, d in by_freq.items():
        summary.append({"country": COUNTRY, "fpr": 0.0, "test_freq": f, "ctt": 1,
                        "reference_no_fpr": True, "n_detected": len(d),
                        "pct_detected": 100 * len(d) / NUM_TRIALS, "ppv_pct": 100 * d.is_true.mean(),
                        "mean_confirm_day": d.confirm_day.mean(),
                        "median_confirm_day": d.confirm_day.median()})
        print(f"every {f} day(s): mean day={d.confirm_day.mean():.1f} | median day={d.confirm_day.median():.0f}")
    summary = pd.DataFrame(summary).drop_duplicates(["fpr", "test_freq", "ctt"])
    summary.to_csv(os.path.join(OUT_DIR, "figure4_summary.csv"), index=False)
    print(f"\nWrote figures, per-trial data and figure4_summary.csv to {OUT_DIR}/")


if __name__ == "__main__":
    main()
