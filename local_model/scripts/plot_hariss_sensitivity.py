"""Plot the HARISS sensitivity analysis written by NBPMscape/hariss_sensitivity.jl.

Figure 1: HARISS detection time vs weekly ED ARI background, one line per swab
          scenario (corrected healthcare rates), one panel per origin country.
          Horizontal reference lines show the main-analysis HARISS value and
          median ICU / AWW detection times.
Figure 2: Median AWW lead time over the earliest of ICU and HARISS for each
          (background, swab scenario), per country.

Detection times are days since the start of the outbreak in the origin country.
Undetected samples are censored at the end of the observation window, so medians
are conservative (they understate how late HARISS is when it often misses).

Usage:
    python local_model/scripts/plot_hariss_sensitivity.py <results.csv> [out_prefix]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SWAB_LABELS = {
    "uniform_0.90": "90% swabbed (main analysis)",
    "uniform_0.25": "25% swabbed",
    "split_0.05_0.90": "5% discharged / 90% admitted",
    "split_0.05_0.60": "5% discharged / 60% admitted",
}
UPDATED_B = {15856: "summer", 27894: "winter"}


def censor(times, window):
    """Replace non-detections (inf) with the end of the observation window."""
    return np.where(np.isfinite(times), times, window)


def summarise(df):
    df = df.copy()
    df["hariss_c"] = censor(df.hariss_time, df.max_obs_time)
    df["icu_c"] = censor(df.icu_time, df.max_obs_time)
    df["aww_c"] = censor(df.aww_time_high, df.max_obs_time)
    df["local_c"] = np.minimum(df.hariss_c, df.icu_c)
    df["lead_high"] = df.local_c - df.aww_c
    df["lead_low"] = df.local_c - censor(df.aww_time_low, df.max_obs_time)
    df["hariss_detected"] = np.isfinite(df.hariss_time)
    keys = ["country", "rate_set", "ed_ari_background", "swab_scenario", "is_main_analysis"]
    return df.groupby(keys).agg(
        hariss_med=("hariss_c", "median"),
        hariss_q25=("hariss_c", lambda x: x.quantile(0.25)),
        hariss_q75=("hariss_c", lambda x: x.quantile(0.75)),
        hariss_pct=("hariss_detected", "mean"),
        icu_med=("icu_c", "median"),
        aww_med=("aww_c", "median"),
        lead_high=("lead_high", "median"),
        lead_low=("lead_low", "median"),
    ).reset_index()


def plot_detection(s, out):
    countries = list(dict.fromkeys(s.country))
    fig, axes = plt.subplots(1, len(countries), figsize=(4.2 * len(countries), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, country in zip(axes, countries):
        c = s[s.country == country]
        corr = c[c.rate_set == "corrected"]
        for scen, label in SWAB_LABELS.items():
            g = corr[corr.swab_scenario == scen].sort_values("ed_ari_background")
            if g.empty:
                continue
            line, = ax.plot(g.ed_ari_background, g.hariss_med, marker="o", label=label)
            ax.fill_between(g.ed_ari_background, g.hariss_q25, g.hariss_q75,
                            color=line.get_color(), alpha=0.12)
        main = c[c.is_main_analysis]
        if not main.empty:
            ax.scatter(main.ed_ari_background, main.hariss_med, marker="*", s=180,
                       color="black", zorder=5, label="Main analysis (original rates)")
        ax.axhline(corr.icu_med.median(), ls="--", color="grey", label="ICU (corrected rates)")
        ax.axhline(corr.aww_med.median(), ls=":", color="green", label="AWW (16% x 100%)")
        for b, name in UPDATED_B.items():
            ax.axvline(b, color="lightgrey", lw=0.8, zorder=0)
            ax.text(b, ax.get_ylim()[1], f" {name}", rotation=90, va="top", fontsize=7, color="grey")
        ax.set_title(country)
        ax.set_xlabel("Weekly ED ARI attendances (background)")
    axes[0].set_ylabel("Median detection time (days)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(f"{out}_detection.pdf")
    fig.savefig(f"{out}_detection.png", dpi=200)


def plot_lead(s, out):
    corr = s[s.rate_set == "corrected"]
    countries = list(dict.fromkeys(corr.country))
    fig, axes = plt.subplots(1, len(countries), figsize=(4.2 * len(countries), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, country in zip(axes, countries):
        c = corr[corr.country == country]
        mat = c.pivot(index="swab_scenario", columns="ed_ari_background", values="lead_high")
        mat = mat.reindex([k for k in SWAB_LABELS if k in mat.index])
        im = ax.imshow(mat.values, cmap="viridis", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.0f}", ha="center", va="center",
                        color="white", fontsize=8)
        ax.set_xticks(range(mat.shape[1]), [f"{b:,}" for b in mat.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(mat.shape[0]), [SWAB_LABELS[k] for k in mat.index], fontsize=7)
        ax.set_title(country)
        ax.set_xlabel("Weekly ED ARI attendances")
        fig.colorbar(im, ax=ax, fraction=0.046, label="AWW lead time (days)")
    fig.suptitle("Median AWW lead time (16% x 100%) over earliest of ICU and HARISS, corrected rates")
    fig.tight_layout()
    fig.savefig(f"{out}_lead_time.pdf")
    fig.savefig(f"{out}_lead_time.png", dpi=200)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0]
    s = summarise(pd.read_csv(path))
    s.to_csv(f"{out}_summary.csv", index=False)
    plot_detection(s, out)
    plot_lead(s, out)
    print(f"Wrote {out}_summary.csv, {out}_detection.(pdf|png), {out}_lead_time.(pdf|png)")


if __name__ == "__main__":
    main()
