"""Plot the HARISS sensitivity analysis written by NBPMscape/hariss_sensitivity.jl.

Axes of the grid (corrected healthcare rates):
  x: share of ED ARI attendances sequenced (weekly samples / weekly ED ARI background)
  y: ratio of swab proportions, admitted : discharged from ED

Figure 1 (_detection): median HARISS detection time (IQR band, corrected healthcare
    rates), one row per origin country. Left: background ED ARI attendances as a
    multiple of the main-analysis value (swab ratio 1:1), with the updated
    summer/winter estimates marked. Right: swab ratio as a multiple of the
    main-analysis 1:1 (main-analysis background). Horizontal lines are median ICU
    and AWW detection. The original-rates main analysis is in the summary CSV
    (rate_set = original) for reporting the rate-correction effect in the text.
Figure 2 (_lead_time): heatmap of median AWW (CTT=1) lead time over the earliest of ICU
    and HARISS across the grid.
Figure 3 (_hariss_heatmap): heatmap of median HARISS detection time across the grid.

Detection times are days since the start of the outbreak in the origin country.
Undetected samples are censored at the end of the observation window, so medians
understate how late a channel is when it often misses.

Usage:
    python local_model/scripts/plot_hariss_sensitivity.py <results.csv> [out_prefix]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UPDATED_ESTIMATES = {"summer": 15856, "winter": 27894}  # weekly ED ARI attendances
RATIO_LABEL = "Swab ratio (admitted : discharged)"
SHARE_LABEL = "ED ARI attendances sequenced (%)"


def censor(times, window):
    """Replace non-detections (inf) with the end of the observation window."""
    return np.where(np.isfinite(times), times, window)


def summarise(df):
    df = df.copy()
    df["share_pct"] = (100 * df.seq_share).round(2)
    df["hariss_c"] = censor(df.hariss_time, df.max_obs_time)
    df["icu_c"] = censor(df.icu_time, df.max_obs_time)
    df["aww_c"] = censor(df.aww_ctt1_time, df.max_obs_time)
    df["local_c"] = np.minimum(df.hariss_c, df.icu_c)
    df["lead_ctt1"] = df.local_c - df.aww_c
    df["lead_ctt2"] = df.local_c - censor(df.aww_ctt2_time, df.max_obs_time)
    df["hariss_detected"] = np.isfinite(df.hariss_time)
    keys = ["country", "rate_set", "ed_ari_background", "share_pct", "swab_ratio", "is_main_analysis"]
    return df.groupby(keys).agg(
        hariss_med=("hariss_c", "median"),
        hariss_q25=("hariss_c", lambda x: x.quantile(0.25)),
        hariss_q75=("hariss_c", lambda x: x.quantile(0.75)),
        hariss_pct_detected=("hariss_detected", "mean"),
        icu_med=("icu_c", "median"),
        aww_med=("aww_c", "median"),
        lead_ctt1=("lead_ctt1", "median"),
        lead_ctt2=("lead_ctt2", "median"),
    ).reset_index()


def ratio_label(r):
    return f"{r:g}:1"


def share_label(v, main_shares):
    return f"{v:.1f}\n(baseline)" if v in main_shares else f"{v:g}"


def _mult_label(m):
    return "Baseline" if np.isclose(m, 1) else f"{m:.1f}\u00d7".replace(".0\u00d7", "\u00d7")


def _detection_panel(ax, g, x, ref, xlabel):
    """One-factor line: median HARISS detection (IQR band) vs multiplier x."""
    g = g.sort_values(x)
    line, = ax.plot(g[x], g.hariss_med, marker="o", color="tab:red", label="HARISS")
    ax.fill_between(g[x], g.hariss_q25, g.hariss_q75, color=line.get_color(), alpha=0.15)
    ax.axhline(ref.icu_med.median(), ls="--", color="grey", label="ICU")
    ax.axhline(ref.aww_med.median(), ls=":", color="green", label="AWW (CTT=1)")
    ax.set_xscale("log")
    ticks = sorted(set(g[x]))
    ax.set_xticks(ticks, [_mult_label(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)


def plot_detection(s, out, n_samples_per_week):
    """Rows = countries; left = background ARI attendances (swab ratio 1:1),
    right = swab ratio (main-analysis background). Both relative to baseline."""
    countries = list(dict.fromkeys(s.country))
    fig, axes = plt.subplots(len(countries), 2, figsize=(9, 3.2 * len(countries)),
                             squeeze=False, sharey="row")
    for row, country in zip(axes, countries):
        c = s[s.country == country]
        corr = c[c.rate_set == "corrected"].copy()
        main = c[c.is_main_analysis]
        base_b = main.ed_ari_background.iloc[0] if not main.empty else corr.ed_ari_background.min()
        corr["b_mult"] = corr.ed_ari_background / base_b
        _detection_panel(row[0], corr[corr.swab_ratio == 1], "b_mult", corr,
                         "Background ED ARI attendances (relative to baseline)")
        for name, b in UPDATED_ESTIMATES.items():
            xm = b / base_b
            row[0].axvline(xm, color="lightgrey", lw=0.8, zorder=0)
            row[0].annotate(f"{name} estimate", (xm, 0), xycoords=("data", "axes fraction"),
                            xytext=(-3, 4), textcoords="offset points", rotation=90,
                            ha="right", va="bottom", fontsize=7, color="grey")
        _detection_panel(row[1], corr[corr.ed_ari_background == base_b], "swab_ratio", corr,
                         "Swab ratio, admitted : discharged (relative to baseline 1:1)")
        row[0].set_ylabel(f"{country}\nMedian detection time (days)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06 if len(countries) > 1 else 0.12, 1, 1))
    fig.savefig(f"{out}_detection.pdf")
    fig.savefig(f"{out}_detection.png", dpi=200)


def plot_heatmap(s, value, cbar_label, title, out_suffix, out, cmap):
    corr = s[s.rate_set == "corrected"]
    main_shares = set(s[s.is_main_analysis].share_pct)
    countries = list(dict.fromkeys(corr.country))
    fig, axes = plt.subplots(1, len(countries), figsize=(4.2 * len(countries), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, country in zip(axes, countries):
        mat = corr[corr.country == country].pivot(index="swab_ratio", columns="share_pct", values=value)
        mat = mat.sort_index().sort_index(axis=1, ascending=False)  # most to least sequenced
        im = ax.imshow(mat.values, cmap=cmap, aspect="auto", origin="lower")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                r, g, b, _ = im.cmap(im.norm(v))
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white")
        ax.set_xticks(range(mat.shape[1]), [share_label(v, main_shares) for v in mat.columns], fontsize=8)
        ax.set_yticks(range(mat.shape[0]), [ratio_label(r) for r in mat.index], fontsize=8)
        ax.set_title(country)
        ax.set_xlabel(SHARE_LABEL)
        fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
    axes[0].set_ylabel(RATIO_LABEL)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{out}_{out_suffix}.pdf")
    fig.savefig(f"{out}_{out_suffix}.png", dpi=200)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit(".", 1)[0]
    raw = pd.read_csv(path)
    n_samples_per_week = float((raw.seq_share * raw.ed_ari_background).median())
    s = summarise(raw)
    s.to_csv(f"{out}_summary.csv", index=False)
    plot_detection(s, out, n_samples_per_week)
    plot_heatmap(s, "lead_ctt1", "AWW lead time (days)",
                 "Median AWW (CTT=1) lead time over earliest of ICU and HARISS",
                 "lead_time", out, "viridis")
    plot_heatmap(s, "hariss_med", "HARISS detection time (days)",
                 "Median HARISS detection time", "hariss_heatmap", out, "magma_r")
    print(f"Wrote {out}_summary.csv and {out}_(detection|lead_time|hariss_heatmap).(pdf|png)")


if __name__ == "__main__":
    main()
