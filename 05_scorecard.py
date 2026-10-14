"""
Step 5: scorecard and charts. Reads app_data/ultimate_results.csv and writes app_data/scorecard.csv
plus three charts in charts/ (headline, by accident-year age, company scatter). The calculations live in scoring.py, which the app also uses.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scoring
from scoring import METHODS, ROBUSTNESS, NAMES, SHORT

APP_DIR, CHART_DIR = "app_data", "charts"
CHART = ["CL", "ML", "ML_HIST", "CORR"]                       # BF is in the table, not the charts

# validated categorical slots 1-3, plus a neutral for the baseline (color follows the method)
COLOR = {"CL": "#8a8984", "ML": "#2a78d6", "ML_HIST": "#eb6834", "CORR": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def bar_panel(ax, vals, title, fmt="{:.2f}%"):
    y = np.arange(len(CHART))[::-1]
    ax.barh(y, [vals[m] for m in CHART], color=[COLOR[m] for m in CHART], height=0.62)
    for yi, m in zip(y, CHART):
        ax.text(vals[m] + max(vals.values()) * 0.015, yi, fmt.format(vals[m]), va="center", fontsize=9.5)
    ax.set_yticks(y, [SHORT[m] for m in CHART])
    ax.set_xlim(0, max(vals[m] for m in CHART) * 1.22)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ["left", "bottom"]:
        ax.spines[s].set_visible(False)


def chart_headline(sc, scope):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    for ax, col, title in zip(axes, ["mean_company_miss_pct", "median_company_miss_pct", "dollar_weighted_miss_pct"],
                              ["Average company", "Typical (median) company", "Dollar-weighted, whole book"]):
        bar_panel(ax, sc[col].to_dict(), title)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle("How far each method misses the final paid amount (lower is better)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.015, f"{scope['companies']} companies, {scope['projections']} company-year projections, "
             "scored against what was actually paid. Data: NAIC Schedule P private passenger auto.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(f"{CHART_DIR}/chart_headline.png", dpi=170); plt.close(fig)


def chart_by_age(sc):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    groups = [("old_years_median_miss_pct", "Old years\n(1999-2002)"),
              ("middle_years_median_miss_pct", "Middle years\n(2003-2004)"),
              ("young_years_median_miss_pct", "Young years\n(2005-2007)")]
    w, x = 0.19, np.arange(len(groups))
    for i, m in enumerate(CHART):
        vals = [sc.loc[m, g] for g, _ in groups]
        pos = x + (i - 1.5) * (w + 0.02)
        ax.bar(pos, vals, w, color=COLOR[m], label=SHORT[m])
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.08, f"{v:.2f}" if v < 1 else f"{v:.1f}", ha="center", fontsize=8.3, color=INK2)
    ax.set_xticks(x, [g[1] for g in groups])
    ax.set_ylabel("Median miss, % of final amount")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_ylim(0, 8.2)
    fig.suptitle("Young accident years are where the methods differ", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{CHART_DIR}/chart_by_age.png", dpi=170); plt.close(fig)


def chart_scatter(per):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lo, hi = 0.15, 60
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls=(0, (4, 3)))
    better = per["CORR"] < per["CL"]
    ax.scatter(per.loc[~better, "CL"], per.loc[~better, "CORR"], s=26, color=COLOR["CL"], alpha=0.85,
               edgecolor=SURFACE, linewidth=0.8, label=f"Chain-ladder better ({(~better).sum()})")
    ax.scatter(per.loc[better, "CL"], per.loc[better, "CORR"], s=26, color=COLOR["CORR"], alpha=0.95,
               edgecolor=SURFACE, linewidth=0.8, label=f"Correction model better ({better.sum()})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 30]
    ax.set_xticks(ticks, [f"{t:g}%" for t in ticks]); ax.set_yticks(ticks, [f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Chain-ladder: average miss per company")
    ax.set_ylabel("Chain-ladder + ML correction: average miss")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(0.97, 0.04, "below the line = correction model\nmisses by less", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=INK2)
    fig.suptitle("One dot per company", x=0.02, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{CHART_DIR}/chart_company_scatter.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    os.makedirs(CHART_DIR, exist_ok=True)
    res = pd.read_csv(f"{APP_DIR}/ultimate_results.csv")
    assert len(res) == 950 and res["GRCODE"].nunique() == 95
    info = scoring.scope(res)

    sc, per = scoring.scorecard(res, METHODS + ROBUSTNESS)
    out = sc.round(3).reset_index()
    out.insert(1, "name", out["method"].map(NAMES))
    out.to_csv(f"{APP_DIR}/scorecard.csv", index=False)
    print(f"{info['projections']} projections, {info['companies']} companies "
          f"(accident year 1998 is already fully paid, so its {info['left_out']} rows are not scored)\n")
    print(sc.rename(index=SHORT).round(2).T)

    d, lo, hi = scoring.bootstrap_pair(per, "CORR", "ML_HIST")
    print(f"\nCL + ML correction minus ML + history: {d:+.3f} pts  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          "(an interval around zero means the two ML ideas are a tie)")

    print("\nMean company miss by company size (exploratory, about 32 companies per tier):")
    print(scoring.size_table(res, METHODS).round(2))

    print("\nThree worst chain-ladder companies, mean miss %:")
    print(per.loc[per["CL"].nlargest(3).index, METHODS].round(1))

    by_lag = pd.read_csv(f"{APP_DIR}/method_vs_cl_by_lag.csv")
    print("\nWhat the correction model changes: how far its growth factor sits from chain-ladder's, in %, by step")
    print(by_lag[by_lag["method"] == "CORR"].set_index("from_lag")[["mean_pct", "median_pct", "n"]].round(3))

    chart_headline(sc, info); chart_by_age(sc); chart_scatter(per)
    print(f"\nwrote {APP_DIR}/scorecard.csv and three charts in {CHART_DIR}/")
