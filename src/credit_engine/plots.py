"""Every chart in the project, drawn with one consistent style.

Style rules: titles state the takeaway, thin lines, hairline grid, one y-axis
per chart, and every line is labelled directly so nothing relies on color alone.
Ordered categories (grades, vintages) use a light->dark blue ramp; unordered
ones use a fixed categorical order; the incumbent benchmark is drawn in gray.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# --- palette (light mode) ---------------------------------------------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BENCHMARK = MUTED
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.labelsize": 10,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 22,
    "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def ordinal_ramp(n: int) -> list[str]:
    """n blue steps from light to dark, for ordered categories (grades, years)."""
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def _titles(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def _label_line_end(ax, x, y, text, color=INK_2, dx=4) -> None:
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=9, color=color)


def _label_line_ends(ax, ends: list[tuple]) -> None:
    """Label line ends [(x, y, text), ...], nudging labels apart so none overlap.

    Call after the axis limits are final: positions are worked out in pixels.
    """
    ax.figure.canvas.draw()
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    min_gap_px = 9 * 1.35 * ax.figure.dpi / 72          # one 9pt text line
    placed = []
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        x_px, y_px = to_px((ax.convert_xunits(x), y))   # dates -> axis numbers
        if placed and y_px - placed[-1] < min_gap_px:
            y_px = placed[-1] + min_gap_px
        placed.append(y_px)
        label_x, label_y = to_data((x_px + 5, y_px))
        ax.text(label_x, label_y, text, va="center", fontsize=9, color=INK_2)


def _pct_axis(ax, axis="y", decimals=0) -> None:
    fmt = mtick.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name: str, cfg: dict) -> None:
    fig.savefig(cfg["paths"]["figures_dir"] / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 2: vintage analysis
# ---------------------------------------------------------------------------

def vintage_default_by_grade(by_grade: pd.DataFrame, cfg: dict, min_n: int = 100) -> None:
    df = by_grade[by_grade["n_loans"] >= min_n]
    grades = sorted(df["grade"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    ends = []
    for grade, color in zip(grades, ordinal_ramp(len(grades))):
        g = df[df["grade"] == grade].sort_values("vintage")
        ax.plot(g["vintage"], g["default_rate"], color=color)
        ends.append((g["vintage"].iloc[-1], g["default_rate"].iloc[-1], f"Grade {grade}"))
    _pct_axis(ax)
    ax.set_ylim(bottom=0)
    _label_line_ends(ax, ends)
    ax.set_xlabel("Issue quarter (vintage)")
    ax.set_ylabel("Lifetime default rate")
    _titles(ax, "Default rates rise with grade and drift up for the 2014–15 vintages",
            f"36-month loans, default rate by issue quarter and LendingClub grade (cells with < {min_n} loans hidden)")
    save(fig, "vintage_default_by_grade.png", cfg)


def cum_default_curves(cum: pd.DataFrame, cfg: dict, max_mob: int = 42) -> None:
    """Quarterly curves are aggregated to annual vintages so the chart stays readable."""
    df = cum.assign(year=cum["vintage"].dt.year).groupby(["year", "mob"], as_index=False)[["cum_defaults", "n_loans"]].sum()
    df["cum_default_rate"] = df["cum_defaults"] / df["n_loans"]
    df = df[df["mob"] <= max_mob]
    years = sorted(df["year"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    ends = []
    for year, color in zip(years, ordinal_ramp(len(years))):
        g = df[df["year"] == year]
        ax.plot(g["mob"], g["cum_default_rate"], color=color)
        ends.append((g["mob"].iloc[-1], g["cum_default_rate"].iloc[-1], f"{year} vintage"))
    _pct_axis(ax)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max_mob + 7)
    _label_line_ends(ax, ends)
    ax.set_xlabel("Months on book (approx.: months from issue to last payment)")
    ax.set_ylabel("Cumulative default rate")
    _titles(ax, "Most defaults happen in the first two years, and later vintages default more",
            "Cumulative share of each annual vintage charged off, by months on book")
    save(fig, "cum_default_curves.png", cfg)


def pricing_vs_risk(pricing: pd.DataFrame, cfg: dict, min_n: int = 500) -> None:
    df = pricing[pricing["n_loans"] >= min_n].sort_values("sub_grade")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["avg_int_rate"] / 100, df["default_rate"], color=SERIES[0], linewidth=1.2, alpha=0.6)
    ax.scatter(df["avg_int_rate"] / 100, df["default_rate"], s=np.clip(df["n_loans"] / 300, 20, 140),
               color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for _, row in df[df["sub_grade"].str.endswith("1")].iterrows():   # label A1, B1, ... G1
        ax.annotate(row["sub_grade"], (row["avg_int_rate"] / 100, row["default_rate"]),
                    xytext=(-6, 8), textcoords="offset points", fontsize=9, color=INK_2, ha="right")
    _pct_axis(ax, "x")
    _pct_axis(ax, "y")
    ax.xaxis.set_major_locator(mtick.MultipleLocator(0.05))
    ax.set_xlabel("Average interest rate charged")
    ax.set_ylabel("Realized lifetime default rate")
    _titles(ax, "Risk-based pricing works: riskier sub-grades pay higher rates",
            f"One dot per LendingClub sub-grade; dot size = number of loans (sub-grades with < {min_n} loans hidden)")
    save(fig, "pricing_vs_risk.png", cfg)


# ---------------------------------------------------------------------------
# Phase 5: model benchmark
# ---------------------------------------------------------------------------

# one fixed color per model, used in every chart (the incumbent is gray)
MODEL_COLORS = {"B0": BENCHMARK, "M1": SERIES[0], "M2": SERIES[1]}
MODEL_LABELS = {"B0": "LC sub-grade (existing)", "M1": "M1 WoE scorecard", "M2": "M2 LightGBM (monotone)"}


def roc_curves(y, scores: dict, aucs: dict, cfg: dict) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for m, score in scores.items():
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], label=f"{MODEL_LABELS[m]}  (AUC {aucs[m]:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(axis="x")
    ax.set_xlabel("False-positive rate (good loans flagged as risky)")
    ax.set_ylabel("True-positive rate (defaults caught)")
    ax.legend(loc="lower right")
    _titles(ax, "ROC curves on the out-of-time test set", "Loans issued 2014–2015, never used for fitting or tuning")
    save(fig, "roc.png", cfg)


def ks_panels(y, scores: dict, ks: dict, cfg: dict) -> None:
    """Cumulative score distribution of defaulters vs non-defaulters; KS = the widest gap."""
    y = np.asarray(y)
    fig, axes = plt.subplots(1, len(scores), figsize=(12, 4.2), sharey=True)
    for ax, (m, score) in zip(axes, scores.items()):
        pct = pd.Series(score).rank(pct=True).to_numpy()          # 0 = safest, 1 = riskiest
        grid = np.linspace(0, 1, 201)
        cdf_bad = np.searchsorted(np.sort(pct[y == 1]), grid, side="right") / (y == 1).sum()
        cdf_good = np.searchsorted(np.sort(pct[y == 0]), grid, side="right") / (y == 0).sum()
        gap = cdf_good - cdf_bad
        i = int(np.argmax(gap))
        ax.plot(grid, cdf_good, color=SERIES[0], label="Non-defaulters")
        ax.plot(grid, cdf_bad, color=SERIES[1], label="Defaulters")
        ax.vlines(grid[i], cdf_bad[i], cdf_good[i], color=INK, linewidth=1.2)
        ax.annotate(f"KS = {ks[m]:.3f}", (grid[i], (cdf_bad[i] + cdf_good[i]) / 2), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))
        ax.set_title(MODEL_LABELS[m], fontsize=10.5, pad=8)
        ax.set_xlabel("Score percentile (safest → riskiest)")
        _pct_axis(ax, "x")
        _pct_axis(ax)
    axes[0].set_ylabel("Cumulative share of group")
    axes[0].legend(loc="upper left")
    fig.suptitle("KS: how far apart the model pushes defaulters and non-defaulters (test set)",
                 x=0.07, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ks.png", cfg)


def calibration_panels(deciles: dict, cfg: dict) -> None:
    """deciles = {split: {model: decile_table}}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    titles = {"valid": "Validation 2013 (calibration was fit here)", "test": "Test 2014–15 (out of time)"}
    top = max(t["actual_rate"].max() for d in deciles.values() for t in d.values()) * 1.08
    for ax, (split, tables) in zip(axes, deciles.items()):
        ax.plot([0, top], [0, top], color=AXIS, linewidth=1)
        for m, t in tables.items():
            ax.plot(t["predicted_pd"], t["actual_rate"], color=MODEL_COLORS[m], marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=MODEL_LABELS[m])
        ax.set_title(titles[split], fontsize=10.5, pad=8)
        ax.set_xlabel("Predicted PD (decile average)")
        ax.grid(axis="x")
        _pct_axis(ax, "x")
        _pct_axis(ax)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
    axes[0].set_ylabel("Actual default rate")
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibration by decile: points on the diagonal mean predicted PD = actual default rate",
                 x=0.07, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "calibration.png", cfg)


# ---------------------------------------------------------------------------
# Phase 6: profit policies
# ---------------------------------------------------------------------------

POLICY_COLORS = {"grade": BENCHMARK, "m1_pd": SERIES[0], "m2_pd": SERIES[1],
                 "m2_profit": SERIES[2], "b0_profit": SERIES[3]}


def profit_curves(policy_table: pd.DataFrame, profit_max: dict, cfg: dict) -> None:
    """Total realized profit and return per dollar lent, by approval rate (test set)."""
    approve_all = policy_table[policy_table["policy_key"] == "all"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for key, color in POLICY_COLORS.items():
        rows = policy_table[policy_table["policy_key"] == key].sort_values("target_rate")
        x = list(rows["target_rate"]) + [1.0]            # at 100% every policy = LC's whole book
        width = 2.6 if key == "m2_profit" else 1.8
        ax1.plot(x, list(rows["total_profit"] / 1e6) + [approve_all["total_profit"] / 1e6],
                 color=color, linewidth=width, label=rows["policy"].iloc[0])
        ax2.plot(x, list(rows["return_on_funded"]) + [approve_all["return_on_funded"]], color=color, linewidth=width)

    pm = profit_max["profit_max_policy"]
    ax1.scatter([pm["approval_rate"]], [pm["total_profit"] / 1e6], marker="D", s=46, color=INK,
                edgecolor=SURFACE, linewidth=1.5, zorder=5)
    ax1.annotate(f"Profit-max rule: approve if E[profit] > 0\n→ approves {pm['approval_rate']:.1%} of the book",
                 (pm["approval_rate"], pm["total_profit"] / 1e6), xytext=(0.985, 0.30), textcoords="axes fraction",
                 ha="right", fontsize=8.5, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkB=4))

    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}M"))
    ax1.set_ylabel("Total realized profit")
    ax1.set_title("Total profit", fontsize=10.5, pad=8)
    ax2.set_title("Profit per dollar lent", fontsize=10.5, pad=8)
    ax2.set_ylabel("Realized profit / funded amount")
    _pct_axis(ax2, decimals=1)
    for ax in (ax1, ax2):
        ax.set_xlabel("Approval rate (share of LendingClub's accepted book)")
        _pct_axis(ax, "x")
    ax1.legend(loc="upper left")
    fig.suptitle("Every model-based policy earns more total profit than the grade policy (test 2014–15)",
                 x=0.06, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "profit_curve.png", cfg)


def swap_set_bars(swaps: pd.DataFrame, cfg: dict) -> None:
    """Loans the model and the grade policy disagree on, at the headline approval rate."""
    swaps = swaps[swaps["group"].str.startswith("swap")]
    comparisons = list(dict.fromkeys(swaps["comparison"]))
    panels = [("bad_rate", "Default rate", "pct"), ("avg_int_rate", "Average interest rate", "rate"),
              ("profit_per_loan", "Realized profit per loan", "usd")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    x, width = np.arange(len(comparisons)), 0.36
    groups = [("swap-in", "Swap-ins: model approves, grade declines", SERIES[2]),
              ("swap-out", "Swap-outs: grade approves, model declines", BENCHMARK)]
    for ax, (col, title, kind) in zip(axes, panels):
        for i, (prefix, label, color) in enumerate(groups):
            vals = [swaps[(swaps["comparison"] == c) & swaps["group"].str.startswith(prefix)][col].iloc[0]
                    for c in comparisons]
            bars = ax.bar(x + (i - 0.5) * (width + 0.02), vals, width, color=color, label=label)
            for bar, v in zip(bars, vals):
                text = f"{v:.1%}" if kind == "pct" else f"{v:.1f}%" if kind == "rate" else f"${v:,.0f}"
                ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
        ax.set_xticks(x, [c.replace(" vs grade policy", "\nvs grade policy") for c in comparisons], fontsize=9)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.tick_params(axis="x", length=0)
        if kind == "pct":
            _pct_axis(ax)
        elif kind == "usd":
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        else:
            ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:.0f}%"))
        ax.margins(y=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 0.93), ncol=2, fontsize=9)
    rate = cfg["profit"]["swap_set_approval_rate"]
    fig.suptitle(f"Swap sets at {rate:.0%} approval: the model trades cheap loans for better-paying ones",
                 x=0.06, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "swap_set.png", cfg)
