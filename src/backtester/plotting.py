"""Charts. matplotlib is imported lazily so the package works without it."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

# Validated categorical palette, light surface.
SERIES_1 = "#2a78d6"     # blue   - strategy
SERIES_2 = "#eb6834"     # orange - benchmark
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"


def plot_equity_vs_benchmark(
    equity: pd.Series,
    benchmark: pd.Series,
    out_path: str = "results/equity_curve.png",
    title: str = "Buy and hold vs SPY, zero costs",
    equity_label: str = "Strategy",
    benchmark_label: str = "SPY",
) -> str:
    """Both series indexed to 100 so they share ONE axis.

    Dollars of equity and dollars of share price are different scales. A
    second y-axis would let any two lines be made to agree, which is why
    this function does not offer one.
    """
    import matplotlib
    matplotlib.use("Agg")          # no GUI, no notebook kernel needed
    import matplotlib.pyplot as plt

    eq = equity / equity.iloc[0] * 100.0
    bm = benchmark / benchmark.iloc[0] * 100.0

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(bm.index, bm.values, color=SERIES_2, linewidth=2.0,
            label=benchmark_label, zorder=2)
    ax.plot(eq.index, eq.values, color=SERIES_1, linewidth=2.0,
            label=equity_label, zorder=3)

    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.set_ylabel("Growth of 100", color=INK_MUTED, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, loc="upper left", fontsize=10,
              labelcolor=INK_MUTED)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path