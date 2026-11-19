# Module 5: Performance Evaluation
# Compares original vs DSatur-optimized vs Backtracking-optimized circuits.

from __future__ import annotations

import math
import textwrap
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from qiskit import QuantumCircuit

matplotlib.use("Agg")  # Non-interactive backend; caller may switch before importing.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _circuit_depth(circuit: QuantumCircuit) -> int:
    """Return the depth of a Qiskit circuit (Qiskit 2.x compatible)."""
    return circuit.depth()


def _gate_count(circuit: QuantumCircuit) -> int:
    """Return the total number of gate operations in a circuit (excluding barriers)."""
    return circuit.size()


def _depth_reduction_pct(original_depth: int, optimized_depth: int) -> float:
    """Percentage depth reduction relative to the original circuit depth."""
    if original_depth == 0:
        return 0.0
    return round((original_depth - optimized_depth) / original_depth * 100.0, 2)


# ---------------------------------------------------------------------------
# 1. evaluate_all
# ---------------------------------------------------------------------------

def evaluate_all(
    original_circuit: QuantumCircuit,
    dsatur_circuit: QuantumCircuit,
    backtracking_circuit: QuantumCircuit,
    dsatur_coloring_result: dict,
    backtracking_coloring_result: dict,
    dsatur_equivalent: bool,
    backtracking_equivalent: bool,
) -> dict:
    """Collect all performance metrics into a structured dictionary.

    Parameters
    ----------
    original_circuit:
        The unoptimized reference circuit.
    dsatur_circuit:
        Circuit produced by the DSatur coloring optimizer.
    backtracking_circuit:
        Circuit produced by the Backtracking coloring optimizer.
    dsatur_coloring_result:
        Dict returned by the DSatur module, expected keys:
        ``n_colors`` (int) and ``coloring_time`` (float, seconds).
    backtracking_coloring_result:
        Same structure as *dsatur_coloring_result* but for Backtracking.
    dsatur_equivalent:
        Boolean — whether DSatur-optimized circuit is equivalent to original.
    backtracking_equivalent:
        Boolean — whether Backtracking-optimized circuit is equivalent to original.

    Returns
    -------
    dict
        Nested dict with keys ``original``, ``dsatur``, ``backtracking``,
        and ``summary``.
    """
    orig_depth = _circuit_depth(original_circuit)
    dsat_depth = _circuit_depth(dsatur_circuit)
    back_depth = _circuit_depth(backtracking_circuit)

    result: dict[str, Any] = {
        "original": {
            "depth": orig_depth,
            "gate_count": _gate_count(original_circuit),
        },
        "dsatur": {
            "depth": dsat_depth,
            "gate_count": _gate_count(dsatur_circuit),
            "n_colors": dsatur_coloring_result.get("n_colors", None),
            "coloring_time": dsatur_coloring_result.get("coloring_time", None),
            "equivalent": dsatur_equivalent,
            "depth_reduction_pct": _depth_reduction_pct(orig_depth, dsat_depth),
        },
        "backtracking": {
            "depth": back_depth,
            "gate_count": _gate_count(backtracking_circuit),
            "n_colors": backtracking_coloring_result.get("n_colors", None),
            "coloring_time": backtracking_coloring_result.get("coloring_time", None),
            "equivalent": backtracking_equivalent,
            "depth_reduction_pct": _depth_reduction_pct(orig_depth, back_depth),
        },
        "summary": {
            "original_depth": orig_depth,
            "dsatur_depth": dsat_depth,
            "backtracking_depth": back_depth,
            "dsatur_depth_reduction_pct": _depth_reduction_pct(orig_depth, dsat_depth),
            "backtracking_depth_reduction_pct": _depth_reduction_pct(orig_depth, back_depth),
            "dsatur_equivalent": dsatur_equivalent,
            "backtracking_equivalent": backtracking_equivalent,
        },
    }
    return result


# ---------------------------------------------------------------------------
# 2. format_comparison_table
# ---------------------------------------------------------------------------

def format_comparison_table(evaluation_result: dict) -> str:
    """Return a human-readable comparison table as a string.

    Parameters
    ----------
    evaluation_result:
        The dict returned by :func:`evaluate_all`.

    Returns
    -------
    str
        A formatted table suitable for ``print()``.
    """
    orig = evaluation_result["original"]
    dsat = evaluation_result["dsatur"]
    back = evaluation_result["backtracking"]

    def _fmt_time(val: Any) -> str:
        if val is None:
            return "N/A"
        return f"{val:.4f} s"

    def _fmt_pct(val: Any) -> str:
        if val is None:
            return "N/A"
        return f"{val:.2f}%"

    def _fmt_bool(val: Any) -> str:
        if val is None:
            return "N/A"
        return "Yes" if val else "No"

    col_w = 20
    sep = "+" + "-" * (col_w + 2) + "+" + "-" * (col_w + 2) + "+" + "-" * (col_w + 2) + "+" + "-" * (col_w + 2) + "+"

    def _row(label: str, orig_val: str, dsat_val: str, back_val: str) -> str:
        return (
            f"| {label:<{col_w}} "
            f"| {orig_val:<{col_w}} "
            f"| {dsat_val:<{col_w}} "
            f"| {back_val:<{col_w}} |"
        )

    header = _row("Metric", "Original", "DSatur", "Backtracking")

    rows = [
        _row("Circuit Depth",
             str(orig["depth"]),
             str(dsat["depth"]),
             str(back["depth"])),
        _row("Gate Count",
             str(orig["gate_count"]),
             str(dsat["gate_count"]),
             str(back["gate_count"])),
        _row("Colors Used",
             "—",
             str(dsat.get("n_colors", "N/A")),
             str(back.get("n_colors", "N/A"))),
        _row("Coloring Time",
             "—",
             _fmt_time(dsat.get("coloring_time")),
             _fmt_time(back.get("coloring_time"))),
        _row("Depth Reduction",
             "—",
             _fmt_pct(dsat.get("depth_reduction_pct")),
             _fmt_pct(back.get("depth_reduction_pct"))),
        _row("Equivalent to Orig.",
             "—",
             _fmt_bool(dsat.get("equivalent")),
             _fmt_bool(back.get("equivalent"))),
    ]

    lines = [
        "",
        "  Quantum Circuit Optimization — Performance Evaluation",
        "  " + "=" * (len(sep) - 2),
        sep,
        header,
        sep,
    ]
    for row in rows:
        lines.append(row)
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. plot_comparison
# ---------------------------------------------------------------------------

def plot_comparison(evaluation_result: dict) -> matplotlib.figure.Figure:
    """Create a 2x2 comparison figure.

    Sub-plots
    ---------
    (a) Bar chart — circuit depth for all three circuits.
    (b) Bar chart — number of colors used (DSatur vs Backtracking).
    (c) Bar chart — coloring execution time (log scale when values differ
        by more than one order of magnitude).
    (d) Table — equivalence and depth-reduction summary.

    Parameters
    ----------
    evaluation_result:
        The dict returned by :func:`evaluate_all`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    orig = evaluation_result["original"]
    dsat = evaluation_result["dsatur"]
    back = evaluation_result["backtracking"]

    colors = ["#4472C4", "#ED7D31", "#70AD47"]  # blue, orange, green
    labels_all = ["Original", "DSatur", "Backtracking"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Quantum Circuit Optimization — Performance Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # ------------------------------------------------------------------ (a)
    ax_depth = axes[0, 0]
    depths = [orig["depth"], dsat["depth"], back["depth"]]
    bars_a = ax_depth.bar(labels_all, depths, color=colors, edgecolor="black", linewidth=0.6)
    ax_depth.set_title("(a) Circuit Depth", fontweight="bold")
    ax_depth.set_ylabel("Depth (gate layers)")
    ax_depth.set_ylim(0, max(depths) * 1.25 if max(depths) > 0 else 1)
    for bar, val in zip(bars_a, depths):
        ax_depth.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(depths) * 0.01,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_depth.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_depth.grid(axis="y", linestyle="--", alpha=0.5)

    # ------------------------------------------------------------------ (b)
    ax_colors = axes[0, 1]
    n_colors_dsat = dsat.get("n_colors") or 0
    n_colors_back = back.get("n_colors") or 0
    color_labels = ["DSatur", "Backtracking"]
    color_vals = [n_colors_dsat, n_colors_back]
    bars_b = ax_colors.bar(
        color_labels, color_vals, color=colors[1:], edgecolor="black", linewidth=0.6
    )
    ax_colors.set_title("(b) Number of Colors Used", fontweight="bold")
    ax_colors.set_ylabel("Colors (commutation groups)")
    ax_colors.set_ylim(0, max(color_vals) * 1.3 if max(color_vals) > 0 else 1)
    for bar, val in zip(bars_b, color_vals):
        ax_colors.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(color_vals) * 0.01,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax_colors.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_colors.grid(axis="y", linestyle="--", alpha=0.5)

    # ------------------------------------------------------------------ (c)
    ax_time = axes[1, 0]
    t_dsat = dsat.get("coloring_time") or 0.0
    t_back = back.get("coloring_time") or 0.0
    time_labels = ["DSatur", "Backtracking"]
    time_vals = [t_dsat, t_back]

    use_log = (
        min(v for v in time_vals if v > 0) > 0
        and max(time_vals) / max(min(v for v in time_vals if v > 0), 1e-12) > 10
        if any(v > 0 for v in time_vals)
        else False
    )

    bars_c = ax_time.bar(
        time_labels, time_vals, color=colors[1:], edgecolor="black", linewidth=0.6
    )
    ax_time.set_title("(c) Coloring Execution Time", fontweight="bold")
    ax_time.set_ylabel("Time (seconds)" + (" [log scale]" if use_log else ""))
    if use_log:
        ax_time.set_yscale("log")
    else:
        ax_time.set_ylim(0, max(time_vals) * 1.3 if max(time_vals) > 0 else 1)
    for bar, val in zip(bars_c, time_vals):
        y_pos = bar.get_height() * 1.05 if use_log else bar.get_height() + max(time_vals) * 0.01
        ax_time.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:.4f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_time.grid(axis="y", linestyle="--", alpha=0.5)

    # ------------------------------------------------------------------ (d)
    ax_table = axes[1, 1]
    ax_table.axis("off")
    ax_table.set_title("(d) Equivalence & Depth Reduction", fontweight="bold")

    table_data = [
        ["Metric", "DSatur", "Backtracking"],
        [
            "Equivalent to Original",
            "✓ Yes" if dsat.get("equivalent") else "✗ No",
            "✓ Yes" if back.get("equivalent") else "✗ No",
        ],
        [
            "Depth Reduction",
            f"{dsat.get('depth_reduction_pct', 0.0):.2f}%",
            f"{back.get('depth_reduction_pct', 0.0):.2f}%",
        ],
        [
            "Optimized Depth",
            str(dsat["depth"]),
            str(back["depth"]),
        ],
        [
            "Original Depth",
            str(orig["depth"]),
            str(orig["depth"]),
        ],
    ]

    tbl = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.4, 1.8)

    # Style header row
    for col_idx in range(3):
        tbl[0, col_idx].set_facecolor("#4472C4")
        tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for row_idx in range(1, len(table_data)):
        bg = "#DCE6F1" if row_idx % 2 == 0 else "white"
        for col_idx in range(3):
            tbl[row_idx, col_idx].set_facecolor(bg)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# 4. plot_depth_reduction
# ---------------------------------------------------------------------------

def plot_depth_reduction(evaluation_result: dict) -> matplotlib.figure.Figure:
    """Single bar chart showing depth reduction percentage for each optimizer.

    Bars are highlighted differently when no reduction was achieved (0%).

    Parameters
    ----------
    evaluation_result:
        The dict returned by :func:`evaluate_all`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dsat = evaluation_result["dsatur"]
    back = evaluation_result["backtracking"]

    labels = ["DSatur", "Backtracking"]
    reductions = [
        dsat.get("depth_reduction_pct", 0.0),
        back.get("depth_reduction_pct", 0.0),
    ]

    # Colour: green for genuine reduction, grey for no change, red for increase.
    def _bar_color(pct: float) -> str:
        if pct > 0:
            return "#70AD47"   # green
        elif pct == 0:
            return "#A6A6A6"   # grey
        else:
            return "#FF0000"   # red (depth increased — edge case)

    bar_colors = [_bar_color(r) for r in reductions]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, reductions, color=bar_colors, edgecolor="black", linewidth=0.7, width=0.4)

    # Annotate bars
    y_offset = max(abs(r) for r in reductions) * 0.02 if any(reductions) else 0.5
    for bar, val in zip(bars, reductions):
        va = "bottom" if val >= 0 else "top"
        y = bar.get_height() + y_offset if val >= 0 else bar.get_height() - y_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{val:.2f}%",
            ha="center",
            va=va,
            fontsize=12,
            fontweight="bold",
        )
        if val == 0.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_offset * 3,
                "(no change)",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555",
                style="italic",
            )

    # Reference line at 0
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    ax.set_title(
        "Circuit Depth Reduction vs Original",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Depth Reduction (%)", fontsize=11)
    ax.set_xlabel("Optimization Method", fontsize=11)

    # Symmetric y-limits with some padding
    max_abs = max(abs(r) for r in reductions) if reductions else 1.0
    padding = max(max_abs * 0.25, 2.0)
    ax.set_ylim(-padding if min(reductions) < 0 else -padding * 0.1, max_abs + padding)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#70AD47", edgecolor="black", label="Depth reduced"),
        Patch(facecolor="#A6A6A6", edgecolor="black", label="No change (0%)"),
        Patch(facecolor="#FF0000", edgecolor="black", label="Depth increased"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig
