# Module 2: Conflict Graph Construction
# A conflict graph represents parallelization constraints of a commuting circuit.
# Each gate is a vertex; an edge means the two gates share a qubit and cannot run simultaneously.
# This is NOT about commutativity — commuting gates can still conflict due to shared qubits.

from __future__ import annotations

import itertools
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_label(gate: dict) -> str:
    """Build a human-readable node label such as 'CZ(0,1)' or 'RZ(2)[0.5]'."""
    name = gate.get("name", "G")
    qubits = gate.get("qubits", [])
    params = gate.get("params", [])
    qubit_str = ",".join(str(q) for q in qubits)
    if params:
        param_str = ",".join(f"{p:.3g}" for p in params)
        return f"{name}({qubit_str})[{param_str}]"
    return f"{name}({qubit_str})"


# ---------------------------------------------------------------------------
# 1. build_conflict_graph
# ---------------------------------------------------------------------------

def build_conflict_graph(gate_info: list[dict]) -> nx.Graph:
    """Construct a conflict graph from a list of gate descriptors.

    Algorithm 1 (Quantum Circuit Optimization by Graph Coloring):
      - One vertex per gate (vertex id = gate["index"]).
      - An edge (i, j) is added iff Q_i ∩ Q_j ≠ ∅  (they share at least one qubit).

    Parameters
    ----------
    gate_info : list[dict]
        Each dict must contain:
          - "name"   : str          – gate identifier (e.g. "CZ", "H", "RZ")
          - "qubits" : list[int]    – qubit indices the gate acts on
          - "params" : list[float]  – rotation/phase parameters (may be empty)
          - "index"  : int          – unique gate index used as the node id

    Returns
    -------
    nx.Graph
        Conflict graph with node attributes: name, qubits, params, label.
    """
    G = nx.Graph()

    # --- add one vertex per gate -------------------------------------------
    for gate in gate_info:
        idx = gate["index"]
        G.add_node(
            idx,
            name=gate.get("name", "G"),
            qubits=list(gate.get("qubits", [])),
            params=list(gate.get("params", [])),
            label=_make_label(gate),
        )

    # --- add edges for every conflicting pair --------------------------------
    # Two gates conflict iff their qubit sets intersect.
    for (ga, gb) in itertools.combinations(gate_info, 2):
        qa = set(ga.get("qubits", []))
        qb = set(gb.get("qubits", []))
        if qa & qb:  # non-empty intersection → conflict
            G.add_edge(ga["index"], gb["index"])

    return G


# ---------------------------------------------------------------------------
# 2. get_graph_stats
# ---------------------------------------------------------------------------

def get_graph_stats(G: nx.Graph) -> dict:
    """Return basic statistics of the conflict graph.

    Parameters
    ----------
    G : nx.Graph
        A conflict graph produced by :func:`build_conflict_graph`.

    Returns
    -------
    dict
        Keys: n_vertices, n_edges, max_degree, avg_degree, is_connected.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {
            "n_vertices": 0,
            "n_edges": 0,
            "max_degree": 0,
            "avg_degree": 0.0,
            "is_connected": True,  # vacuously
        }

    degrees = [d for _, d in G.degree()]
    max_deg = int(max(degrees))
    avg_deg = float(sum(degrees) / n)
    connected = nx.is_connected(G)

    return {
        "n_vertices": n,
        "n_edges": m,
        "max_degree": max_deg,
        "avg_degree": round(avg_deg, 4),
        "is_connected": connected,
    }


# ---------------------------------------------------------------------------
# 3. visualize_conflict_graph
# ---------------------------------------------------------------------------

def visualize_conflict_graph(
    G: nx.Graph,
    coloring: Optional[dict] = None,
    title: str = "Conflict Graph",
) -> matplotlib.figure.Figure:
    """Draw the conflict graph with optional node coloring.

    Parameters
    ----------
    G : nx.Graph
        Conflict graph.
    coloring : dict, optional
        Mapping ``{node_id: color_int}`` produced by a graph coloring algorithm.
        When provided, nodes are colored according to their assigned color class
        using a qualitative colormap (tab10).  When ``None``, all nodes share a
        single default color.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.  ``plt.show()`` is *not* called; the caller decides
        whether to display or save the figure.
    """
    fig, ax = plt.subplots(figsize=(max(6, G.number_of_nodes() * 0.8), 5))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Node colours
    if coloring is not None:
        # Determine number of distinct colors used
        color_ids = sorted(set(coloring.values()))
        n_colors = max(color_ids) + 1 if color_ids else 1
        # Use tab10 (10 qualitative colors); cycle if more colors are needed
        cmap = cm.get_cmap("tab10", max(n_colors, 10))
        node_colors = [
            mcolors.to_hex(cmap(coloring.get(node, 0) % cmap.N))
            for node in G.nodes()
        ]
    else:
        node_colors = ["#4C9BE8"] * G.number_of_nodes()

    # Labels from node attribute
    labels = {node: data.get("label", str(node))
               for node, data in G.nodes(data=True)}

    # Draw
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#888888",
        width=1.5,
        alpha=0.7,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        edgecolors="#333333",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )

    # Legend when coloring is provided
    if coloring is not None:
        cmap = cm.get_cmap("tab10", max(max(coloring.values()) + 1, 10))
        for color_id in sorted(set(coloring.values())):
            ax.scatter(
                [], [],
                color=mcolors.to_hex(cmap(color_id % cmap.N)),
                label=f"Color class {color_id}",
                s=100,
                edgecolors="#333333",
                linewidths=0.8,
            )
        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.8,
            title="Coloring",
            title_fontsize=9,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. adjacency_matrix
# ---------------------------------------------------------------------------

def adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """Return the adjacency matrix of the conflict graph as a NumPy array.

    The rows/columns are ordered by the sorted node ids.

    Parameters
    ----------
    G : nx.Graph
        Conflict graph.

    Returns
    -------
    np.ndarray
        Symmetric binary matrix of shape (n, n) where n = number of nodes.
    """
    nodelist = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodelist, dtype=np.int32)
    return A


# ---------------------------------------------------------------------------
# 5. get_node_labels
# ---------------------------------------------------------------------------

def get_node_labels(G: nx.Graph) -> dict:
    """Return a mapping from node id to its display label.

    Parameters
    ----------
    G : nx.Graph
        Conflict graph.

    Returns
    -------
    dict
        ``{node_id: label_string}`` for every node in *G*.
        Falls back to ``str(node_id)`` when the node has no "label" attribute.
    """
    return {
        node: data.get("label", str(node))
        for node, data in G.nodes(data=True)
    }


# ---------------------------------------------------------------------------
# Quick smoke-test (run this module directly to verify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_gates = [
        {"name": "H",  "qubits": [0],    "params": [],         "index": 0},
        {"name": "CZ", "qubits": [0, 1], "params": [],         "index": 1},
        {"name": "RZ", "qubits": [1],    "params": [1.5708],   "index": 2},
        {"name": "CZ", "qubits": [1, 2], "params": [],         "index": 3},
        {"name": "H",  "qubits": [2],    "params": [],         "index": 4},
        {"name": "T",  "qubits": [3],    "params": [],         "index": 5},
    ]

    G = build_conflict_graph(sample_gates)
    stats = get_graph_stats(G)
    labels = get_node_labels(G)
    A = adjacency_matrix(G)

    print("Graph stats :", stats)
    print("Node labels :", labels)
    print("Adj. matrix :\n", A)

    dummy_coloring = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 0}
    fig = visualize_conflict_graph(G, coloring=dummy_coloring, title="Smoke-test")
    fig.savefig("conflict_graph_test.png", dpi=120)
    print("Figure saved to conflict_graph_test.png")
