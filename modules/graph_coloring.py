# Module 3: Graph Coloring Algorithms
# DSatur (Degree of Saturation): a greedy heuristic that picks the most constrained vertex first.
# Backtracking: an exact algorithm that finds the chromatic number (minimum colors needed).
# The chromatic number of the conflict graph equals the minimum circuit depth achievable
# by reordering commuting gates.

import time
import networkx as nx
from typing import Any


def dsatur(G: nx.Graph) -> dict:
    """
    DSatur (Degree of Saturation) graph coloring algorithm.

    At each step, selects the uncolored vertex with the highest saturation
    (number of distinct colors among already-colored neighbors). Ties are
    broken by highest degree, then by lowest node id.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    dict with keys:
        coloring      : dict mapping node -> color (int >= 0)
        n_colors      : int, number of distinct colors used
        time_seconds  : float, wall-clock time taken
        steps         : list of dicts, one per vertex assignment:
                        {vertex, color, saturation, degree}
    """
    start = time.perf_counter()

    nodes = list(G.nodes())
    if not nodes:
        return {
            "coloring": {},
            "n_colors": 0,
            "time_seconds": time.perf_counter() - start,
            "steps": [],
        }

    # Precompute degrees
    degree = dict(G.degree())

    # coloring: node -> int color (-1 = uncolored)
    coloring: dict[Any, int] = {v: -1 for v in nodes}

    # saturation: node -> set of colors used by colored neighbors
    neighbor_colors: dict[Any, set] = {v: set() for v in nodes}

    steps = []
    uncolored = set(nodes)

    while uncolored:
        # Pick vertex: max saturation, break tie by max degree, then min node id
        def _priority(v):
            sat = len(neighbor_colors[v])
            deg = degree[v]
            # We want max sat, max degree, min node id
            # Return a tuple to sort descending on sat and degree, ascending on id
            try:
                node_id_key = (0, v)  # comparable if nodes are homogeneous type
            except TypeError:
                node_id_key = (0, str(v))
            return (-sat, -deg, v)

        # Sort to find the best candidate (smallest tuple wins)
        try:
            chosen = min(uncolored, key=lambda v: (-len(neighbor_colors[v]), -degree[v], v))
        except TypeError:
            # Fallback if node ids are not directly comparable (mixed types)
            chosen = min(uncolored, key=lambda v: (-len(neighbor_colors[v]), -degree[v], str(v)))

        # Find smallest color not used by any neighbor
        used_by_neighbors = {coloring[nb] for nb in G.neighbors(chosen) if coloring[nb] != -1}
        color = 0
        while color in used_by_neighbors:
            color += 1

        sat = len(neighbor_colors[chosen])
        deg = degree[chosen]

        coloring[chosen] = color
        uncolored.remove(chosen)

        # Update neighbor_colors for all neighbors of chosen
        for nb in G.neighbors(chosen):
            if coloring[nb] == -1:  # still uncolored
                neighbor_colors[nb].add(color)

        steps.append({
            "vertex": chosen,
            "color": color,
            "saturation": sat,
            "degree": deg,
        })

    n_colors = len(set(coloring.values()))
    elapsed = time.perf_counter() - start

    return {
        "coloring": coloring,
        "n_colors": n_colors,
        "time_seconds": elapsed,
        "steps": steps,
    }


def backtracking_coloring(G: nx.Graph, time_limit: float = 30.0) -> dict:
    """
    Exact graph coloring via backtracking with forward checking.

    1. Runs DSatur to obtain an upper bound on the number of colors.
    2. Tries to color with k = 1, 2, ..., dsatur_colors colors.
    3. Returns the coloring for the minimum k that succeeds.

    Uses a fixed vertex ordering (by degree descending, then node id ascending).
    Forward checking: before recursing on a vertex, verifies that every
    unassigned neighbor still has at least one valid color remaining.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    time_limit : float
        Wall-clock seconds before giving up (returns best found so far).

    Returns
    -------
    dict with keys:
        coloring            : dict mapping node -> color (int >= 0)
        n_colors            : int, number of distinct colors used
        time_seconds        : float
        optimal             : bool, True if chromatic number was found
        timed_out           : bool
    """
    start = time.perf_counter()

    nodes = list(G.nodes())
    if not nodes:
        return {
            "coloring": {},
            "n_colors": 0,
            "time_seconds": time.perf_counter() - start,
            "optimal": True,
            "timed_out": False,
        }

    # Step 1: DSatur upper bound
    dsatur_result = dsatur(G)
    upper_bound = dsatur_result["n_colors"]
    best_coloring = dsatur_result["coloring"]
    best_k = upper_bound

    # Fixed vertex ordering: degree descending, node id ascending (stable)
    degree = dict(G.degree())
    try:
        order = sorted(nodes, key=lambda v: (-degree[v], v))
    except TypeError:
        order = sorted(nodes, key=lambda v: (-degree[v], str(v)))

    n = len(order)
    # Map node -> index in ordering for quick lookup
    idx_of = {v: i for i, v in enumerate(order)}

    # Adjacency sets restricted to vertices that appear EARLIER in `order`
    # (for forward checking we also need later neighbors)
    neighbors_of = {v: set(G.neighbors(v)) for v in nodes}

    timed_out = False
    optimal = False

    # We'll try k from 1 up to upper_bound
    for k in range(1, upper_bound + 1):
        if time.perf_counter() - start >= time_limit:
            timed_out = True
            break

        # coloring array indexed by position in `order`
        col = [-1] * n

        # domain[i] = set of colors still available for order[i]
        # Recomputed at each assignment during backtracking
        # We track it as a stack of (vertex_index, color_removed, affected_index) for undo

        # --- Backtracking with forward checking ---
        def _bt(pos: int) -> bool:
            """Return True if a valid k-coloring exists from position `pos` onward."""
            if time.perf_counter() - start >= time_limit:
                return False  # signal timeout

            if pos == n:
                return True

            v = order[pos]

            # Colors used by already-colored neighbors (those with pos < current)
            used = set()
            for nb in neighbors_of[v]:
                nb_pos = idx_of[nb]
                if nb_pos < pos and col[nb_pos] != -1:
                    used.add(col[nb_pos])

            for c in range(k):
                if c in used:
                    continue

                col[pos] = c

                # Forward checking: for each uncolored neighbor (later in order),
                # check it still has at least one valid color
                fc_ok = True
                for nb in neighbors_of[v]:
                    nb_pos = idx_of[nb]
                    if nb_pos <= pos:
                        continue  # already colored or current
                    # What colors are used by nb's already-assigned neighbors?
                    nb_used = set()
                    for nb2 in neighbors_of[nb]:
                        nb2_pos = idx_of[nb2]
                        if nb2_pos <= pos and col[nb2_pos] != -1:
                            nb_used.add(col[nb2_pos])
                    # Include the color we just assigned to v
                    nb_used.add(c)
                    if len(nb_used) >= k:
                        # nb would have no valid color left
                        fc_ok = False
                        break

                if fc_ok:
                    if _bt(pos + 1):
                        return True

                col[pos] = -1

            return False

        success = _bt(0)

        if time.perf_counter() - start >= time_limit:
            timed_out = True
            break

        if success:
            best_coloring = {order[i]: col[i] for i in range(n)}
            best_k = k
            optimal = True
            break  # found minimum k

    elapsed = time.perf_counter() - start

    return {
        "coloring": best_coloring,
        "n_colors": best_k,
        "time_seconds": elapsed,
        "optimal": optimal,
        "timed_out": timed_out,
    }


def validate_coloring(G: nx.Graph, coloring: dict) -> tuple:
    """
    Validate a graph coloring.

    Parameters
    ----------
    G : nx.Graph
        The graph whose edges define adjacency constraints.
    coloring : dict
        Mapping from node to integer color.

    Returns
    -------
    (True, []) if the coloring is valid (no two adjacent vertices share a color).
    (False, [(u, v, color), ...]) listing each conflicting edge and the shared color.
    """
    conflicts = []
    for u, v in G.edges():
        cu = coloring.get(u)
        cv = coloring.get(v)
        if cu is None or cv is None:
            conflicts.append((u, v, None))
        elif cu == cv:
            conflicts.append((u, v, cu))
    if conflicts:
        return (False, conflicts)
    return (True, [])


def compare_algorithms(G: nx.Graph) -> dict:
    """
    Run DSatur and Backtracking on the same graph and return a side-by-side
    comparison.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    dict with keys:
        dsatur_colors           : int
        backtracking_colors     : int
        dsatur_time             : float
        backtracking_time       : float
        dsatur_valid            : bool
        backtracking_valid      : bool
        dsatur_coloring         : dict
        backtracking_coloring   : dict
        backtracking_optimal    : bool
        backtracking_timed_out  : bool
    """
    d_result = dsatur(G)
    b_result = backtracking_coloring(G)

    d_valid, _ = validate_coloring(G, d_result["coloring"])
    b_valid, _ = validate_coloring(G, b_result["coloring"])

    return {
        "dsatur_colors": d_result["n_colors"],
        "backtracking_colors": b_result["n_colors"],
        "dsatur_time": d_result["time_seconds"],
        "backtracking_time": b_result["time_seconds"],
        "dsatur_valid": d_valid,
        "backtracking_valid": b_valid,
        "dsatur_coloring": d_result["coloring"],
        "backtracking_coloring": b_result["coloring"],
        "backtracking_optimal": b_result["optimal"],
        "backtracking_timed_out": b_result["timed_out"],
    }
