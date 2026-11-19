# Module 4: Quantum Circuit Reconstruction
# Implements Algorithm 2: colored graph -> optimized circuit.
# Gates assigned the same color form a parallel execution layer.
# The number of layers (colors) gives an upper bound on the new circuit depth.

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


# ---------------------------------------------------------------------------
# Helper: apply a single gate dict onto an open QuantumCircuit
# ---------------------------------------------------------------------------

def _apply_gate(qc: QuantumCircuit, gate: dict[str, Any]) -> None:
    """Apply a single gate (described by a gate_info dict) to *qc*.

    Supported gate names (Qiskit 2.x standard names):
        cz, cp, rz, z, s, t, sdg, tdg
    """
    name: str = gate["name"].lower()
    qubits: list[int] = gate["qubits"]
    params: list[float] = gate.get("params", [])

    if name == "cz":
        qc.cz(qubits[0], qubits[1])
    elif name == "cp":
        qc.cp(params[0], qubits[0], qubits[1])
    elif name == "rz":
        qc.rz(params[0], qubits[0])
    elif name == "z":
        qc.z(qubits[0])
    elif name == "s":
        qc.s(qubits[0])
    elif name == "t":
        qc.t(qubits[0])
    elif name == "sdg":
        qc.sdg(qubits[0])
    elif name == "tdg":
        qc.tdg(qubits[0])
    else:
        raise ValueError(
            f"Unsupported gate name '{name}'. "
            "Supported: cz, cp, rz, z, s, t, sdg, tdg."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_color_layers(
    coloring: dict[int, int],
    gate_info: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group gate dicts by their assigned color (execution layer).

    Parameters
    ----------
    coloring:
        Mapping of gate_index -> color (int).  The color represents the
        execution layer produced by the graph-coloring algorithm.
    gate_info:
        List of gate descriptor dicts, each containing at minimum
        ``{"name", "qubits", "params", "index"}``.

    Returns
    -------
    dict
        ``{color_int: [gate_dict, ...]}``, sorted ascending by color key.
    """
    # Build an index for fast lookup
    index_to_gate: dict[int, dict[str, Any]] = {g["index"]: g for g in gate_info}

    layers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for gate_index, color in coloring.items():
        gate = index_to_gate.get(gate_index)
        if gate is None:
            raise KeyError(
                f"Gate index {gate_index} from coloring not found in gate_info."
            )
        layers[color].append(gate)

    # Return sorted by color (layer number)
    return dict(sorted(layers.items()))


def reconstruct_circuit(
    gate_info: list[dict[str, Any]],
    coloring: dict[int, int],
    n_qubits: int,
) -> QuantumCircuit:
    """Convert a colored conflict graph back into an optimized quantum circuit.

    This is Algorithm 2 from the paper.  Gates that share the same color have
    no qubit conflicts (by the graph-coloring construction) and can therefore
    be executed in parallel within the same layer.

    Parameters
    ----------
    gate_info:
        List of gate descriptors::

            [
                {"name": str, "qubits": list[int],
                 "params": list[float], "index": int},
                ...
            ]

    coloring:
        Mapping ``gate_index -> color`` produced by a graph-coloring algorithm
        (e.g. DSatur or backtracking).  The color is a non-negative integer
        representing the parallel execution layer.
    n_qubits:
        Total number of qubits in the circuit.

    Returns
    -------
    QuantumCircuit
        A new :class:`~qiskit.circuit.QuantumCircuit` where each color group
        forms a barrier-separated layer.  Gates within a layer are appended
        in ascending order of their original ``index`` for determinism.
    """
    qc = QuantumCircuit(n_qubits)

    color_layers = get_color_layers(coloring, gate_info)

    for layer_idx, (color, gates_in_layer) in enumerate(color_layers.items()):
        # Sort within the layer by original gate index for reproducibility
        gates_in_layer_sorted = sorted(gates_in_layer, key=lambda g: g["index"])

        for gate in gates_in_layer_sorted:
            _apply_gate(qc, gate)

        # Add a barrier after each layer (except the last) to prevent
        # Qiskit's compiler from merging gates across logical layers.
        if layer_idx < len(color_layers) - 1:
            qc.barrier()

    return qc


def circuits_are_equivalent(
    circ1: QuantumCircuit,
    circ2: QuantumCircuit,
) -> tuple[bool, float]:
    """Compare two quantum circuits for unitary equivalence up to global phase.

    Uses :class:`qiskit.quantum_info.Operator` to build the full unitary
    matrices and then compares them element-wise after normalising away the
    global phase.

    Parameters
    ----------
    circ1, circ2:
        The two circuits to compare.  They must have the same number of
        qubits.

    Returns
    -------
    (is_equivalent, metric)
        * ``is_equivalent`` – ``True`` when the circuits implement the same
          unitary up to a global phase factor.
        * ``metric`` – The maximum absolute difference between the
          phase-normalised matrices (0.0 means identical).  Returns
          ``float('inf')`` when the unitary could not be computed (e.g. the
          circuit is too large to fit in memory).
    """
    try:
        op1 = Operator(circ1)
        op2 = Operator(circ2)
    except Exception as exc:  # noqa: BLE001
        # Circuit too large or unsupported gate — cannot compare
        return False, float("inf")

    mat1: np.ndarray = op1.data
    mat2: np.ndarray = op2.data

    if mat1.shape != mat2.shape:
        return False, float("inf")

    # Normalise global phase: find the first non-zero element of mat1 and
    # rotate mat2 so that element matches.
    flat1 = mat1.ravel()
    flat2 = mat2.ravel()

    # Find first element of mat1 with significant magnitude
    pivot_idx = np.argmax(np.abs(flat1))
    pivot1 = flat1[pivot_idx]
    pivot2 = flat2[pivot_idx]

    if np.abs(pivot1) < 1e-10 or np.abs(pivot2) < 1e-10:
        # Degenerate case — fall back to direct comparison
        phase_factor = 1.0 + 0j
    else:
        # Phase factor that rotates mat2 onto mat1
        phase_factor = pivot1 / pivot2
        phase_factor /= np.abs(phase_factor)  # unit modulus

    mat2_aligned = mat2 * phase_factor

    max_diff: float = float(np.max(np.abs(mat1 - mat2_aligned)))
    is_equivalent: bool = bool(np.allclose(mat1, mat2_aligned, atol=1e-6))

    return is_equivalent, max_diff


def circuit_depth_comparison(
    original: QuantumCircuit,
    dsatur_circuit: QuantumCircuit,
    backtracking_circuit: QuantumCircuit,
) -> dict[str, Any]:
    """Produce a side-by-side depth and gate-count comparison.

    Parameters
    ----------
    original:
        The original (un-optimised) quantum circuit.
    dsatur_circuit:
        Circuit reconstructed from DSatur coloring.
    backtracking_circuit:
        Circuit reconstructed from backtracking (optimal) coloring.

    Returns
    -------
    dict
        A structured dict with the following schema::

            {
                "original": {
                    "depth": int,
                    "gate_count": int,
                    "width": int,       # number of qubits
                },
                "dsatur": {
                    "depth": int,
                    "gate_count": int,
                    "width": int,
                    "depth_reduction": float,   # fraction vs original
                    "depth_reduction_pct": str,
                },
                "backtracking": {
                    "depth": int,
                    "gate_count": int,
                    "width": int,
                    "depth_reduction": float,
                    "depth_reduction_pct": str,
                },
                "summary": {
                    "best_depth": int,
                    "best_method": str,
                    "original_depth": int,
                    "max_depth_reduction_pct": str,
                },
            }
    """
    def _stats(circ: QuantumCircuit) -> dict[str, int]:
        return {
            "depth": circ.depth(),
            "gate_count": circ.size(),
            "width": circ.num_qubits,
        }

    orig_stats = _stats(original)
    dsat_stats = _stats(dsatur_circuit)
    back_stats = _stats(backtracking_circuit)

    orig_depth: int = orig_stats["depth"]

    def _reduction(new_depth: int) -> tuple[float, str]:
        if orig_depth == 0:
            return 0.0, "0.00%"
        frac = (orig_depth - new_depth) / orig_depth
        return round(frac, 6), f"{frac * 100:.2f}%"

    dsat_red, dsat_red_pct = _reduction(dsat_stats["depth"])
    back_red, back_red_pct = _reduction(back_stats["depth"])

    best_depth = min(dsat_stats["depth"], back_stats["depth"])
    best_method = (
        "backtracking"
        if back_stats["depth"] <= dsat_stats["depth"]
        else "dsatur"
    )
    max_red_pct = (
        back_red_pct
        if back_stats["depth"] <= dsat_stats["depth"]
        else dsat_red_pct
    )

    return {
        "original": orig_stats,
        "dsatur": {
            **dsat_stats,
            "depth_reduction": dsat_red,
            "depth_reduction_pct": dsat_red_pct,
        },
        "backtracking": {
            **back_stats,
            "depth_reduction": back_red,
            "depth_reduction_pct": back_red_pct,
        },
        "summary": {
            "best_depth": best_depth,
            "best_method": best_method,
            "original_depth": orig_depth,
            "max_depth_reduction_pct": max_red_pct,
        },
    }
