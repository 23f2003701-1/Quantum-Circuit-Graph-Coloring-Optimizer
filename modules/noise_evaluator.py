# Module 6: Noise-Aware Performance Evaluation (Research Extension)
# Research Question: Does reducing circuit depth via graph coloring improve noise robustness?
# This module uses Qiskit Aer to simulate depolarizing and amplitude damping noise.

import numpy as np
import warnings
from typing import Optional

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        amplitude_damping_error,
        pauli_error,
    )
    QISKIT_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Qiskit/Aer import error: {e}. Noise evaluation will be disabled.")
    QISKIT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    warnings.warn("Matplotlib not available. Plotting functions will be disabled.")
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. build_noise_model
# ---------------------------------------------------------------------------

def build_noise_model(noise_type: str, error_rate: float) -> "NoiseModel":
    """
    Build a Qiskit Aer NoiseModel.

    Parameters
    ----------
    noise_type : str
        One of ``"depolarizing"``, ``"amplitude_damping"``, or ``"combined"``.
    error_rate : float
        Base error probability (0 < error_rate < 1).

    Returns
    -------
    NoiseModel
        Configured noise model ready to pass to AerSimulator.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit Aer is not installed. Cannot build noise model.")

    noise_model = NoiseModel()

    # Gate sets
    single_qubit_gates = ["rz", "z", "s", "t", "sdg", "tdg"]
    two_qubit_gates    = ["cx", "cz", "cp"]
    measure_gates      = ["measure"]

    if noise_type == "depolarizing":
        dep_1q = depolarizing_error(error_rate, 1)
        dep_2q = depolarizing_error(min(error_rate * 2, 1.0), 2)

        noise_model.add_all_qubit_quantum_error(dep_1q, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(dep_2q, two_qubit_gates)
        noise_model.add_all_qubit_quantum_error(dep_1q, measure_gates)

    elif noise_type == "amplitude_damping":
        ad_1q = amplitude_damping_error(min(error_rate, 1.0))

        noise_model.add_all_qubit_quantum_error(ad_1q, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(ad_1q, measure_gates)
        # Amplitude damping for 2-qubit gates: tensor product of two single-qubit errors
        ad_2q = ad_1q.expand(ad_1q)
        noise_model.add_all_qubit_quantum_error(ad_2q, two_qubit_gates)

    elif noise_type == "combined":
        dep_1q = depolarizing_error(error_rate, 1)
        dep_2q = depolarizing_error(min(error_rate * 2, 1.0), 2)
        ad_1q  = amplitude_damping_error(min(error_rate, 1.0))
        ad_2q  = ad_1q.expand(ad_1q)

        # Compose depolarizing + amplitude damping for single-qubit gates
        combined_1q = dep_1q.compose(ad_1q)
        combined_2q = dep_2q.compose(ad_2q)

        noise_model.add_all_qubit_quantum_error(combined_1q, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(combined_2q, two_qubit_gates)
        noise_model.add_all_qubit_quantum_error(combined_1q, measure_gates)

    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Choose from 'depolarizing', 'amplitude_damping', or 'combined'."
        )

    return noise_model


# ---------------------------------------------------------------------------
# 2. run_simulation
# ---------------------------------------------------------------------------

def run_simulation(
    circuit: "QuantumCircuit",
    noise_model: Optional["NoiseModel"] = None,
    shots: int = 2048,
    method: str = "density_matrix",
) -> dict:
    """
    Simulate a quantum circuit with optional noise.

    Parameters
    ----------
    circuit     : QuantumCircuit  – circuit to simulate (measurements added if absent).
    noise_model : NoiseModel | None – ``None`` for ideal simulation.
    shots       : int             – number of measurement shots.
    method      : str             – ``"statevector"`` (ideal) or ``"density_matrix"`` (noisy).

    Returns
    -------
    dict with keys:
        ``"counts"``        – raw measurement counts {bitstring: int}
        ``"probabilities"`` – normalised probabilities {bitstring: float}
        ``"circuit_depth"`` – integer depth of the circuit
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit Aer is not installed. Cannot run simulation.")

    # Work on a copy so we do not mutate the caller's circuit
    circ = circuit.copy()

    # Add measurements if none are present
    if circ.num_clbits == 0 or not any(
        instr.operation.name == "measure" for instr in circ.data
    ):
        circ.measure_all()

    circuit_depth = circ.depth()

    # Build simulator kwargs
    sim_kwargs: dict = {"method": method}
    if noise_model is not None:
        sim_kwargs["noise_model"] = noise_model

    simulator = AerSimulator(**sim_kwargs)

    try:
        job    = simulator.run(circ, shots=shots)
        result = job.result()
        counts = result.get_counts(circ)
    except Exception as exc:
        # Fall back to density_matrix if statevector is too large
        warnings.warn(
            f"Simulation failed with method='{method}': {exc}. "
            "Retrying with method='density_matrix'."
        )
        sim_kwargs["method"] = "density_matrix"
        simulator = AerSimulator(**sim_kwargs)
        job    = simulator.run(circ, shots=shots)
        result = job.result()
        counts = result.get_counts(circ)

    total = sum(counts.values())
    probabilities = {k: v / total for k, v in counts.items()}

    return {
        "counts":        counts,
        "probabilities": probabilities,
        "circuit_depth": circuit_depth,
    }


# ---------------------------------------------------------------------------
# 3. compute_fidelity
# ---------------------------------------------------------------------------

def compute_fidelity(
    ideal_counts: dict,
    noisy_counts: dict,
    n_qubits: int,
) -> float:
    """
    Estimate state fidelity via the Bhattacharyya coefficient.

    Fidelity = ( sum_x  sqrt( p_ideal(x) * p_noisy(x) ) )^2

    Parameters
    ----------
    ideal_counts : dict  – counts from ideal simulation.
    noisy_counts : dict  – counts from noisy simulation.
    n_qubits     : int   – number of qubits (determines Hilbert space size).

    Returns
    -------
    float in [0, 1]
    """
    dim = 2 ** n_qubits
    all_bitstrings = [format(i, f"0{n_qubits}b") for i in range(dim)]

    total_ideal = sum(ideal_counts.values()) or 1
    total_noisy = sum(noisy_counts.values()) or 1

    p_ideal = np.array(
        [ideal_counts.get(b, 0) / total_ideal for b in all_bitstrings],
        dtype=float,
    )
    p_noisy = np.array(
        [noisy_counts.get(b, 0) / total_noisy for b in all_bitstrings],
        dtype=float,
    )

    bhattacharyya = float(np.sum(np.sqrt(p_ideal * p_noisy)))
    fidelity = bhattacharyya ** 2
    return float(np.clip(fidelity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 4. evaluate_noise
# ---------------------------------------------------------------------------

def evaluate_noise(
    original_circuit: "QuantumCircuit",
    dsatur_circuit: "QuantumCircuit",
    backtracking_circuit: "QuantumCircuit",
    noise_type: str = "depolarizing",
    error_rate: float = 0.01,
    shots: int = 2048,
) -> dict:
    """
    Run ideal and noisy simulations for all three circuits and report results.

    Parameters
    ----------
    original_circuit     : QuantumCircuit – baseline (unoptimised) circuit.
    dsatur_circuit       : QuantumCircuit – circuit optimised via DSatur coloring.
    backtracking_circuit : QuantumCircuit – circuit optimised via backtracking coloring.
    noise_type           : str            – passed to :func:`build_noise_model`.
    error_rate           : float          – base error probability.
    shots                : int            – measurement shots per simulation.

    Returns
    -------
    dict
        Nested dictionary::

            {
              "original":    {"ideal": {...}, "noisy": {...}, "fidelity": float,
                              "depth_original": int, "depth_noisy": int},
              "dsatur":      { ... },
              "backtracking":{ ... },
              "noise_type":  str,
              "error_rate":  float,
              "shots":       int,
            }
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit Aer is not installed. Cannot evaluate noise.")

    noise_model = build_noise_model(noise_type, error_rate)

    circuits = {
        "original":     original_circuit,
        "dsatur":       dsatur_circuit,
        "backtracking": backtracking_circuit,
    }

    results: dict = {
        "noise_type": noise_type,
        "error_rate":  error_rate,
        "shots":       shots,
    }

    for label, circ in circuits.items():
        n_qubits = circ.num_qubits

        # Ideal simulation
        try:
            ideal_result = run_simulation(
                circ,
                noise_model=None,
                shots=shots,
                method="statevector",
            )
        except Exception as exc:
            warnings.warn(f"Ideal simulation failed for '{label}': {exc}")
            ideal_result = {"counts": {}, "probabilities": {}, "circuit_depth": circ.depth()}

        # Noisy simulation
        try:
            noisy_result = run_simulation(
                circ,
                noise_model=noise_model,
                shots=shots,
                method="density_matrix",
            )
        except Exception as exc:
            warnings.warn(f"Noisy simulation failed for '{label}': {exc}")
            noisy_result = {"counts": {}, "probabilities": {}, "circuit_depth": circ.depth()}

        fidelity = compute_fidelity(
            ideal_result["counts"],
            noisy_result["counts"],
            n_qubits,
        )

        results[label] = {
            "ideal":          ideal_result,
            "noisy":          noisy_result,
            "fidelity":       fidelity,
            "depth_ideal":    ideal_result["circuit_depth"],
            "depth_noisy":    noisy_result["circuit_depth"],
            "n_qubits":       n_qubits,
        }

    return results


# ---------------------------------------------------------------------------
# 5. plot_noise_comparison
# ---------------------------------------------------------------------------

def plot_noise_comparison(noise_results: dict):
    """
    Create a summary comparison figure.

    Sub-plots
    ---------
    (a) Bar chart of fidelity for original / dsatur / backtracking.
    (b) Probability distribution overlays (ideal vs noisy) for each circuit.

    Parameters
    ----------
    noise_results : dict – output of :func:`evaluate_noise`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("Matplotlib is not installed. Cannot plot.")

    circuit_labels = ["original", "dsatur", "backtracking"]
    display_labels = ["Original", "DSatur", "Backtracking"]
    colors         = ["#4C72B0", "#DD8452", "#55A868"]

    fidelities = [
        noise_results.get(lbl, {}).get("fidelity", 0.0)
        for lbl in circuit_labels
    ]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Noise Evaluation  |  type={noise_results.get('noise_type', 'N/A')}  "
        f"error_rate={noise_results.get('error_rate', 'N/A')}",
        fontsize=14,
        fontweight="bold",
    )

    # ---- (a) Fidelity bar chart ----------------------------------------
    ax_fid = fig.add_subplot(2, 1, 1)
    bars = ax_fid.bar(display_labels, fidelities, color=colors, edgecolor="black", width=0.5)
    ax_fid.set_ylim(0, 1.05)
    ax_fid.set_ylabel("Fidelity (Bhattacharyya)", fontsize=12)
    ax_fid.set_title("(a) Circuit Fidelity Under Noise", fontsize=12)
    ax_fid.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Ideal fidelity")
    ax_fid.legend(fontsize=10)

    for bar, fid in zip(bars, fidelities):
        ax_fid.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{fid:.4f}",
            ha="center", va="bottom", fontsize=11,
        )

    # ---- (b) Probability distributions ---------------------------------
    for col_idx, (lbl, disp, color) in enumerate(
        zip(circuit_labels, display_labels, colors)
    ):
        ax = fig.add_subplot(2, 3, 4 + col_idx)
        entry = noise_results.get(lbl, {})
        ideal_probs = entry.get("ideal", {}).get("probabilities", {})
        noisy_probs = entry.get("noisy", {}).get("probabilities", {})

        all_keys = sorted(set(ideal_probs) | set(noisy_probs))
        if not all_keys:
            ax.set_title(f"(b) {disp} – no data")
            continue

        x      = np.arange(len(all_keys))
        width  = 0.4
        ideal_vals = [ideal_probs.get(k, 0.0) for k in all_keys]
        noisy_vals = [noisy_probs.get(k, 0.0) for k in all_keys]

        ax.bar(x - width / 2, ideal_vals, width, label="Ideal",
               color=color, alpha=0.8, edgecolor="black")
        ax.bar(x + width / 2, noisy_vals, width, label="Noisy",
               color=color, alpha=0.4, edgecolor="black", hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(all_keys, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_title(f"(b) {disp}", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# 6. plot_probability_distributions
# ---------------------------------------------------------------------------

def plot_probability_distributions(noise_results: dict):
    """
    Plot ideal vs noisy probability distributions for each of the three circuits.

    Parameters
    ----------
    noise_results : dict – output of :func:`evaluate_noise`.

    Returns
    -------
    matplotlib.figure.Figure  – 1-row, 3-column figure.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("Matplotlib is not installed. Cannot plot.")

    circuit_labels = ["original", "dsatur", "backtracking"]
    display_labels = ["Original", "DSatur", "Backtracking"]
    colors         = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Measurement Probability Distributions  |  "
        f"type={noise_results.get('noise_type', 'N/A')}  "
        f"error_rate={noise_results.get('error_rate', 'N/A')}",
        fontsize=13,
        fontweight="bold",
    )

    ideal_patch = mpatches.Patch(color="steelblue", alpha=0.85, label="Ideal")
    noisy_patch = mpatches.Patch(color="tomato",   alpha=0.55, label="Noisy", hatch="//")

    for ax, lbl, disp, color in zip(axes, circuit_labels, display_labels, colors):
        entry = noise_results.get(lbl, {})
        ideal_probs = entry.get("ideal", {}).get("probabilities", {})
        noisy_probs = entry.get("noisy", {}).get("probabilities", {})
        fidelity    = entry.get("fidelity", float("nan"))
        depth       = entry.get("depth_ideal", "?")

        all_keys = sorted(set(ideal_probs) | set(noisy_probs))
        if not all_keys:
            ax.set_title(f"{disp} – no data")
            ax.axis("off")
            continue

        x          = np.arange(len(all_keys))
        width      = 0.4
        ideal_vals = [ideal_probs.get(k, 0.0) for k in all_keys]
        noisy_vals = [noisy_probs.get(k, 0.0) for k in all_keys]

        ax.bar(x - width / 2, ideal_vals, width,
               label="Ideal", color="steelblue", alpha=0.85, edgecolor="black")
        ax.bar(x + width / 2, noisy_vals, width,
               label="Noisy", color="tomato", alpha=0.55, edgecolor="black", hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(all_keys, rotation=45, ha="right",
                           fontsize=max(6, 10 - len(all_keys) // 4))
        ax.set_ylabel("Probability", fontsize=10)
        ax.set_xlabel("Bitstring", fontsize=10)
        ax.set_title(
            f"{disp}\n"
            f"Depth={depth}  |  Fidelity={fidelity:.4f}",
            fontsize=10,
        )
        ax.set_ylim(0, 1.05)
        ax.legend(handles=[ideal_patch, noisy_patch], fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Quick self-test (run this module directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit Aer not available – skipping self-test.")
    else:
        from qiskit.circuit.library import QFT

        n = 3
        original     = QFT(n)
        dsatur_circ  = QFT(n).decompose()
        backtracking = QFT(n).decompose().decompose()

        print("Running noise evaluation self-test …")
        results = evaluate_noise(
            original,
            dsatur_circ,
            backtracking,
            noise_type="depolarizing",
            error_rate=0.01,
            shots=512,
        )

        for key in ["original", "dsatur", "backtracking"]:
            fid   = results[key]["fidelity"]
            depth = results[key]["depth_ideal"]
            print(f"  {key:>12s}  depth={depth}  fidelity={fid:.4f}")

        if MATPLOTLIB_AVAILABLE:
            fig1 = plot_noise_comparison(results)
            fig1.savefig("/tmp/noise_comparison.png", dpi=100)
            fig2 = plot_probability_distributions(results)
            fig2.savefig("/tmp/probability_distributions.png", dpi=100)
            print("Figures saved to /tmp/")
        print("Self-test complete.")
