"""
app.py — Streamlit Dashboard
Quantum Circuit Optimization by Graph Coloring
Based on: "Quantum Circuit Optimization by Graph Coloring" (arXiv:2501.14447)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="QC Optimizer — Graph Coloring",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports ────────────────────────────────────────────────────────────
from modules.circuit_generator import (
    get_predefined_circuits,
    generate_random_commuting_circuit,
    verify_commuting_circuit,
    extract_gate_info,
    circuit_summary,
)
from modules.conflict_graph import (
    build_conflict_graph,
    get_graph_stats,
    visualize_conflict_graph,
    adjacency_matrix,
    get_node_labels,
)
from modules.graph_coloring import (
    dsatur,
    backtracking_coloring,
    validate_coloring,
)
from modules.circuit_reconstructor import (
    reconstruct_circuit,
    circuits_are_equivalent,
    get_color_layers,
    circuit_depth_comparison,
)
from modules.performance_evaluator import (
    evaluate_all,
    format_comparison_table,
    plot_comparison,
    plot_depth_reduction,
)
from modules.noise_evaluator import (
    build_noise_model,
    run_simulation,
    evaluate_noise,
    compute_fidelity,
    plot_noise_comparison,
    plot_probability_distributions,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_pyplot(fig):
    """Display a matplotlib figure in Streamlit and close it."""
    st.pyplot(fig)
    plt.close(fig)


def _run_pipeline(circuit, circuit_name="circuit"):
    """Run the full optimization pipeline and cache results in session_state."""
    key = f"pipeline_{circuit_name}_{circuit.num_qubits}_{circuit.depth()}"
    if key not in st.session_state:
        gate_info = extract_gate_info(circuit)
        G = build_conflict_graph(gate_info)
        graph_stats = get_graph_stats(G)

        with st.spinner("Running DSatur coloring…"):
            d_result = dsatur(G)
        with st.spinner("Running Backtracking coloring…"):
            b_result = backtracking_coloring(G)

        d_circ = reconstruct_circuit(gate_info, d_result["coloring"], circuit.num_qubits)
        b_circ = reconstruct_circuit(gate_info, b_result["coloring"], circuit.num_qubits)

        eq_d, diff_d = circuits_are_equivalent(circuit, d_circ)
        eq_b, diff_b = circuits_are_equivalent(circuit, b_circ)

        eval_result = evaluate_all(
            circuit, d_circ, b_circ,
            {"n_colors": d_result["n_colors"], "coloring_time": d_result["time_seconds"]},
            {"n_colors": b_result["n_colors"], "coloring_time": b_result["time_seconds"]},
            eq_d, eq_b,
        )

        st.session_state[key] = {
            "gate_info": gate_info,
            "G": G,
            "graph_stats": graph_stats,
            "d_result": d_result,
            "b_result": b_result,
            "d_circ": d_circ,
            "b_circ": b_circ,
            "eq_d": eq_d,
            "eq_b": eq_b,
            "diff_d": diff_d,
            "diff_b": diff_b,
            "eval_result": eval_result,
        }
    return st.session_state[key]


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

n_qubits = st.sidebar.slider("Number of Qubits", min_value=3, max_value=8, value=4, step=1)

PREDEFINED_NAMES = ["Linear Chain", "Full Parallel", "Mixed", "Dense", "Random"]
circuit_choice = st.sidebar.selectbox("Circuit Type", PREDEFINED_NAMES, index=2)

rand_n_gates = None
rand_seed = None
if circuit_choice == "Random":
    rand_n_gates = st.sidebar.slider("Number of Gates", min_value=5, max_value=25, value=10)
    rand_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("**Research Paper:**")
st.sidebar.markdown("[arXiv:2501.14447](https://arxiv.org/abs/2501.14447)")
st.sidebar.markdown("Lee, Jeong & Kim (2025)")


# ── Load circuit ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_circuit(n_qubits, choice, n_gates=None, seed=None):
    if choice == "Random":
        circ = generate_random_commuting_circuit(n_qubits, n_gates, seed=seed)
    else:
        predefined = get_predefined_circuits(n_qubits)
        circ = predefined[choice]
    return circ


try:
    circuit = load_circuit(n_qubits, circuit_choice, rand_n_gates, rand_seed)
    is_valid, validity_msg = verify_commuting_circuit(circuit)
except Exception as e:
    st.error(f"Circuit creation failed: {e}")
    st.stop()


# ── Main header ───────────────────────────────────────────────────────────────

st.title("⚛️ Quantum Circuit Optimization by Graph Coloring")
st.markdown(
    "> *Implements the methodology of* **\"Quantum Circuit Optimization by Graph Coloring\"** "
    "*(Lee, Jeong & Kim — arXiv:2501.14447)*"
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🔌 Input Circuit",
    "🕸️ Conflict Graph",
    "🎨 Graph Coloring",
    "⚡ Optimized Circuits",
    "📊 Performance",
    "📡 Noise Evaluation",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — INPUT CIRCUIT
# ════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("🔌 Input Circuit")

    # Validity banner
    if is_valid:
        st.success(f"✅ {validity_msg}")
    else:
        st.error(f"❌ {validity_msg}")
        st.stop()

    # Circuit summary metrics
    summary = circuit_summary(circuit)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Qubits", summary["n_qubits"])
    col2.metric("Circuit Depth", summary["depth"])
    col3.metric("Gate Count", summary["gate_count"])
    col4.metric("Gate Types", len(summary["gate_types"]))

    st.markdown("**Gate composition:**")
    gate_comp = {k: v for k, v in summary["gate_types"].items()}
    st.json(gate_comp)

    # Circuit diagram
    st.markdown("**Circuit diagram:**")
    try:
        circ_text = str(circuit.draw(output="text"))
        st.code(circ_text, language="text")
    except Exception:
        st.info("Circuit diagram unavailable.")

    # Gate table
    st.markdown("**Gate list:**")
    gate_info = extract_gate_info(circuit)
    gate_rows = []
    for g in gate_info:
        params_str = ", ".join(f"{p:.4f}" for p in g["params"]) if g["params"] else "—"
        gate_rows.append({
            "Index": g["index"],
            "Gate": g["name"].upper(),
            "Qubits": str(g["qubits"]),
            "Parameters": params_str,
        })
    st.dataframe(pd.DataFrame(gate_rows), use_container_width=True)

    # Why commuting?
    with st.expander("ℹ️ Why can these gates be reordered freely?"):
        st.markdown("""
All supported gates (CZ, CP, RZ, Z, S, T, Sdg, Tdg) are **diagonal** in the
computational basis. A diagonal gate acts as:

$$D|x\\rangle = e^{i\\phi(x)}|x\\rangle$$

Any two diagonal matrices commute, since diagonal multiplication is component-wise:

$$(D_1 D_2)_{xx} = D_{1,xx} \\cdot D_{2,xx} = D_{2,xx} \\cdot D_{1,xx} = (D_2 D_1)_{xx}$$

Therefore the **order of these gates does not change the unitary**, and
we can freely reorder them to minimise circuit depth.
""")


# ════════════════════════════════════════════════════════════════════
# TAB 2 — CONFLICT GRAPH
# ════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("🕸️ Conflict Graph")

    pipe = _run_pipeline(circuit, f"{circuit_choice}_{n_qubits}")
    G = pipe["G"]
    stats = pipe["graph_stats"]

    st.markdown("""
The **conflict graph** encodes which gates *cannot* run simultaneously.

- Each gate → one vertex
- Edge (i, j) ↔ gates i and j share at least one qubit: $Q_i \\cap Q_j \\neq \\emptyset$

This is **Algorithm 1** from the paper.
""")

    # Stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vertices (Gates)", stats["n_vertices"])
    c2.metric("Edges (Conflicts)", stats["n_edges"])
    c3.metric("Max Degree", stats["max_degree"])
    c4.metric("Avg Degree", f"{stats['avg_degree']:.2f}")
    c5.metric("Connected", "Yes" if stats["is_connected"] else "No")

    # Graph visualization (uncolored)
    st.markdown("**Conflict graph (uncolored):**")
    fig_g = visualize_conflict_graph(G, coloring=None, title="Conflict Graph")
    _safe_pyplot(fig_g)

    # Node ↔ Gate legend
    st.markdown("**Vertex → Gate mapping:**")
    labels = get_node_labels(G)
    rows = [{"Vertex ID": node, "Gate": label} for node, label in sorted(labels.items())]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Adjacency matrix
    with st.expander("📐 Adjacency Matrix"):
        A = adjacency_matrix(G)
        node_ids = sorted(G.nodes())
        adj_df = pd.DataFrame(A, index=node_ids, columns=node_ids, dtype=int)
        st.dataframe(adj_df, use_container_width=True)
        st.caption("Entry [i,j] = 1 means gates i and j share a qubit and cannot run in the same layer.")

    # Edge list with explanation
    with st.expander("📋 Conflict Edge List"):
        node_data = dict(G.nodes(data=True))
        edge_rows = []
        for u, v in sorted(G.edges()):
            qu = node_data[u]["qubits"]
            qv = node_data[v]["qubits"]
            shared = sorted(set(qu) & set(qv))
            edge_rows.append({
                "Gate A (vertex)": f"{node_data[u]['label']} (v{u})",
                "Gate B (vertex)": f"{node_data[v]['label']} (v{v})",
                "Shared Qubits": str(shared),
                "Conflict Reason": f"Q_{u} ∩ Q_{v} = {shared} ≠ ∅",
            })
        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPH COLORING
# ════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("🎨 Graph Coloring")

    pipe = _run_pipeline(circuit, f"{circuit_choice}_{n_qubits}")
    G = pipe["G"]
    d_result = pipe["d_result"]
    b_result = pipe["b_result"]

    st.markdown("""
Graph coloring assigns a **color (layer number)** to each gate vertex such that
no two adjacent vertices share a color. The number of colors = minimum circuit depth
achievable by reordering commuting gates.
""")

    col_d, col_b = st.columns(2)

    # ── DSatur ──────────────────────────────────────────────────────
    with col_d:
        st.subheader("🔵 DSatur (Heuristic)")
        st.metric("Colors Used", d_result["n_colors"])
        st.metric("Execution Time", f"{d_result['time_seconds']*1000:.3f} ms")
        d_valid, d_conflicts = validate_coloring(G, d_result["coloring"])
        if d_valid:
            st.success("✅ Valid coloring")
        else:
            st.error(f"❌ Invalid: {len(d_conflicts)} conflicts")

        fig_d = visualize_conflict_graph(
            G, coloring=d_result["coloring"],
            title=f"DSatur Coloring ({d_result['n_colors']} colors)"
        )
        _safe_pyplot(fig_d)

        with st.expander("Step-by-step DSatur assignments"):
            steps_df = pd.DataFrame(d_result["steps"])
            steps_df.columns = ["Vertex", "Color Assigned", "Saturation", "Degree"]
            steps_df["Gate"] = steps_df["Vertex"].apply(
                lambda v: G.nodes[v].get("label", str(v))
            )
            st.dataframe(steps_df[["Vertex", "Gate", "Saturation", "Degree", "Color Assigned"]],
                         use_container_width=True)

    # ── Backtracking ─────────────────────────────────────────────────
    with col_b:
        st.subheader("🟢 Backtracking (Exact)")
        st.metric("Colors Used", b_result["n_colors"])
        st.metric("Execution Time", f"{b_result['time_seconds']*1000:.3f} ms")
        b_valid, b_conflicts = validate_coloring(G, b_result["coloring"])
        if b_valid:
            st.success("✅ Valid coloring")
        else:
            st.error(f"❌ Invalid: {len(b_conflicts)} conflicts")

        if b_result["optimal"]:
            st.info("🏆 Optimal (chromatic number found)")
        elif b_result["timed_out"]:
            st.warning("⏱️ Time limit reached — result may not be optimal")

        fig_b = visualize_conflict_graph(
            G, coloring=b_result["coloring"],
            title=f"Backtracking Coloring ({b_result['n_colors']} colors)"
        )
        _safe_pyplot(fig_b)

        with st.expander("Color class assignments"):
            color_to_gates = {}
            node_data = dict(G.nodes(data=True))
            for node, color in sorted(b_result["coloring"].items()):
                color_to_gates.setdefault(color, []).append(
                    node_data[node].get("label", str(node))
                )
            rows_b = [{"Color (Layer)": c, "Gates in Layer": ", ".join(gates)}
                      for c, gates in sorted(color_to_gates.items())]
            st.dataframe(pd.DataFrame(rows_b), use_container_width=True)

    # Algorithm comparison
    st.markdown("---")
    st.subheader("Algorithm Comparison")
    cmp_data = {
        "Metric": ["Colors Used", "Execution Time (ms)", "Valid Coloring", "Optimal"],
        "DSatur": [
            d_result["n_colors"],
            f"{d_result['time_seconds']*1000:.3f}",
            "✅ Yes" if d_valid else "❌ No",
            "— (heuristic)",
        ],
        "Backtracking": [
            b_result["n_colors"],
            f"{b_result['time_seconds']*1000:.3f}",
            "✅ Yes" if b_valid else "❌ No",
            "✅ Yes" if b_result["optimal"] else "⏱️ Timed out",
        ],
    }
    st.table(pd.DataFrame(cmp_data))

    with st.expander("ℹ️ Algorithm details"):
        st.markdown("""
**DSatur** (Degree of Saturation):
- At each step, selects the uncolored vertex with the **highest saturation** (number of distinct colors among its colored neighbors).
- Ties broken by vertex degree, then by node id.
- O(n²) time complexity. Not guaranteed optimal, but often close.

**Backtracking**:
- Tries k = 1, 2, … colors until a valid coloring exists.
- Uses forward checking to prune the search tree early.
- Finds the **chromatic number** χ(G) — the theoretical minimum circuit depth.
- Has a configurable time limit (30s default) to prevent runaway computation.
""")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZED CIRCUITS
# ════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("⚡ Optimized Circuits")

    pipe = _run_pipeline(circuit, f"{circuit_choice}_{n_qubits}")
    d_result = pipe["d_result"]
    b_result = pipe["b_result"]
    d_circ = pipe["d_circ"]
    b_circ = pipe["b_circ"]
    gate_info = pipe["gate_info"]

    st.markdown("""
**Algorithm 2** from the paper: group gates by their color → each color class forms
one parallel execution layer in the reconstructed circuit.
""")

    orig_depth = circuit.depth()
    d_depth = d_circ.depth()
    b_depth = b_circ.depth()

    # Depth comparison metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Original Depth", orig_depth)
    c2.metric("DSatur Depth", d_depth,
              delta=f"{d_depth - orig_depth:+d} layers",
              delta_color="inverse")
    c3.metric("Backtracking Depth", b_depth,
              delta=f"{b_depth - orig_depth:+d} layers",
              delta_color="inverse")

    # Sub-tabs for two circuits
    sub_tabs = st.tabs(["🔵 DSatur Circuit", "🟢 Backtracking Circuit"])

    with sub_tabs[0]:
        st.markdown(f"**Depth:** {d_depth} &nbsp;|&nbsp; **Gates:** {d_circ.size()} &nbsp;|&nbsp; **Colors:** {d_result['n_colors']}")
        try:
            st.code(str(d_circ.draw(output="text")), language="text")
        except Exception:
            st.info("Circuit diagram unavailable.")

        st.markdown("**Execution layers (by color class):**")
        d_layers = get_color_layers(d_result["coloring"], gate_info)
        layer_rows_d = []
        for color, gates in sorted(d_layers.items()):
            gate_labels = [f"{g['name'].upper()}({','.join(str(q) for q in g['qubits'])})" for g in gates]
            layer_rows_d.append({"Layer (Color)": color, "Gates (run in parallel)": " | ".join(gate_labels)})
        st.dataframe(pd.DataFrame(layer_rows_d), use_container_width=True)

    with sub_tabs[1]:
        st.markdown(f"**Depth:** {b_depth} &nbsp;|&nbsp; **Gates:** {b_circ.size()} &nbsp;|&nbsp; **Colors:** {b_result['n_colors']}")
        try:
            st.code(str(b_circ.draw(output="text")), language="text")
        except Exception:
            st.info("Circuit diagram unavailable.")

        st.markdown("**Execution layers (by color class):**")
        b_layers = get_color_layers(b_result["coloring"], gate_info)
        layer_rows_b = []
        for color, gates in sorted(b_layers.items()):
            gate_labels = [f"{g['name'].upper()}({','.join(str(q) for q in g['qubits'])})" for g in gates]
            layer_rows_b.append({"Layer (Color)": color, "Gates (run in parallel)": " | ".join(gate_labels)})
        st.dataframe(pd.DataFrame(layer_rows_b), use_container_width=True)

    # Equivalence check
    st.markdown("---")
    st.subheader("🔬 Unitary Equivalence Verification")
    eq_d = pipe["eq_d"]
    eq_b = pipe["eq_b"]
    diff_d = pipe["diff_d"]
    diff_b = pipe["diff_b"]

    eq_col1, eq_col2 = st.columns(2)
    with eq_col1:
        if eq_d:
            st.success(f"✅ Original ≡ DSatur (max diff = {diff_d:.2e})")
        else:
            st.error(f"❌ NOT equivalent (max diff = {diff_d:.2e})")
    with eq_col2:
        if eq_b:
            st.success(f"✅ Original ≡ Backtracking (max diff = {diff_b:.2e})")
        else:
            st.error(f"❌ NOT equivalent (max diff = {diff_b:.2e})")

    with st.expander("ℹ️ How is equivalence checked?"):
        st.markdown("""
We use `qiskit.quantum_info.Operator` to compute the full unitary matrix of each circuit,
then compare them **up to global phase**:

1. Build U₁ and U₂ as complex matrices.
2. Find the first element of U₁ with significant magnitude.
3. Multiply U₂ by a phase factor that aligns that element.
4. Check `np.allclose(U₁, U₂_aligned, atol=1e-6)`.

A result of `True` with `max_diff ≈ 0` confirms the optimization did not change the computation.
""")


# ════════════════════════════════════════════════════════════════════
# TAB 5 — PERFORMANCE COMPARISON
# ════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("📊 Performance Comparison")

    pipe = _run_pipeline(circuit, f"{circuit_choice}_{n_qubits}")
    eval_result = pipe["eval_result"]

    orig = eval_result["original"]
    dsat = eval_result["dsatur"]
    back = eval_result["backtracking"]

    # Summary table
    st.subheader("Summary Table")
    table_data = {
        "Metric": ["Circuit Depth", "Gate Count", "Colors Used", "Coloring Time (ms)", "Depth Reduction", "Equivalent to Original"],
        "Original": [
            orig["depth"], orig["gate_count"], "—", "—", "—", "—"
        ],
        "DSatur": [
            dsat["depth"],
            dsat["gate_count"],
            dsat.get("n_colors", "N/A"),
            f"{(dsat.get('coloring_time') or 0)*1000:.3f}",
            f"{dsat.get('depth_reduction_pct', 0):.2f}%",
            "✅ Yes" if dsat.get("equivalent") else "❌ No",
        ],
        "Backtracking": [
            back["depth"],
            back["gate_count"],
            back.get("n_colors", "N/A"),
            f"{(back.get('coloring_time') or 0)*1000:.3f}",
            f"{back.get('depth_reduction_pct', 0):.2f}%",
            "✅ Yes" if back.get("equivalent") else "❌ No",
        ],
    }
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # Plots
    st.subheader("Visual Comparison")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_cmp = plot_comparison(eval_result)
        _safe_pyplot(fig_cmp)
    with col_p2:
        fig_red = plot_depth_reduction(eval_result)
        _safe_pyplot(fig_red)

    # Text table
    with st.expander("📋 Text comparison table"):
        st.code(format_comparison_table(eval_result), language="text")

    # Interpretation
    st.markdown("---")
    st.subheader("📝 Interpretation")

    orig_d = orig["depth"]
    dsat_d = dsat["depth"]
    back_d = back["depth"]
    dsat_pct = dsat.get("depth_reduction_pct", 0)
    back_pct = back.get("depth_reduction_pct", 0)

    if back_d < orig_d:
        st.success(
            f"✅ **Depth reduced:** Original {orig_d} → Backtracking {back_d} "
            f"({back_pct:.1f}% reduction)"
        )
    elif back_d == orig_d:
        st.info(
            "ℹ️ **No depth reduction:** The chromatic number equals the original depth. "
            "This is expected for circuits where every gate conflicts with every other gate "
            "in a critical path."
        )

    if orig["gate_count"] == dsat["gate_count"] == back["gate_count"]:
        st.success("✅ **Gate count preserved:** Optimization only reorders gates, never removes them.")
    else:
        st.warning("⚠️ Gate count mismatch — please investigate.")


# ════════════════════════════════════════════════════════════════════
# TAB 6 — NOISE EVALUATION
# ════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("📡 Noise-Aware Evaluation")
    st.markdown("""
**Research Extension:** Does reducing circuit depth through graph coloring
improve robustness against simulated quantum noise?

This module simulates the original and both optimized circuits under identical
noise conditions and compares their output fidelity.
""")

    if circuit.num_qubits > 5:
        st.warning(
            f"⚠️ The selected circuit has {circuit.num_qubits} qubits. "
            "Noise simulation uses density_matrix method which scales exponentially. "
            "Reduce to ≤ 5 qubits for practical simulation times."
        )

    # Noise settings
    st.subheader("⚙️ Noise Configuration")
    noise_col1, noise_col2, noise_col3 = st.columns(3)
    with noise_col1:
        noise_type = st.selectbox(
            "Noise Model",
            ["depolarizing", "amplitude_damping", "combined"],
            index=0,
        )
    with noise_col2:
        error_rate = st.slider(
            "Error Rate (per gate)",
            min_value=0.001, max_value=0.10,
            value=0.01, step=0.001, format="%.3f",
        )
    with noise_col3:
        shots = st.select_slider(
            "Shots",
            options=[512, 1024, 2048, 4096],
            value=1024,
        )

    run_noise = st.button("▶️ Run Noise Simulation", type="primary")

    if run_noise:
        pipe = _run_pipeline(circuit, f"{circuit_choice}_{n_qubits}")
        d_circ = pipe["d_circ"]
        b_circ = pipe["b_circ"]

        noise_key = f"noise_{circuit_choice}_{n_qubits}_{noise_type}_{error_rate}_{shots}"

        with st.spinner(f"Running {noise_type} noise simulation ({shots} shots)…"):
            try:
                noise_results = evaluate_noise(
                    circuit, d_circ, b_circ,
                    noise_type=noise_type,
                    error_rate=error_rate,
                    shots=shots,
                )
                st.session_state[noise_key] = noise_results
            except Exception as e:
                st.error(f"Noise simulation failed: {e}")
                st.stop()

        noise_results = st.session_state.get(noise_key)

        if noise_results:
            st.markdown("---")
            st.subheader("📊 Fidelity Results")

            # Fidelity table
            fid_data = {
                "Circuit": ["Original", "DSatur", "Backtracking"],
                "Depth": [
                    noise_results["original"]["depth_ideal"],
                    noise_results["dsatur"]["depth_ideal"],
                    noise_results["backtracking"]["depth_ideal"],
                ],
                "Fidelity (vs ideal)": [
                    f"{noise_results['original']['fidelity']:.4f}",
                    f"{noise_results['dsatur']['fidelity']:.4f}",
                    f"{noise_results['backtracking']['fidelity']:.4f}",
                ],
            }
            st.dataframe(pd.DataFrame(fid_data), use_container_width=True)

            # Plots
            st.markdown("**Fidelity comparison:**")
            try:
                fig_nc = plot_noise_comparison(noise_results)
                _safe_pyplot(fig_nc)
            except Exception as e:
                st.warning(f"Fidelity plot unavailable: {e}")

            st.markdown("**Probability distributions (ideal vs noisy):**")
            try:
                fig_pd = plot_probability_distributions(noise_results)
                _safe_pyplot(fig_pd)
            except Exception as e:
                st.warning(f"Distribution plot unavailable: {e}")

            # Research conclusion
            st.markdown("---")
            st.subheader("🔬 Research Finding")

            orig_fid = noise_results["original"]["fidelity"]
            dsat_fid = noise_results["dsatur"]["fidelity"]
            back_fid = noise_results["backtracking"]["fidelity"]
            orig_d = noise_results["original"]["depth_ideal"]
            back_d = noise_results["backtracking"]["depth_ideal"]

            if back_d < orig_d:
                if back_fid > orig_fid:
                    st.success(
                        f"✅ **Depth reduction improved fidelity.** "
                        f"Backtracking reduced depth {orig_d}→{back_d} and fidelity improved "
                        f"from {orig_fid:.4f} to {back_fid:.4f} (+{back_fid-orig_fid:.4f})."
                    )
                else:
                    st.info(
                        f"ℹ️ **Depth was reduced ({orig_d}→{back_d}) but fidelity did not improve significantly.** "
                        f"Orig: {orig_fid:.4f}, BT: {back_fid:.4f}. "
                        "This can occur when noise is not strictly depth-limited "
                        "(e.g., gate-count-dominant or shot noise)."
                    )
            else:
                st.info(
                    f"ℹ️ **No depth reduction achieved for this circuit** (depth = {orig_d}). "
                    "Fidelity differences are within shot noise. "
                    "Try a circuit with more parallelism opportunities (e.g., 'Dense' or 'Mixed')."
                )

            with st.expander("ℹ️ About fidelity estimation"):
                st.markdown(f"""
**Noise model:** {noise_type}  
**Error rate:** {error_rate} per gate (2× for 2-qubit gates)  
**Shots:** {shots}

Fidelity is estimated using the **Bhattacharyya coefficient** between the ideal
and noisy probability distributions:

$$F \\approx \\left(\\sum_x \\sqrt{{p_{{ideal}}(x)\\, p_{{noisy}}(x)}}\\right)^2$$

This is an approximation. True state fidelity requires density matrix computation.
Fidelity values close to 1.0 indicate low noise impact.
""")

    else:
        st.info("👆 Configure noise parameters above and click **Run Noise Simulation** to begin.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<small>**Quantum Circuit Optimization by Graph Coloring** | "
    "Based on arXiv:2501.14447 (Lee, Jeong & Kim) | "
    "Implementation: Qiskit 2.4.1 + Qiskit Aer 0.17.2</small>",
    unsafe_allow_html=True,
)
