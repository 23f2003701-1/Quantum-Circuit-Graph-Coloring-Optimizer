"""tests/test_all.py

Comprehensive test suite for the Quantum Circuit Optimization project.
Tests cover circuit generation, conflict graph construction, graph coloring,
and circuit reconstruction modules.

Run with:
    python -m pytest tests/test_all.py -v          (if pytest is installed)
    python tests/test_all.py                        (plain unittest runner)
    python -m unittest tests.test_all -v            (unittest discovery)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import networkx as nx
from qiskit.circuit import QuantumCircuit

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
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
    adjacency_matrix,
    get_node_labels,
)
from modules.graph_coloring import (
    dsatur,
    backtracking_coloring,
    validate_coloring,
    compare_algorithms,
)
from modules.circuit_reconstructor import (
    reconstruct_circuit,
    circuits_are_equivalent,
    circuit_depth_comparison,
)


# ===========================================================================
# Helper utilities
# ===========================================================================

def _build_pipeline(qc: QuantumCircuit):
    """Run the full optimization pipeline on a circuit and return
    (gate_info, conflict_graph, dsatur_result, backtracking_result,
     dsatur_circuit, backtracking_circuit)."""
    gate_info = extract_gate_info(qc)
    G = build_conflict_graph(gate_info)
    d_res = dsatur(G)
    b_res = backtracking_coloring(G)
    d_circ = reconstruct_circuit(gate_info, d_res["coloring"], qc.num_qubits)
    b_circ = reconstruct_circuit(gate_info, b_res["coloring"], qc.num_qubits)
    return gate_info, G, d_res, b_res, d_circ, b_circ


def _make_simple_conflict_graph(gate_specs):
    """Create a conflict graph from a list of (name, qubits) tuples."""
    gate_info = [
        {"index": i, "name": name, "qubits": list(qubits), "params": []}
        for i, (name, qubits) in enumerate(gate_specs)
    ]
    return build_conflict_graph(gate_info), gate_info


# ===========================================================================
# TestCircuitGenerator
# ===========================================================================

class TestCircuitGenerator(unittest.TestCase):
    """Tests for modules/circuit_generator.py"""

    # ------------------------------------------------------------------
    # get_predefined_circuits
    # ------------------------------------------------------------------

    def test_predefined_circuits_exist(self):
        """get_predefined_circuits(4) returns a dict with exactly 4 keys."""
        circuits = get_predefined_circuits(4)
        self.assertIsInstance(circuits, dict)
        self.assertEqual(len(circuits), 4)

    def test_predefined_circuits_range(self):
        """get_predefined_circuits works for n_qubits in [3, 8] without raising."""
        for n in range(3, 9):
            with self.subTest(n_qubits=n):
                circuits = get_predefined_circuits(n)
                self.assertIsInstance(circuits, dict)
                self.assertEqual(len(circuits), 4)
                for name, circ in circuits.items():
                    self.assertIsInstance(circ, QuantumCircuit)
                    self.assertEqual(circ.num_qubits, n)

    def test_predefined_circuits_invalid_range(self):
        """get_predefined_circuits raises ValueError for n_qubits outside [3, 8]."""
        for bad_n in (2, 9):
            with self.subTest(n_qubits=bad_n):
                with self.assertRaises(ValueError):
                    get_predefined_circuits(bad_n)

    # ------------------------------------------------------------------
    # generate_random_commuting_circuit
    # ------------------------------------------------------------------

    def test_random_circuit_gate_count(self):
        """generate_random_commuting_circuit(4, 10, seed=42) has exactly 10 gates."""
        qc = generate_random_commuting_circuit(4, 10, seed=42)
        self.assertEqual(len(qc.data), 10)

    def test_random_circuit_reproducible(self):
        """Same seed produces identical circuits (instruction-by-instruction)."""
        qc1 = generate_random_commuting_circuit(4, 15, seed=0)
        qc2 = generate_random_commuting_circuit(4, 15, seed=0)
        self.assertEqual(len(qc1.data), len(qc2.data))
        for inst1, inst2 in zip(qc1.data, qc2.data):
            self.assertEqual(inst1.operation.name, inst2.operation.name)
            self.assertEqual(
                [q._index for q in inst1.qubits],
                [q._index for q in inst2.qubits],
            )

    # ------------------------------------------------------------------
    # verify_commuting_circuit
    # ------------------------------------------------------------------

    def test_verify_valid_circuit(self):
        """A circuit with only CZ, CP, RZ gates passes verification."""
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.cp(math.pi / 4, 1, 2)
        qc.rz(math.pi / 3, 0)
        valid, msg = verify_commuting_circuit(qc)
        self.assertTrue(valid, msg)

    def test_verify_invalid_circuit_h(self):
        """A circuit containing an H gate fails verification."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cz(0, 1)
        valid, _ = verify_commuting_circuit(qc)
        self.assertFalse(valid)

    def test_verify_invalid_circuit_cx(self):
        """A circuit containing a CNOT (cx) gate fails verification."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        valid, _ = verify_commuting_circuit(qc)
        self.assertFalse(valid)

    # ------------------------------------------------------------------
    # extract_gate_info
    # ------------------------------------------------------------------

    def test_extract_gate_info_count(self):
        """extract_gate_info returns one entry per gate in the circuit."""
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.rz(math.pi / 2, 2)
        qc.cp(math.pi / 3, 0, 2)
        info = extract_gate_info(qc)
        self.assertEqual(len(info), 3)

    def test_extract_gate_info_qubit_field(self):
        """Every entry returned by extract_gate_info has a 'qubits' key with a list."""
        qc = generate_random_commuting_circuit(4, 8, seed=7)
        info = extract_gate_info(qc)
        for entry in info:
            self.assertIn("qubits", entry)
            self.assertIsInstance(entry["qubits"], list)

    # ------------------------------------------------------------------
    # circuit_summary
    # ------------------------------------------------------------------

    def test_circuit_summary_keys(self):
        """circuit_summary returns a dict with the required keys."""
        required_keys = {"depth", "gate_count", "n_qubits", "gate_types"}
        qc = generate_random_commuting_circuit(4, 10, seed=1)
        summary = circuit_summary(qc)
        self.assertIsInstance(summary, dict)
        self.assertTrue(required_keys.issubset(summary.keys()))


# ===========================================================================
# TestConflictGraph
# ===========================================================================

class TestConflictGraph(unittest.TestCase):
    """Tests for modules/conflict_graph.py"""

    # ------------------------------------------------------------------
    # build_conflict_graph
    # ------------------------------------------------------------------

    def test_build_conflict_graph_vertices(self):
        """A 5-gate circuit produces a conflict graph with 5 vertices."""
        qc = generate_random_commuting_circuit(4, 5, seed=99)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        self.assertEqual(G.number_of_nodes(), 5)

    def test_build_conflict_graph_edges_shared_qubit(self):
        """Two gates that share a qubit must be connected by an edge."""
        # Gate 0: CZ on (0,1), Gate 1: RZ on qubit 1 — share qubit 1
        G, _ = _make_simple_conflict_graph([("CZ", (0, 1)), ("RZ", (1,))])
        self.assertTrue(G.has_edge(0, 1), "Gates sharing qubit 1 must be adjacent.")

    def test_build_conflict_graph_no_edge_disjoint(self):
        """Two gates on completely disjoint qubits must NOT share an edge."""
        # Gate 0: CZ on (0,1), Gate 1: CZ on (2,3) — disjoint
        G, _ = _make_simple_conflict_graph([("CZ", (0, 1)), ("CZ", (2, 3))])
        self.assertFalse(G.has_edge(0, 1), "Disjoint qubit gates must not conflict.")

    def test_conflict_graph_complete_case(self):
        """Linear chain: each gate shares a qubit with its neighbor (at least)."""
        # CZ(0,1), CZ(1,2), CZ(2,3) — each adjacent pair conflicts
        G, _ = _make_simple_conflict_graph([
            ("CZ", (0, 1)),
            ("CZ", (1, 2)),
            ("CZ", (2, 3)),
        ])
        self.assertTrue(G.has_edge(0, 1))  # share qubit 1
        self.assertTrue(G.has_edge(1, 2))  # share qubit 2
        # Gates 0 and 2 do NOT share a qubit
        self.assertFalse(G.has_edge(0, 2))

    def test_conflict_graph_star_case(self):
        """All gates on qubit 0 must form a clique (every pair conflicts)."""
        # Four single-qubit gates all on qubit 0
        specs = [("RZ", (0,)), ("RZ", (0,)), ("RZ", (0,)), ("RZ", (0,))]
        G, _ = _make_simple_conflict_graph(specs)
        nodes = list(G.nodes())
        # Every pair of the 4 nodes must be adjacent
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.assertTrue(
                    G.has_edge(nodes[i], nodes[j]),
                    f"Nodes {nodes[i]} and {nodes[j]} should be adjacent (both on qubit 0).",
                )

    def test_node_attributes(self):
        """Every node must carry 'label', 'name', and 'qubits' attributes."""
        qc = generate_random_commuting_circuit(4, 6, seed=3)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        for node, data in G.nodes(data=True):
            self.assertIn("label", data, f"Node {node} missing 'label'.")
            self.assertIn("name", data, f"Node {node} missing 'name'.")
            self.assertIn("qubits", data, f"Node {node} missing 'qubits'.")

    # ------------------------------------------------------------------
    # adjacency_matrix
    # ------------------------------------------------------------------

    def test_adjacency_matrix_shape(self):
        """adjacency_matrix returns an (n, n) numpy array."""
        qc = generate_random_commuting_circuit(4, 7, seed=5)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        A = adjacency_matrix(G)
        n = G.number_of_nodes()
        self.assertIsInstance(A, np.ndarray)
        self.assertEqual(A.shape, (n, n))

    def test_adjacency_matrix_symmetric(self):
        """The adjacency matrix of an undirected graph must be symmetric."""
        qc = generate_random_commuting_circuit(4, 8, seed=6)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        A = adjacency_matrix(G)
        np.testing.assert_array_equal(A, A.T)

    # ------------------------------------------------------------------
    # get_graph_stats
    # ------------------------------------------------------------------

    def test_get_graph_stats_keys(self):
        """get_graph_stats returns a dict with all required keys."""
        required_keys = {"n_vertices", "n_edges", "max_degree", "avg_degree", "is_connected"}
        qc = generate_random_commuting_circuit(4, 8, seed=10)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        stats = get_graph_stats(G)
        self.assertIsInstance(stats, dict)
        self.assertTrue(required_keys.issubset(stats.keys()))

    # ------------------------------------------------------------------
    # get_node_labels
    # ------------------------------------------------------------------

    def test_node_labels_complete(self):
        """get_node_labels returns a label for every node in the graph."""
        qc = generate_random_commuting_circuit(4, 6, seed=11)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        labels = get_node_labels(G)
        self.assertEqual(set(labels.keys()), set(G.nodes()))
        for node, label in labels.items():
            self.assertIsInstance(label, str)
            self.assertTrue(len(label) > 0)


# ===========================================================================
# TestGraphColoring
# ===========================================================================

class TestGraphColoring(unittest.TestCase):
    """Tests for modules/graph_coloring.py"""

    # ------------------------------------------------------------------
    # DSatur
    # ------------------------------------------------------------------

    def test_dsatur_valid_coloring(self):
        """DSatur must produce a valid coloring (no two adjacent vertices share a color)."""
        G = nx.petersen_graph()
        result = dsatur(G)
        valid, conflicts = validate_coloring(G, result["coloring"])
        self.assertTrue(valid, f"DSatur coloring invalid, conflicts: {conflicts}")

    def test_dsatur_all_vertices_colored(self):
        """DSatur must assign a color to every vertex."""
        G = nx.cycle_graph(7)
        result = dsatur(G)
        self.assertEqual(set(result["coloring"].keys()), set(G.nodes()))

    def test_dsatur_on_complete_graph(self):
        """DSatur on K_n requires exactly n colors."""
        for n in (3, 4, 5):
            with self.subTest(n=n):
                G = nx.complete_graph(n)
                result = dsatur(G)
                self.assertEqual(result["n_colors"], n)

    def test_dsatur_on_independent_set(self):
        """DSatur on a graph with no edges requires exactly 1 color."""
        G = nx.Graph()
        G.add_nodes_from(range(6))  # no edges
        result = dsatur(G)
        self.assertEqual(result["n_colors"], 1)

    def test_dsatur_on_bipartite(self):
        """DSatur on a bipartite graph requires at most 2 colors."""
        G = nx.complete_bipartite_graph(3, 4)
        result = dsatur(G)
        self.assertLessEqual(result["n_colors"], 2)

    # ------------------------------------------------------------------
    # Backtracking
    # ------------------------------------------------------------------

    def test_backtracking_valid_coloring(self):
        """Backtracking must produce a valid coloring."""
        G = nx.petersen_graph()
        result = backtracking_coloring(G)
        valid, conflicts = validate_coloring(G, result["coloring"])
        self.assertTrue(valid, f"Backtracking coloring invalid, conflicts: {conflicts}")

    def test_backtracking_optimal_on_clique(self):
        """Backtracking on K_n finds the chromatic number n."""
        for n in (3, 4, 5):
            with self.subTest(n=n):
                G = nx.complete_graph(n)
                result = backtracking_coloring(G)
                self.assertEqual(result["n_colors"], n)

    def test_backtracking_optimal_on_path(self):
        """Backtracking on a path graph finds chromatic number 2."""
        G = nx.path_graph(8)
        result = backtracking_coloring(G)
        self.assertEqual(result["n_colors"], 2)

    def test_backtracking_optimal_on_cycle_even(self):
        """Backtracking on an even cycle (C_6) finds chromatic number 2."""
        G = nx.cycle_graph(6)
        result = backtracking_coloring(G)
        self.assertEqual(result["n_colors"], 2)

    # ------------------------------------------------------------------
    # validate_coloring
    # ------------------------------------------------------------------

    def test_validate_coloring_correct(self):
        """validate_coloring returns (True, []) for a known-valid 2-coloring of K_2."""
        G = nx.complete_graph(2)
        coloring = {0: 0, 1: 1}
        valid, conflicts = validate_coloring(G, coloring)
        self.assertTrue(valid)
        self.assertEqual(conflicts, [])

    def test_validate_coloring_conflict(self):
        """validate_coloring returns (False, [...]) when two adjacent vertices share a color."""
        G = nx.complete_graph(3)  # K_3: every pair is adjacent
        bad_coloring = {0: 0, 1: 0, 2: 1}  # 0 and 1 adjacent but same color
        valid, conflicts = validate_coloring(G, bad_coloring)
        self.assertFalse(valid)
        self.assertTrue(len(conflicts) > 0)

    # ------------------------------------------------------------------
    # compare_algorithms
    # ------------------------------------------------------------------

    def test_compare_algorithms_keys(self):
        """compare_algorithms returns a dict with all required keys."""
        required_keys = {
            "dsatur_colors", "backtracking_colors",
            "dsatur_time", "backtracking_time",
            "dsatur_valid", "backtracking_valid",
            "dsatur_coloring", "backtracking_coloring",
            "backtracking_optimal", "backtracking_timed_out",
        }
        G = nx.cycle_graph(5)
        result = compare_algorithms(G)
        self.assertIsInstance(result, dict)
        self.assertTrue(required_keys.issubset(result.keys()))


# ===========================================================================
# TestCircuitReconstruction
# ===========================================================================

class TestCircuitReconstruction(unittest.TestCase):
    """Tests for modules/circuit_reconstructor.py"""

    @classmethod
    def setUpClass(cls):
        """Build a shared test circuit once to avoid repetition."""
        cls.qc4 = generate_random_commuting_circuit(4, 8, seed=42)
        cls.gate_info4 = extract_gate_info(cls.qc4)
        cls.G4 = build_conflict_graph(cls.gate_info4)
        cls.d_res4 = dsatur(cls.G4)
        cls.b_res4 = backtracking_coloring(cls.G4)
        cls.d_circ4 = reconstruct_circuit(
            cls.gate_info4, cls.d_res4["coloring"], cls.qc4.num_qubits
        )
        cls.b_circ4 = reconstruct_circuit(
            cls.gate_info4, cls.b_res4["coloring"], cls.qc4.num_qubits
        )

    # ------------------------------------------------------------------
    # reconstruct_circuit
    # ------------------------------------------------------------------

    def test_reconstruct_preserves_gate_count(self):
        """Reconstructed circuit has the same number of gates as the original."""
        orig_count = len(self.gate_info4)
        d_count = sum(1 for inst in self.d_circ4.data if inst.operation.name != "barrier")
        self.assertEqual(d_count, orig_count)

    def test_reconstruct_valid_circuit_type(self):
        """reconstruct_circuit returns a QuantumCircuit instance."""
        self.assertIsInstance(self.d_circ4, QuantumCircuit)
        self.assertIsInstance(self.b_circ4, QuantumCircuit)

    def test_reconstruct_dsatur_equivalent(self):
        """DSatur-reconstructed circuit is unitarily equivalent to the original."""
        is_eq, metric = circuits_are_equivalent(self.qc4, self.d_circ4)
        self.assertTrue(
            is_eq,
            f"DSatur circuit not equivalent to original (max_diff={metric:.2e}).",
        )

    def test_reconstruct_backtracking_equivalent(self):
        """Backtracking-reconstructed circuit is unitarily equivalent to the original."""
        is_eq, metric = circuits_are_equivalent(self.qc4, self.b_circ4)
        self.assertTrue(
            is_eq,
            f"Backtracking circuit not equivalent to original (max_diff={metric:.2e}).",
        )

    def test_reconstruct_no_extra_gates(self):
        """Gate count (excluding barriers) exactly matches original gate count."""
        orig_count = len(self.gate_info4)
        b_count = sum(1 for inst in self.b_circ4.data if inst.operation.name != "barrier")
        self.assertEqual(b_count, orig_count)

    # ------------------------------------------------------------------
    # circuits_are_equivalent
    # ------------------------------------------------------------------

    def test_circuits_equivalent_same(self):
        """A circuit must be unitarily equivalent to itself."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        qc.rz(math.pi / 4, 0)
        is_eq, metric = circuits_are_equivalent(qc, qc)
        self.assertTrue(is_eq)
        self.assertAlmostEqual(metric, 0.0, places=6)

    def test_circuits_equivalent_different(self):
        """A circuit with CZ(0,1) must NOT be equivalent to an empty circuit."""
        qc_cz = QuantumCircuit(2)
        qc_cz.cz(0, 1)
        qc_empty = QuantumCircuit(2)
        is_eq, _ = circuits_are_equivalent(qc_cz, qc_empty)
        self.assertFalse(is_eq)

    # ------------------------------------------------------------------
    # circuit_depth_comparison
    # ------------------------------------------------------------------

    def test_depth_comparison_keys(self):
        """circuit_depth_comparison returns a dict with required top-level keys."""
        required_top = {"original", "dsatur", "backtracking", "summary"}
        result = circuit_depth_comparison(self.qc4, self.d_circ4, self.b_circ4)
        self.assertIsInstance(result, dict)
        self.assertTrue(required_top.issubset(result.keys()))
        # Check sub-keys
        for section in ("original", "dsatur", "backtracking"):
            self.assertIn("depth", result[section])
            self.assertIn("gate_count", result[section])
            self.assertIn("width", result[section])
        summary_keys = {"best_depth", "best_method", "original_depth", "max_depth_reduction_pct"}
        self.assertTrue(summary_keys.issubset(result["summary"].keys()))


# ===========================================================================
# TestNoCasesTrickier
# ===========================================================================

class TestNoCasesTrickier(unittest.TestCase):
    """Edge-case and end-to-end pipeline tests."""

    def test_no_parallelism_circuit(self):
        """Linear chain: coloring depth >= original depth (cannot increase).
        
        A linear chain CZ(0,1), CZ(1,2), CZ(2,3) has every adjacent gate
        in conflict, so the conflict graph is itself a path; path chromatic
        number is 2, which equals the minimum reachable depth.
        The number of colors used must be <= original circuit depth.
        """
        qc = QuantumCircuit(4)
        qc.cz(0, 1)
        qc.cz(1, 2)
        qc.cz(2, 3)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        d_res = dsatur(G)
        b_res = backtracking_coloring(G)
        orig_depth = qc.depth()
        # Depth of reconstructed circuit must not exceed original depth
        d_circ = reconstruct_circuit(gate_info, d_res["coloring"], qc.num_qubits)
        b_circ = reconstruct_circuit(gate_info, b_res["coloring"], qc.num_qubits)
        # Barrier-free depth == number of distinct colors
        d_colors = d_res["n_colors"]
        b_colors = b_res["n_colors"]
        self.assertLessEqual(d_colors, orig_depth)
        self.assertLessEqual(b_colors, orig_depth)

    def test_maximum_parallelism_circuit(self):
        """Full parallel circuit: all gates disjoint → 1 color (depth == 1 from coloring)."""
        # CZ(0,1), CZ(2,3) — completely disjoint qubit pairs
        qc = QuantumCircuit(4)
        qc.cz(0, 1)
        qc.cz(2, 3)
        gate_info = extract_gate_info(qc)
        G = build_conflict_graph(gate_info)
        d_res = dsatur(G)
        b_res = backtracking_coloring(G)
        # Both gates are non-conflicting; chromatic number should be 1
        self.assertEqual(b_res["n_colors"], 1,
                         "Fully disjoint gates should require only 1 color.")
        self.assertEqual(d_res["n_colors"], 1,
                         "DSatur should also find 1 color for fully disjoint gates.")

    def test_dsatur_backtracking_agree_on_path(self):
        """On a path conflict graph, DSatur and backtracking both use <= 2 colors."""
        # 4-gate linear chain: CZ(0,1) → CZ(1,2) → CZ(2,3) → CZ(3,0 is not a path)
        # Use a pure path: conflict graph is a path graph P_4
        # Build manually: gate 0 conflicts with gate 1 only; gate 1 with 0 and 2; etc.
        gate_specs = [
            ("CZ", (0, 1)),  # conflicts with gate 1
            ("CZ", (1, 2)),  # conflicts with gate 0 and gate 2
            ("CZ", (2, 3)),  # conflicts with gate 1
        ]
        G, _ = _make_simple_conflict_graph(gate_specs)
        d_res = dsatur(G)
        b_res = backtracking_coloring(G)
        self.assertLessEqual(d_res["n_colors"], 2)
        self.assertLessEqual(b_res["n_colors"], 2)

    def test_full_pipeline_mixed_circuit(self):
        """End-to-end: mixed 4-qubit circuit remains unitarily equivalent after optimization."""
        circuits = get_predefined_circuits(4)
        qc = circuits["Mixed"]
        gate_info, G, d_res, b_res, d_circ, b_circ = _build_pipeline(qc)

        # Both reconstructed circuits must be equivalent to the original
        is_eq_d, diff_d = circuits_are_equivalent(qc, d_circ)
        is_eq_b, diff_b = circuits_are_equivalent(qc, b_circ)

        self.assertTrue(
            is_eq_d,
            f"DSatur circuit not equivalent to original mixed circuit (diff={diff_d:.2e}).",
        )
        self.assertTrue(
            is_eq_b,
            f"Backtracking circuit not equivalent to original mixed circuit (diff={diff_b:.2e}).",
        )

        # Sanity: optimized depth should not exceed the original depth
        orig_depth = qc.depth()
        opt_depth_d = d_res["n_colors"]
        opt_depth_b = b_res["n_colors"]
        self.assertLessEqual(opt_depth_d, orig_depth)
        self.assertLessEqual(opt_depth_b, orig_depth)



# ===========================================================================
# TestNoiseEvaluator
# ===========================================================================

# Import noise evaluator at module level to avoid Windows DLL crash in subprocess
try:
    from modules.noise_evaluator import (
        build_noise_model as _build_noise_model_fn,
        run_simulation as _run_simulation_fn,
        compute_fidelity as _compute_fidelity_fn,
        evaluate_noise as _evaluate_noise_fn,
    )
    _NOISE_AVAILABLE = True
except Exception:
    _NOISE_AVAILABLE = False


@unittest.skipUnless(_NOISE_AVAILABLE, "noise_evaluator not importable — skip noise tests")
class TestNoiseEvaluator(unittest.TestCase):
    """Tests for modules.noise_evaluator (Module 6 — Research Extension)."""

    @classmethod
    def setUpClass(cls):
        """Build a small 3-qubit circuit once for all noise tests."""
        from modules.circuit_generator import generate_random_commuting_circuit, extract_gate_info
        from modules.conflict_graph import build_conflict_graph
        from modules.graph_coloring import dsatur, backtracking_coloring
        from modules.circuit_reconstructor import reconstruct_circuit

        cls._build_noise_model = staticmethod(_build_noise_model_fn)
        cls._run_simulation = staticmethod(_run_simulation_fn)
        cls._compute_fidelity = staticmethod(_compute_fidelity_fn)
        cls._evaluate_noise = staticmethod(_evaluate_noise_fn)

        cls.n = 3
        circ = generate_random_commuting_circuit(cls.n, 5, seed=99)
        gi = extract_gate_info(circ)
        G = build_conflict_graph(gi)
        d_res = dsatur(G)
        b_res = backtracking_coloring(G)
        cls.orig_circ = circ
        cls.d_circ = reconstruct_circuit(gi, d_res["coloring"], cls.n)
        cls.b_circ = reconstruct_circuit(gi, b_res["coloring"], cls.n)

    # ── Noise model construction ───────────────────────────────────────────

    def test_build_depolarizing_model(self):
        """build_noise_model returns a NoiseModel for depolarizing noise."""
        from qiskit_aer.noise import NoiseModel
        nm = _build_noise_model_fn("depolarizing", 0.01)
        self.assertIsInstance(nm, NoiseModel)

    def test_build_amplitude_damping_model(self):
        """build_noise_model returns a NoiseModel for amplitude damping."""
        from qiskit_aer.noise import NoiseModel
        nm = _build_noise_model_fn("amplitude_damping", 0.01)
        self.assertIsInstance(nm, NoiseModel)

    def test_build_combined_model(self):
        """build_noise_model returns a NoiseModel for combined noise."""
        from qiskit_aer.noise import NoiseModel
        nm = _build_noise_model_fn("combined", 0.01)
        self.assertIsInstance(nm, NoiseModel)

    def test_build_invalid_noise_type_raises(self):
        """build_noise_model raises ValueError for unknown noise type."""
        with self.assertRaises(ValueError):
            _build_noise_model_fn("unknown_noise", 0.01)

    # ── Ideal simulation ──────────────────────────────────────────────────

    def test_ideal_simulation_returns_dict(self):
        """run_simulation with no noise model returns required keys."""
        result = _run_simulation_fn(self.orig_circ, noise_model=None, shots=256)
        self.assertIn("counts", result)
        self.assertIn("probabilities", result)
        self.assertIn("circuit_depth", result)

    def test_ideal_simulation_probabilities_sum_to_one(self):
        """Ideal simulation probabilities sum to approximately 1."""
        result = _run_simulation_fn(self.orig_circ, noise_model=None, shots=512)
        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_ideal_simulation_depth_positive(self):
        """Ideal simulation reports a positive circuit depth."""
        result = _run_simulation_fn(self.orig_circ, noise_model=None, shots=256)
        self.assertGreater(result["circuit_depth"], 0)

    # ── Noisy simulation ──────────────────────────────────────────────────

    def test_noisy_simulation_runs(self):
        """Noisy simulation completes without error."""
        nm = _build_noise_model_fn("depolarizing", 0.02)
        result = _run_simulation_fn(self.orig_circ, noise_model=nm, shots=256,
                                    method="density_matrix")
        self.assertIn("counts", result)

    def test_noisy_probabilities_sum_to_one(self):
        """Noisy simulation probabilities sum to approximately 1."""
        nm = _build_noise_model_fn("depolarizing", 0.02)
        result = _run_simulation_fn(self.orig_circ, noise_model=nm, shots=512,
                                    method="density_matrix")
        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    # ── Fidelity ──────────────────────────────────────────────────────────

    def test_compute_fidelity_range(self):
        """compute_fidelity returns a value in [0, 1]."""
        ideal = _run_simulation_fn(self.orig_circ, noise_model=None, shots=512)
        nm = _build_noise_model_fn("depolarizing", 0.05)
        noisy = _run_simulation_fn(self.orig_circ, noise_model=nm, shots=512,
                                   method="density_matrix")
        fid = _compute_fidelity_fn(ideal["counts"], noisy["counts"], self.n)
        self.assertGreaterEqual(fid, 0.0)
        self.assertLessEqual(fid, 1.0 + 1e-6)

    def test_zero_noise_fidelity_near_one(self):
        """Fidelity of ideal vs ideal distribution should be close to 1."""
        ideal = _run_simulation_fn(self.orig_circ, noise_model=None, shots=2048)
        fid = _compute_fidelity_fn(ideal["counts"], ideal["counts"], self.n)
        self.assertGreater(fid, 0.95)

    # ── evaluate_noise ────────────────────────────────────────────────────

    def test_evaluate_noise_keys(self):
        """evaluate_noise returns required top-level keys."""
        result = self._evaluate_noise(
            self.orig_circ, self.d_circ, self.b_circ,
            noise_type="depolarizing", error_rate=0.01, shots=256,
        )
        for key in ["original", "dsatur", "backtracking", "noise_type", "error_rate"]:
            self.assertIn(key, result)

    def test_evaluate_noise_per_circuit_keys(self):
        """Each circuit entry in evaluate_noise result has required keys."""
        result = self._evaluate_noise(
            self.orig_circ, self.d_circ, self.b_circ,
            noise_type="depolarizing", error_rate=0.01, shots=256,
        )
        for circ_key in ["original", "dsatur", "backtracking"]:
            entry = result[circ_key]
            for field in ["ideal", "noisy", "fidelity", "depth_ideal"]:
                self.assertIn(field, entry, f"Missing '{field}' in result['{circ_key}']")

    def test_evaluate_noise_fidelity_values(self):
        """All fidelity values from evaluate_noise are in [0, 1]."""
        result = self._evaluate_noise(
            self.orig_circ, self.d_circ, self.b_circ,
            noise_type="depolarizing", error_rate=0.01, shots=256,
        )
        for key in ["original", "dsatur", "backtracking"]:
            fid = result[key]["fidelity"]
            self.assertGreaterEqual(fid, 0.0, f"{key} fidelity < 0")
            self.assertLessEqual(fid, 1.0 + 1e-6, f"{key} fidelity > 1")


# ===========================================================================
# TestExperiments — 7 explicit end-to-end experiments
# ===========================================================================

class TestExperiments(unittest.TestCase):
    """
    End-to-end experiment tests matching the project's Experiment 1–7 specification.
    These tests document ACTUAL results and verify they are scientifically correct.
    """

    @classmethod
    def setUpClass(cls):
        # Store as staticmethods so instance access does not inject `self`
        from modules.circuit_generator import get_predefined_circuits, extract_gate_info, generate_random_commuting_circuit
        from modules.conflict_graph import build_conflict_graph
        from modules.graph_coloring import dsatur, backtracking_coloring, validate_coloring
        from modules.circuit_reconstructor import reconstruct_circuit, circuits_are_equivalent
        cls.get_pred   = staticmethod(get_predefined_circuits)
        cls.extract    = staticmethod(extract_gate_info)
        cls.build_cg   = staticmethod(build_conflict_graph)
        cls.dsatur     = staticmethod(dsatur)
        cls.bt         = staticmethod(backtracking_coloring)
        cls.validate   = staticmethod(validate_coloring)
        cls.reconstruct = staticmethod(reconstruct_circuit)
        cls.equiv      = staticmethod(circuits_are_equivalent)
        cls.gen_rand   = staticmethod(generate_random_commuting_circuit)

    def _pipeline(self, circ):
        gi = self.__class__.extract(circ)
        G  = self.__class__.build_cg(gi)
        d  = self.__class__.dsatur(G)
        b  = self.__class__.bt(G)
        dc = self.__class__.reconstruct(gi, d["coloring"], circ.num_qubits)
        bc = self.__class__.reconstruct(gi, b["coloring"], circ.num_qubits)
        return gi, G, d, b, dc, bc

    def test_exp1_basic_cz_rz_circuit(self):
        """Exp 1: basic CZ/RZ circuit processes correctly end-to-end."""
        from qiskit.circuit import QuantumCircuit
        import math
        qc = QuantumCircuit(3)
        qc.cz(0, 1); qc.rz(math.pi/4, 2); qc.cz(1, 2); qc.rz(math.pi/3, 0)
        gi, G, d, b, dc, bc = self._pipeline(qc)
        self.assertEqual(len(gi), 4)
        self.assertTrue(self.validate(G, d["coloring"])[0])
        self.assertTrue(self.equiv(qc, dc)[0])

    def test_exp2_disjoint_qubits_parallelized(self):
        """Exp 2: gates on disjoint qubits receive the same color (can run in parallel)."""
        from qiskit.circuit import QuantumCircuit
        qc = QuantumCircuit(4)
        qc.cz(0, 1); qc.cz(2, 3)  # These two are disjoint → same color
        gi, G, d, b, dc, bc = self._pipeline(qc)
        coloring = d["coloring"]
        # Vertices 0 and 1 (the two gates) should have the same color
        self.assertEqual(coloring[0], coloring[1],
                         "Disjoint gates should be in the same execution layer")
        self.assertEqual(d["n_colors"], 1, "Two fully disjoint gates → 1 color")

    def test_exp3_shared_qubit_conflict(self):
        """Exp 3: gates sharing a qubit must be in different layers (conflict edge exists)."""
        from qiskit.circuit import QuantumCircuit
        qc = QuantumCircuit(3)
        qc.cz(0, 1); qc.cz(1, 2)  # Both share qubit 1 → conflict
        gi, G, d, b, dc, bc = self._pipeline(qc)
        # Edge between gate 0 and gate 1 must exist
        self.assertTrue(G.has_edge(0, 1), "Gates sharing qubit 1 must have a conflict edge")
        # They must receive different colors
        self.assertNotEqual(d["coloring"][0], d["coloring"][1],
                            "Conflicting gates must have different colors")

    def test_exp4_dsatur_vs_backtracking_coloring(self):
        """Exp 4: DSatur and Backtracking both produce valid colorings; BT is optimal.

        Uses a 4-qubit Dense circuit (16 vertices, 78 edges) where BT terminates
        within the time limit. The 5-qubit Dense circuit has 25 vertices and 170
        edges — it exceeds BT's 30-second time limit on typical hardware and is
        documented as a scalability limitation in the project report.
        """
        # Use 4-qubit Dense: 16 vertices, 78 edges — BT terminates optimally
        circuits = self.get_pred(4)
        qc = circuits["Dense"]
        gi, G, d, b, dc, bc = self._pipeline(qc)
        # Both valid
        self.assertTrue(self.validate(G, d["coloring"])[0], "DSatur coloring invalid")
        self.assertTrue(self.validate(G, b["coloring"])[0], "BT coloring invalid")
        # Backtracking must find the optimum (not timed out)
        self.assertFalse(b["timed_out"], "BT timed out on 4-qubit Dense — unexpected")
        self.assertTrue(b["optimal"], "BT should report optimal when it does not time out")
        # BT colors <= DSatur colors (BT finds minimum)
        self.assertLessEqual(b["n_colors"], d["n_colors"])

    def test_exp5_circuit_equivalence(self):
        """Exp 5: Optimized circuits implement the same unitary as the original."""
        for name in ["Linear Chain", "Mixed", "Dense"]:
            with self.subTest(circuit=name):
                qc = self.get_pred(4)[name]
                _, _, d, b, dc, bc = self._pipeline(qc)
                eq_d, diff_d = self.equiv(qc, dc)
                eq_b, diff_b = self.equiv(qc, bc)
                self.assertTrue(eq_d, f"[{name}] DSatur circuit not equivalent (diff={diff_d:.2e})")
                self.assertTrue(eq_b, f"[{name}] BT circuit not equivalent (diff={diff_b:.2e})")

    def test_exp6_noise_evaluation_runs(self):
        """Exp 6: Noise simulation completes; fidelity is in [0,1]."""
        from modules.noise_evaluator import evaluate_noise
        qc = self.get_pred(3)["Mixed"]
        _, _, d, b, dc, bc = self._pipeline(qc)
        result = evaluate_noise(qc, dc, bc, noise_type="depolarizing",
                                error_rate=0.02, shots=256)
        for key in ["original", "dsatur", "backtracking"]:
            fid = result[key]["fidelity"]
            self.assertGreaterEqual(fid, 0.0)
            self.assertLessEqual(fid, 1.0 + 1e-6)

    def test_exp7_no_improvement_case(self):
        """Exp 7: Linear chain gives depth = #colors; app reports honestly (no fabrication)."""
        qc = self.get_pred(4)["Linear Chain"]
        gi, G, d, b, dc, bc = self._pipeline(qc)
        # The original depth and optimized depth are both reported truthfully
        orig_d = qc.depth()
        opt_d = bc.depth()
        # We do NOT assert opt_d < orig_d — that would be fabricating improvement.
        # We only assert no extra depth was added.
        self.assertLessEqual(opt_d, orig_d,
                             "Optimized depth should never exceed original")
        # Gate count preserved
        self.assertEqual(dc.size(), len(gi))
        self.assertEqual(bc.size(), len(gi))


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

