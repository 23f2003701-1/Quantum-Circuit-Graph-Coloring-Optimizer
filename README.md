# Quantum Circuit Optimization by Graph Coloring

> **Based on:** "Quantum Circuit Optimization by Graph Coloring"  
> Lee, Hochang; Jeong, Kyung Chul; Kim, Panjin — [arXiv:2501.14447](https://arxiv.org/abs/2501.14447) | [Quantum Journal, 2026](https://quantum-journal.org/papers/q-2026-02-06-1996/)

A Python implementation that optimizes commuting quantum circuits by converting the depth-minimization problem into a graph vertex-coloring problem, plus an additional **noise-aware evaluation module** that investigates the effect of depth reduction on noisy quantum simulations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Mathematical Background](#mathematical-background)
4. [Module Descriptions](#module-descriptions)
5. [Algorithm Details](#algorithm-details)
6. [Usage Examples](#usage-examples)
7. [Experimental Results](#experimental-results)
8. [Running Tests](#running-tests)
9. [Research Extension: Noise Evaluation](#research-extension-noise-evaluation)
10. [Limitations](#limitations)
11. [References](#references)

---

## Quick Start

### Prerequisites

```
Python >= 3.10
```

### Install dependencies

```bash
pip install qiskit==2.4.1 qiskit-aer==0.17.2 networkx matplotlib numpy streamlit pandas
```

### Launch the dashboard

```bash
cd "d:\VJTI sem stuffs\sem7\QC\project"
streamlit run app.py
```

### Run the test suite

```bash
python -m pytest tests/test_all.py -v
```

---

## Project Structure

```
project/
├── modules/
│   ├── __init__.py
│   ├── circuit_generator.py       # Module 1: Circuit generation & validation
│   ├── conflict_graph.py          # Module 2: Conflict graph construction (Algorithm 1)
│   ├── graph_coloring.py          # Module 3: DSatur + Backtracking algorithms
│   ├── circuit_reconstructor.py   # Module 4: Optimized circuit reconstruction (Algorithm 2)
│   ├── performance_evaluator.py   # Module 5: Metrics, comparison, visualization
│   └── noise_evaluator.py         # Module 6: Noise-aware simulation (research extension)
├── tests/
│   └── test_all.py                # 45 unit + integration tests
├── app.py                         # Streamlit 6-tab interactive dashboard
├── README.md                      # This file
├── PROJECT_AUDIT.md               # Audit report with verified experimental results
└── SCREENSHOT_GUIDE.md            # Guide for capturing demo screenshots
```

---

## Mathematical Background

### Why These Gates Commute

The supported gate set — **CZ, CP, RZ, Z, S, T, Sdg, Tdg** — are all **diagonal** in the computational basis:

$$D|x\rangle = e^{i\phi(x)}|x\rangle$$

Any two diagonal matrices commute because diagonal multiplication is component-wise:

$$(D_1 D_2)_{xx} = D_{1,xx} \cdot D_{2,xx} = D_{2,xx} \cdot D_{1,xx} = (D_2 D_1)_{xx}$$

Therefore the **order** of these gates does not affect the final unitary, and we can freely reorder them to minimize circuit depth.

### Conflict Graph (Algorithm 1)

Given a commuting circuit with gates $g_0, g_1, \ldots, g_{n-1}$, the **conflict graph** $G = (V, E)$ is:

- **Vertices:** $V = \{g_0, g_1, \ldots, g_{n-1}\}$ — one per gate
- **Edges:** $(g_i, g_j) \in E$ iff $Q_i \cap Q_j \neq \emptyset$ (they share at least one qubit)

An edge means the two gates **cannot execute simultaneously** (even though they commute).

### Graph Coloring → Circuit Depth

A proper **vertex coloring** assigns each gate a color (integer) such that no two adjacent gates share a color. Gates with the same color form a **parallel execution layer**.

$$\text{optimized depth} \leq \chi(G)$$

where $\chi(G)$ is the chromatic number of the conflict graph.

### Optimized Circuit (Algorithm 2)

Group gates by their color → each color class = one barrier-separated layer in the reconstructed circuit.

---

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `circuit_generator.py` | Generates predefined (Linear Chain, Full Parallel, Mixed, Dense) and random commuting circuits. Validates gate commutativity. |
| `conflict_graph.py` | Implements Algorithm 1. Builds NetworkX conflict graph, computes stats, visualizes with optional coloring overlay. |
| `graph_coloring.py` | Implements DSatur (heuristic) and Backtracking (exact) coloring from scratch. Includes validation and comparison. |
| `circuit_reconstructor.py` | Implements Algorithm 2. Reconstructs optimized circuits from colorings. Tests unitary equivalence via `Operator`. |
| `performance_evaluator.py` | Collects depth, gate count, timing, equivalence. Produces comparison tables and matplotlib figures. |
| `noise_evaluator.py` | Builds Qiskit Aer noise models. Runs ideal and noisy simulations. Computes fidelity estimates. |

---

## Algorithm Details

### DSatur (Degree of Saturation)

```
1. All vertices start uncolored.
2. Loop until all colored:
   a. Find uncolored vertex v with highest saturation
      (number of distinct colors used by its colored neighbors).
   b. Break ties: highest degree → lowest node id.
   c. Assign the smallest color not used by any neighbor of v.
   d. Update saturation of v's uncolored neighbors.
```

- **Time complexity:** O(n²)
- **Property:** Heuristic — not guaranteed to find the chromatic number, but often close in practice.

### Backtracking (Exact)

```
1. Run DSatur to get upper bound k_max.
2. For k = 1, 2, ..., k_max:
   a. Try to color the graph with exactly k colors (backtracking with forward checking).
   b. If successful, this k is the chromatic number χ(G). Stop.
3. Time limit (default 30s) prevents runaway on large instances.
```

- **Property:** Finds the **chromatic number** (minimum possible depth) for small-to-medium graphs.
- **Forward checking:** Before assigning color c to vertex v, verify every uncolored neighbor still has at least one valid color.

---

## Usage Examples

### Full pipeline (programmatic)

```python
import sys; sys.path.insert(0, '.')

from modules.circuit_generator import get_predefined_circuits, extract_gate_info
from modules.conflict_graph import build_conflict_graph, get_graph_stats
from modules.graph_coloring import dsatur, backtracking_coloring, validate_coloring
from modules.circuit_reconstructor import reconstruct_circuit, circuits_are_equivalent
from modules.performance_evaluator import evaluate_all, format_comparison_table

# 1. Create circuit
circuits = get_predefined_circuits(4)
circ = circuits["Dense"]
print(f"Original depth: {circ.depth()}, gates: {len(circ.data)}")

# 2. Build conflict graph
gate_info = extract_gate_info(circ)
G = build_conflict_graph(gate_info)
print("Graph:", get_graph_stats(G))

# 3. Graph coloring
d = dsatur(G)
b = backtracking_coloring(G)
print(f"DSatur: {d['n_colors']} colors, Backtracking: {b['n_colors']} colors (optimal={b['optimal']})")

# 4. Reconstruct optimized circuits
d_circ = reconstruct_circuit(gate_info, d['coloring'], circ.num_qubits)
b_circ = reconstruct_circuit(gate_info, b['coloring'], circ.num_qubits)
print(f"DSatur depth: {d_circ.depth()}, BT depth: {b_circ.depth()}")

# 5. Verify equivalence
eq_d, diff_d = circuits_are_equivalent(circ, d_circ)
eq_b, diff_b = circuits_are_equivalent(circ, b_circ)
print(f"Equivalent: DSatur={eq_d} (diff={diff_d:.2e}), BT={eq_b} (diff={diff_b:.2e})")

# 6. Performance evaluation
result = evaluate_all(
    circ, d_circ, b_circ,
    {'n_colors': d['n_colors'], 'coloring_time': d['time_seconds']},
    {'n_colors': b['n_colors'], 'coloring_time': b['time_seconds']},
    eq_d, eq_b
)
print(format_comparison_table(result))
```

### Noise evaluation

```python
from modules.noise_evaluator import evaluate_noise

results = evaluate_noise(
    original_circuit, dsatur_circuit, backtracking_circuit,
    noise_type="depolarizing",
    error_rate=0.01,
    shots=2048,
)
for key in ["original", "dsatur", "backtracking"]:
    print(f"{key}: depth={results[key]['depth_ideal']}, fidelity={results[key]['fidelity']:.4f}")
```

---

## Experimental Results

Results from actual runs (4-qubit predefined circuits, Python 3.12.2):

| Circuit | Orig Depth | DSatur Depth | BT Depth | Colors | BT Optimal | Equivalent |
|---------|-----------|--------------|----------|--------|-----------|-----------|
| Linear Chain | 6 | 5 | **5** | 5 | ✅ | ✅ |
| Full Parallel | 2 | 2 | **2** | 2 | ✅ | ✅ |
| Mixed | 6 | 5 | **5** | 5 | ✅ | ✅ |
| Dense | 11 | 7 | **7** | 7 | ✅ | ✅ |

**Key observations:**
- Gate count is preserved in all cases (optimization only reorders).
- DSatur and Backtracking agree on the chromatic number for all tested circuits.
- Unitary equivalence verified for all circuits (max diff < 1e-14).

---

## Running Tests

```bash
# Full suite (45 tests)
python -m pytest tests/test_all.py -v

# With HTML report
python -m pytest tests/test_all.py -v --html=test_report.html
```

Expected output:
```
45 passed, 14 subtests passed in 6.44s
```

---

## Research Extension: Noise Evaluation

This module goes beyond the base paper by investigating whether depth reduction improves noise robustness.

### Supported noise models

| Model | Description |
|-------|-------------|
| `depolarizing` | Uniform depolarizing channel on each gate |
| `amplitude_damping` | Models energy relaxation (T1 decay) |
| `combined` | Depolarizing + amplitude damping |

### Fidelity estimation

Fidelity is estimated via the Bhattacharyya coefficient between ideal and noisy measurement distributions:

$$F \approx \left(\sum_x \sqrt{p_{\text{ideal}}(x)\, p_{\text{noisy}}(x)}\right)^2$$

### Research finding

Depth reduction improves noise robustness **when noise is primarily depth-limited** (i.e., each gate layer introduces fixed error). For small circuits where depth reduction is minimal, fidelity differences fall within shot noise. Use the Dense or Mixed circuits with high qubit counts for the most visible effect.

---

## Limitations

1. **Commuting circuits only:** The optimization applies exclusively to circuits composed of diagonal gates. CNOT, H, X, and other non-diagonal gates are explicitly rejected.
2. **Backtracking complexity:** Exact coloring is NP-complete; the 30s time limit may be hit for dense graphs with many gates.
3. **Local simulation:** Statevector and density_matrix simulators are limited by RAM. 8+ qubit noisy simulations may be slow.
4. **Fidelity approximation:** The Bhattacharyya-based fidelity is an approximation; true state fidelity requires the full density matrix.
5. **Hardware assumptions:** We assume any two disjoint-qubit gates can run in parallel. Real hardware has connectivity constraints.

---

## References

1. H. Lee, K. C. Jeong, P. Kim, "Quantum Circuit Optimization by Graph Coloring," *Quantum* **10**, 1996 (2026). [arXiv:2501.14447](https://arxiv.org/abs/2501.14447)
2. Qiskit Development Team. *Qiskit: An Open-source Framework for Quantum Computing*. (2024). https://qiskit.org
3. Qiskit Aer. *High-performance quantum circuit simulation*. https://qiskit.github.io/qiskit-aer/
4. NetworkX Developers. *NetworkX — Network Analysis in Python*. https://networkx.org
5. D. Brélaz, "New methods to color the vertices of a graph," *Communications of the ACM*, 22(4):251–256, 1979. (DSatur algorithm)
