# Module 1: Quantum Circuit Generation
# All supported gates (CZ, CP, RZ, Z, S, T, Sdg, Tdg) are diagonal in the computational basis
# and therefore mutually commuting. This is the key mathematical property exploited by
# the graph-coloring optimization approach.

"""circuit_generator.py

Generates and validates small quantum circuits composed exclusively of
diagonal (Z-axis) gates: CZ, CP, RZ, Z, S, T, Sdg, Tdg.

Because every gate in the supported set is diagonal in the computational
basis, they all share the same set of eigenstates (the computational-basis
states) and therefore mutually commute.  This property is exploited by the
graph-coloring optimizer: any two gates that commute can be executed in
parallel without changing the unitary of the circuit.
"""

import math
import random
from collections import Counter
from typing import Optional

from qiskit.circuit import QuantumCircuit, Parameter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Gates that are diagonal in the computational basis and mutually commute.
SUPPORTED_GATES: frozenset[str] = frozenset(
    {"cz", "cp", "rz", "z", "s", "t", "sdg", "tdg"}
)

#: Discrete angle pool used by the random circuit generator for CP gates.
CP_ANGLE_POOL: list[float] = [
    math.pi / 8,
    math.pi / 4,
    math.pi / 3,
    math.pi / 2,
    2 * math.pi / 3,
    3 * math.pi / 4,
    math.pi,
]


# ---------------------------------------------------------------------------
# 1. Predefined circuits
# ---------------------------------------------------------------------------

def get_predefined_circuits(n_qubits: int) -> dict[str, QuantumCircuit]:
    """Return a dictionary of named predefined circuits for *n_qubits* qubits.

    Four circuit archetypes are provided to exercise different levels of
    gate parallelism:

    * **Linear Chain** – every gate shares a qubit with the next gate so
      no two gates can be executed simultaneously (zero parallelism).
    * **Full Parallel** – gates act on completely disjoint qubit pairs /
      single qubits so the entire circuit collapses to depth 1
      (maximum parallelism).
    * **Mixed** – alternating layers of parallel and sequential gates,
      giving moderate parallelism.
    * **Dense** – many overlapping two-qubit gates on all qubit pairs,
      creating a heavily connected dependency graph.

    All circuits use only CZ, CP, and RZ gates.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Should be in the range [3, 8].

    Returns
    -------
    dict[str, QuantumCircuit]
        Mapping of circuit name to its :class:`~qiskit.circuit.QuantumCircuit`.

    Raises
    ------
    ValueError
        If *n_qubits* is outside the supported range [3, 8].
    """
    if not (3 <= n_qubits <= 8):
        raise ValueError(
            f"n_qubits must be between 3 and 8 (inclusive), got {n_qubits}."
        )

    circuits: dict[str, QuantumCircuit] = {}

    # ------------------------------------------------------------------
    # (a) Linear Chain  – sequential, no parallelism
    # ------------------------------------------------------------------
    lc = QuantumCircuit(n_qubits, name="Linear Chain")
    # Apply CZ along the chain: (0,1), (1,2), ..., (n-2, n-1)
    for i in range(n_qubits - 1):
        lc.cz(i, i + 1)
    # Follow with RZ on each qubit in order
    for i in range(n_qubits):
        lc.rz(math.pi / 4, i)
    # Finish with CP along the chain
    for i in range(n_qubits - 1):
        lc.cp(math.pi / 3, i, i + 1)
    circuits["Linear Chain"] = lc

    # ------------------------------------------------------------------
    # (b) Full Parallel  – disjoint qubits, maximum parallelism
    # ------------------------------------------------------------------
    fp = QuantumCircuit(n_qubits, name="Full Parallel")
    # Pair up qubits: (0,1), (2,3), (4,5), …
    # Leftover qubit (when n_qubits is odd) gets an RZ.
    qubit_list = list(range(n_qubits))
    idx = 0
    while idx + 1 < len(qubit_list):
        fp.cz(qubit_list[idx], qubit_list[idx + 1])
        idx += 2
    if idx < len(qubit_list):
        fp.rz(math.pi / 2, qubit_list[idx])
    # Second layer: shift by one to get different disjoint pairs
    # (1,2), (3,4), (5,6), …; qubit 0 gets RZ
    fp.rz(math.pi / 4, 0)
    idx = 1
    while idx + 1 < n_qubits:
        fp.cp(math.pi / 4, idx, idx + 1)
        idx += 2
    if idx < n_qubits:
        fp.rz(math.pi / 3, idx)
    circuits["Full Parallel"] = fp

    # ------------------------------------------------------------------
    # (c) Mixed  – some parallel, some sequential
    # ------------------------------------------------------------------
    mx = QuantumCircuit(n_qubits, name="Mixed")
    # Layer 1: parallel CZ on even pairs
    for i in range(0, n_qubits - 1, 2):
        mx.cz(i, i + 1)
    # Layer 2: sequential CP along chain (creates dependencies)
    for i in range(n_qubits - 1):
        mx.cp(math.pi / 2, i, i + 1)
    # Layer 3: parallel RZ on all qubits
    for i in range(n_qubits):
        mx.rz(math.pi / 8 * (i + 1), i)
    # Layer 4: parallel CZ on odd pairs
    for i in range(1, n_qubits - 1, 2):
        mx.cz(i, i + 1)
    circuits["Mixed"] = mx

    # ------------------------------------------------------------------
    # (d) Dense  – many overlapping gates, heavily connected
    # ------------------------------------------------------------------
    dn = QuantumCircuit(n_qubits, name="Dense")
    # Apply CZ to every pair (i, j) with i < j (complete graph)
    angle_cycle = CP_ANGLE_POOL.copy()
    angle_idx = 0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            dn.cz(i, j)
            # Interleave CP with varying angles
            dn.cp(angle_cycle[angle_idx % len(angle_cycle)], i, j)
            angle_idx += 1
    # Add RZ rotations on each qubit to increase gate density
    for i in range(n_qubits):
        dn.rz(math.pi / (i + 2), i)
    circuits["Dense"] = dn

    return circuits


# ---------------------------------------------------------------------------
# 2. Random commuting circuit
# ---------------------------------------------------------------------------

def generate_random_commuting_circuit(
    n_qubits: int,
    n_gates: int,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """Generate a random circuit whose gates all belong to the commuting set.

    Gate types are chosen uniformly from {"cz", "cp", "rz"}.  For two-qubit
    gates (CZ, CP) two *distinct* qubits are selected at random.  For RZ a
    single qubit is selected.  CP angles are drawn from :data:`CP_ANGLE_POOL`.

    Because every generated gate is diagonal in the computational basis the
    resulting circuit is guaranteed to be a valid commuting circuit that can
    be optimized with the graph-coloring approach.

    Parameters
    ----------
    n_qubits:
        Number of qubits in the circuit.
    n_gates:
        Total number of gates to append.
    seed:
        Optional integer seed for reproducibility.

    Returns
    -------
    QuantumCircuit
        A randomly generated commuting quantum circuit.

    Raises
    ------
    ValueError
        If *n_qubits* < 2 or *n_gates* < 1.
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be >= 2 to allow two-qubit gates.")
    if n_gates < 1:
        raise ValueError("n_gates must be >= 1.")

    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits, name="Random Commuting Circuit")

    gate_choices = ["cz", "cp", "rz"]

    for _ in range(n_gates):
        gate_type = rng.choice(gate_choices)

        if gate_type == "cz":
            q0, q1 = rng.sample(range(n_qubits), 2)
            qc.cz(q0, q1)

        elif gate_type == "cp":
            q0, q1 = rng.sample(range(n_qubits), 2)
            angle = rng.choice(CP_ANGLE_POOL)
            qc.cp(angle, q0, q1)

        else:  # "rz"
            q0 = rng.randrange(n_qubits)
            angle = rng.choice(CP_ANGLE_POOL)
            qc.rz(angle, q0)

    return qc


# ---------------------------------------------------------------------------
# 3. Verification
# ---------------------------------------------------------------------------

def verify_commuting_circuit(circuit: QuantumCircuit) -> tuple[bool, str]:
    """Check that every gate in *circuit* belongs to the supported commuting set.

    Mathematical background
    -----------------------
    All gates in :data:`SUPPORTED_GATES` (CZ, CP, RZ, Z, S, T, Sdg, Tdg) are
    *diagonal* in the computational basis.  A diagonal gate acts as:

        D |x⟩ = e^{iφ(x)} |x⟩

    for some phase φ(x) that depends only on the basis state *x*.  Any two
    diagonal matrices commute because diagonal matrix multiplication is
    component-wise:

        (D₁ D₂)_{xx} = D₁_{xx} · D₂_{xx} = D₂_{xx} · D₁_{xx} = (D₂ D₁)_{xx}

    Hence the order in which these gates appear in a circuit is irrelevant to
    the final unitary, and the graph-coloring optimizer may freely reorder
    them to maximise parallelism without altering the computation.

    Parameters
    ----------
    circuit:
        The :class:`~qiskit.circuit.QuantumCircuit` to validate.

    Returns
    -------
    tuple[bool, str]
        ``(True, message)`` when every gate is supported;
        ``(False, message)`` when an unsupported gate is found, where
        *message* names the offending gate.
    """
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        if gate_name not in SUPPORTED_GATES:
            return (
                False,
                f"Error: gate '{instruction.operation.name}' is not supported. "
                f"Only the following diagonal gates are allowed: "
                f"{sorted(SUPPORTED_GATES)}.",
            )

    return (
        True,
        "Circuit is valid: all gates belong to the commuting set "
        f"{sorted(SUPPORTED_GATES)} (diagonal in the computational basis).",
    )


# ---------------------------------------------------------------------------
# 4. Gate info extraction
# ---------------------------------------------------------------------------

def extract_gate_info(circuit: QuantumCircuit) -> list[dict]:
    """Return per-gate metadata for every instruction in *circuit*.

    Parameters
    ----------
    circuit:
        The source :class:`~qiskit.circuit.QuantumCircuit`.

    Returns
    -------
    list[dict]
        One dict per gate with the keys:

        * ``"index"``  – 0-based position in the circuit's instruction list.
        * ``"name"``   – Gate name string (e.g. ``"cz"``, ``"cp"``, ``"rz"``).
        * ``"qubits"`` – List of qubit indices (integers) the gate acts on.
        * ``"params"`` – List of numeric parameter values (floats); empty for
          gates without parameters.
    """
    result: list[dict] = []
    qubit_index = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}

    for index, instruction in enumerate(circuit.data):
        gate = instruction.operation
        qubits = [qubit_index[q] for q in instruction.qubits]

        # Resolve symbolic parameters to floats where possible.
        params: list[float] = []
        for p in gate.params:
            try:
                params.append(float(p))
            except (TypeError, Exception):  # noqa: BLE001  – symbolic parameter
                params.append(p)  # keep as-is (ParameterExpression)

        result.append(
            {
                "index": index,
                "name": gate.name,
                "qubits": qubits,
                "params": params,
            }
        )

    return result


# ---------------------------------------------------------------------------
# 5. Circuit summary
# ---------------------------------------------------------------------------

def circuit_summary(circuit: QuantumCircuit) -> dict:
    """Return high-level statistics for *circuit*.

    Parameters
    ----------
    circuit:
        The :class:`~qiskit.circuit.QuantumCircuit` to summarise.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``"depth"``      – Circuit depth (critical-path length).
        * ``"gate_count"`` – Total number of gate instructions.
        * ``"n_qubits"``   – Number of qubits in the circuit.
        * ``"gate_types"`` – :class:`collections.Counter` mapping gate name
          to its occurrence count.
    """
    gate_types: Counter = Counter(
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name != "barrier"
    )

    return {
        "depth": circuit.depth(),
        "gate_count": circuit.size(),
        "n_qubits": circuit.num_qubits,
        "gate_types": gate_types,
    }


# ---------------------------------------------------------------------------
# Quick self-test (run with: python -m modules.circuit_generator)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("circuit_generator.py – self-test")
    print("=" * 60)

    N = 4

    # 1. Predefined circuits
    print(f"\n[1] Predefined circuits for n_qubits={N}:")
    predefined = get_predefined_circuits(N)
    for name, circ in predefined.items():
        s = circuit_summary(circ)
        print(
            f"  {name:15s}  depth={s['depth']:3d}  "
            f"gates={s['gate_count']:3d}  types={dict(s['gate_types'])}"
        )

    # 2. Random circuit
    print("\n[2] Random commuting circuit (n_qubits=4, n_gates=10, seed=42):")
    rand_circ = generate_random_commuting_circuit(N, 10, seed=42)
    print(rand_circ.draw(output="text"))

    # 3. Verification – valid circuit
    valid, msg = verify_commuting_circuit(rand_circ)
    print(f"\n[3] Verification (valid): {valid}  |  {msg}")

    # 4. Verification – invalid circuit (H gate)
    bad = QuantumCircuit(2, name="Bad")
    bad.h(0)
    bad.cz(0, 1)
    valid2, msg2 = verify_commuting_circuit(bad)
    print(f"\n[3] Verification (invalid): {valid2}  |  {msg2}")

    # 5. Gate info
    print("\n[4] Gate info for random circuit:")
    for entry in extract_gate_info(rand_circ):
        print(f"  {entry}")

    # 6. Summary
    print("\n[5] Summary of random circuit:")
    print(" ", circuit_summary(rand_circ))
