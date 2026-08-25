# ============================================================
# QUANTUM CIRCUIT UTILITIES & EQUIVALENCE SAFETY LAYER
# ============================================================

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
import numpy as np


def create_messy_circuit():
    """Create a simple messy quantum circuit with redundant gates"""
    qc = QuantumCircuit(1, name="messy")
    
    qc.x(0)   # Flip
    qc.x(0)   # Flip again (these two cancel)
    qc.h(0)   # Hadamard
    qc.x(0)   # Flip
    qc.x(0)   # Flip again (these two also cancel)
    
    return qc


def create_clean_circuit():
    """The optimized version of the messy circuit"""
    qc = QuantumCircuit(1, name="clean")
    qc.h(0)
    return qc


def get_circuit_info(qc):
    """Return basic information about a circuit"""
    depth = qc.depth()
    gates = len(qc.data)
    return depth, gates


def get_statevector(qc):
    """Simulate the circuit and return the final statevector from |0...0>."""
    return Statevector.from_instruction(qc)


def compare_circuits(qc1, qc2):
    """
    Check statevector fidelity between two circuits starting from |0...0>.

    NOTE: Statevector fidelity measured from |0...0> is useful for basic 
    state diagnostics, but is NOT mathematically sufficient to prove overall 
    circuit/operator equivalence.
    """
    try:
        state1 = get_statevector(qc1)
        state2 = get_statevector(qc2)
        fidelity = np.abs(np.vdot(state1, state2))**2
        return float(fidelity)
    except Exception:
        return 0.0


def circuit_to_features(circuit):
    """Extract summary numerical features from a quantum circuit."""
    counts = circuit.count_ops()

    num_x = counts.get("x", 0)
    num_h = counts.get("h", 0)
    num_z = counts.get("z", 0)
    num_cx = counts.get("cx", 0)

    depth = circuit.depth()
    gate_count = sum(counts.values())

    return [
        num_x,
        num_h,
        num_z,
        num_cx,
        depth,
        gate_count
    ]


# ============================================================
# EXACT OPERATOR EQUIVALENCE CHECK (FULL QUANTUM SAFETY LAYER)
# ============================================================

def check_circuit_equivalence(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    tolerance: float = 1e-7
) -> dict:
    """
    Verify full quantum circuit equivalence up to global phase.

    This function compares the full unitary matrices U1 and U2 representing 
    the two circuits rather than merely testing state outputs from |0...0>.

    SCALABILITY WARNING:
    Exact unitary calculation scales exponentially with the number of qubits (2^N x 2^N).
    This method is used as an exact safety check for small circuits (<= 4 qubits)
    and will not scale to large quantum algorithms.

    Parameters:
        circuit1: Original QuantumCircuit.
        circuit2: Optimized/Candidate QuantumCircuit.
        tolerance: Numerical tolerance for matrix equality.

    Returns:
        dict containing:
            - "equivalent": bool
            - "reason": str
            - "equivalence_error": float
            - "global_phase_angle": float or None
    """
    # 1. Qubit Count Check
    if circuit1.num_qubits != circuit2.num_qubits:
        return {
            "equivalent": False,
            "reason": f"Qubit mismatch: circuit1 has {circuit1.num_qubits} qubits, circuit2 has {circuit2.num_qubits} qubits.",
            "equivalence_error": 1.0,
            "global_phase_angle": None
        }

    # 2. Non-unitary instruction check (e.g., measurements, resets)
    for qc, name in [(circuit1, "Original"), (circuit2, "Candidate")]:
        for instruction in qc.data:
            op_name = instruction.operation.name.lower()
            if op_name in ["measure", "reset", "barrier"]:
                return {
                    "equivalent": False,
                    "reason": f"Non-unitary or control instruction '{op_name}' present in {name} circuit.",
                    "equivalence_error": 1.0,
                    "global_phase_angle": None
                }

    # 3. Obtain Operator Matrices
    try:
        U1 = Operator(circuit1).data
        U2 = Operator(circuit2).data
    except Exception as error:
        return {
            "equivalent": False,
            "reason": f"Could not construct Operator matrix: {error}",
            "equivalence_error": 1.0,
            "global_phase_angle": None
        }

    # 4. Compare Matrices up to Global Phase
    # Find the first entry in U1 with magnitude significantly above zero to extract global phase factor.
    non_zero_indices = np.where(np.abs(U1) > 1e-5)
    if len(non_zero_indices[0]) == 0:
        return {
            "equivalent": False,
            "reason": "Operator matrix U1 is zero or invalid.",
            "equivalence_error": 1.0,
            "global_phase_angle": None
        }

    idx = (non_zero_indices[0][0], non_zero_indices[1][0])
    u1_elem = U1[idx]
    u2_elem = U2[idx]

    if np.abs(u2_elem) < 1e-5:
        return {
            "equivalent": False,
            "reason": f"Matrix mismatch at element {idx}: U1 is non-zero but U2 is zero.",
            "equivalence_error": 1.0,
            "global_phase_angle": None
        }

    # Phase factor phase_factor = e^(i * phi) such that U2 ≈ phase_factor * U1
    phase_factor = u2_elem / u1_elem
    # Normalize to pure unit complex number |e^(i*phi)| = 1
    phase_factor /= np.abs(phase_factor)

    # Re-aligned matrix check: U2 - exp(i * phi) * U1
    diff_matrix = U2 - (phase_factor * U1)
    max_diff = float(np.max(np.abs(diff_matrix)))

    is_equivalent = max_diff <= tolerance
    phase_angle = float(np.angle(phase_factor))

    if is_equivalent:
        reason = "Circuits are mathematically equivalent up to global phase."
    else:
        reason = f"Unitary operators differ. Maximum entrywise discrepancy: {max_diff:.8e}"

    return {
        "equivalent": is_equivalent,
        "reason": reason,
        "equivalence_error": max_diff,
        "global_phase_angle": phase_angle
    }