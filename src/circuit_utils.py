from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
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
    """Simulate the circuit and return the final state"""
    return Statevector.from_instruction(qc)


def compare_circuits(qc1, qc2):
    """Check how similar two circuits are (fidelity)"""
    state1 = get_statevector(qc1)
    state2 = get_statevector(qc2)
    fidelity = np.abs(np.vdot(state1, state2))**2
    return fidelity
def circuit_to_features(circuit):

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