from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction


def remove_redundant_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """
    A simple optimizer that removes pairs of gates that cancel each other.
    Currently handles: X-X, H-H, Z-Z, Y-Y, CX-CX
    """

    # Create a new empty circuit with same number of qubits
    new_qc = QuantumCircuit(qc.num_qubits, name=qc.name + "_optimized")

    # We will keep a list of gates we decide to keep
    kept_gates = []

    for instruction in qc.data:
        gate_name = instruction.operation.name
        qubits = instruction.qubits

        # Check if the last kept gate cancels with the current one
        if kept_gates:
            last_gate = kept_gates[-1]
            last_name = last_gate.operation.name
            last_qubits = last_gate.qubits

            # Same gate on the same qubit(s) → they cancel
            if (gate_name == last_name and 
                qubits == last_qubits and 
                gate_name in ["x", "h", "y", "z", "cx"]):
                
                # Remove the last gate (they cancel each other)
                kept_gates.pop()
                continue  # skip adding the current gate too

        # If it doesn't cancel, keep this gate
        kept_gates.append(instruction)

    # Add all the remaining gates to the new circuit
    for instruction in kept_gates:
        new_qc.append(instruction.operation, instruction.qubits)

    return new_qc
def get_redundancy_mask(qc):
    """
    Create a label for every gate in the circuit.

    1 = gate should be removed
    0 = gate should be kept

    Example:

    X X H H Z

    becomes:

    1 1 1 1 0
    """

    mask = [0] * len(qc.data)

    # Stack stores:
    # (gate_index, instruction)
    stack = []

    cancellable_gates = [
        "x",
        "h",
        "y",
        "z",
        "cx"
    ]

    for index, instruction in enumerate(qc.data):

        gate_name = instruction.operation.name
        qubits = instruction.qubits

        if stack:

            previous_index, previous_instruction = stack[-1]

            previous_name = previous_instruction.operation.name
            previous_qubits = previous_instruction.qubits

            # Check whether two neighbouring gates cancel
            if (
                gate_name == previous_name
                and qubits == previous_qubits
                and gate_name in cancellable_gates
            ):

                # Mark BOTH gates for removal
                mask[previous_index] = 1
                mask[index] = 1

                # Remove previous gate from stack
                stack.pop()

                continue

        stack.append(
            (index, instruction)
        )

    return mask