# ============================================================
# QUANTUM CIRCUIT OPTIMIZER
#
# Contains:
# 1. Rule-based optimization
# 2. Redundancy-mask generation
# 3. Safety validation for ML optimizations
# ============================================================

from qiskit import QuantumCircuit

from .circuit_utils import compare_circuits

from qiskit import QuantumCircuit

from .circuit_utils import (
    compare_circuits,
    check_circuit_equivalence
)

# ============================================================
# 1. RULE-BASED OPTIMIZER
# ============================================================

def remove_redundant_gates(
    qc: QuantumCircuit
) -> QuantumCircuit:
    """
    Remove simple neighbouring gate pairs
    that cancel each other.

    Currently handles:

    X-X
    H-H
    Y-Y
    Z-Z
    CX-CX

    Example:

    X X H Z Z

    becomes:

    H
    """

    # Create an empty circuit with the same
    # registers/qubits/classical bits.
    new_qc = qc.copy_empty_like()

    # Gates that are self-inverse:
    #
    # applying them twice gives identity.
    cancellable_gates = {
        "x",
        "h",
        "y",
        "z",
        "cx"
    }

    # Store gates that survive optimization
    kept_gates = []

    # --------------------------------------------------------
    # Examine each gate
    # --------------------------------------------------------

    for instruction in qc.data:

        gate_name = (
            instruction
            .operation
            .name
        )

        qubits = instruction.qubits

        # ----------------------------------------------------
        # Compare with previous gate
        # ----------------------------------------------------

        if kept_gates:

            last_instruction = (
                kept_gates[-1]
            )

            last_name = (
                last_instruction
                .operation
                .name
            )

            last_qubits = (
                last_instruction.qubits
            )

            # ------------------------------------------------
            # Two neighbouring identical self-inverse gates
            # on the same qubits cancel.
            # ------------------------------------------------

            if (
                gate_name == last_name
                and qubits == last_qubits
                and gate_name in cancellable_gates
            ):

                # Remove previous gate
                kept_gates.pop()

                # Do not add current gate either
                continue

        # Otherwise keep the gate
        kept_gates.append(
            instruction
        )

    # --------------------------------------------------------
    # Rebuild optimized circuit
    # --------------------------------------------------------

    for instruction in kept_gates:

        new_qubits = []

        for qubit in instruction.qubits:

            qubit_index = (
                qc
                .find_bit(qubit)
                .index
            )

            new_qubits.append(
                new_qc.qubits[
                    qubit_index
                ]
            )

        new_clbits = []

        for clbit in instruction.clbits:

            clbit_index = (
                qc
                .find_bit(clbit)
                .index
            )

            new_clbits.append(
                new_qc.clbits[
                    clbit_index
                ]
            )

        new_qc.append(
            instruction.operation,
            new_qubits,
            new_clbits
        )

    return new_qc


# ============================================================
# 2. REDUNDANCY MASK GENERATOR
# ============================================================

def get_redundancy_mask(qc):
    """
    Generate the correct gate-removal labels
    used to train the ML model.

    1 = REMOVE
    0 = KEEP

    Example:

    Circuit:

    X X H Z Z

    Mask:

    1 1 0 1 1
    """

    mask = [
        0
        for _ in range(
            len(qc.data)
        )
    ]

    # Stores:
    #
    # (gate_index, instruction)
    #
    # Think of this as remembering
    # the previous surviving gate.
    stack = []

    cancellable_gates = {
        "x",
        "h",
        "y",
        "z",
        "cx"
    }

    # --------------------------------------------------------
    # Examine circuit
    # --------------------------------------------------------

    for index, instruction in enumerate(
        qc.data
    ):

        gate_name = (
            instruction
            .operation
            .name
        )

        qubits = (
            instruction.qubits
        )

        # ----------------------------------------------------
        # Check previous surviving gate
        # ----------------------------------------------------

        if stack:

            (
                previous_index,
                previous_instruction
            ) = stack[-1]

            previous_name = (
                previous_instruction
                .operation
                .name
            )

            previous_qubits = (
                previous_instruction
                .qubits
            )

            # ------------------------------------------------
            # Same gate
            # +
            # same qubits
            # +
            # self-inverse gate
            #
            # → both gates can be removed
            # ------------------------------------------------

            if (
                gate_name == previous_name
                and qubits == previous_qubits
                and gate_name in cancellable_gates
            ):

                # Mark BOTH gates for removal
                mask[
                    previous_index
                ] = 1

                mask[
                    index
                ] = 1

                # Remove previous gate from stack
                stack.pop()

                continue

        # Keep this gate available for comparison
        stack.append(
            (
                index,
                instruction
            )
        )

    return mask

# ============================================================
# RULE-BASED OPTIMIZER & SAFETY VALIDATION LAYER
# ============================================================



def get_redundancy_mask(circuit):
    """
    Generate rule-based redundancy mask for a quantum circuit.
    Finds adjacent self-inverse gate pairs on the same qubit(s).
    """
    mask = [0] * len(circuit.data)
    i = 0
    while i < len(circuit.data) - 1:
        current_inst = circuit.data[i]
        next_inst = circuit.data[i + 1]

        current_op = current_inst.operation.name
        next_op = next_inst.operation.name

        current_qubits = [circuit.find_bit(q).index for q in current_inst.qubits]
        next_qubits = [circuit.find_bit(q).index for q in next_inst.qubits]

        # Check self-inverse adjacent single/two-qubit gates
        if (current_op == next_op) and (current_qubits == next_qubits):
            if current_op in ["x", "y", "z", "h", "cx"]:
                mask[i] = 1
                mask[i + 1] = 1
                i += 2
                continue
        i += 1

    return mask

def validate_optimization(
    original_circuit: QuantumCircuit,
    candidate_circuit: QuantumCircuit,
    fidelity_threshold: float = 0.99
):
    """
    Validate an ML-proposed optimization using exact
    operator equivalence up to global phase.

    IMPORTANT:
    `fidelity_threshold` is retained temporarily for
    backward compatibility with existing call sites.
    It is NOT used as the safety criterion.

    Safety policy:
    - Equivalent -> accept candidate.
    - Not equivalent -> reject candidate and restore original.
    - Verification failure -> reject candidate.

    Returns:
        final_circuit,
        equivalence_metric,
        accepted
    """

    result = check_circuit_equivalence(
        original_circuit,
        candidate_circuit
    )

    accepted = bool(
        result["equivalent"]
    )

    if accepted:

        final_circuit = candidate_circuit
        equivalence_metric = 1.0

    else:

        final_circuit = original_circuit
        equivalence_metric = 0.0

    return (
        final_circuit,
        equivalence_metric,
        accepted
    )


def print_safety_report(
    original_circuit,
    candidate_circuit,
    final_circuit,
    equivalence_metric,
    accepted
):
    """
    Print a formatted safety report for circuit optimization.
    """
    print("\n------------------------------------")
    print("SAFETY VERIFICATION REPORT")
    print("------------------------------------")
    print(f"Original Gate Count  : {len(original_circuit.data)}")
    print(f"Candidate Gate Count : {len(candidate_circuit.data)}")
    print(f"Final Gate Count     : {len(final_circuit.data)}")
    print(f"Operator Equivalence : {'VERIFIED (1.0)' if accepted else 'FAILED (0.0)'}")
    print(f"Optimization Status  : {'ACCEPTED' if accepted else 'REJECTED (Original Restored)'}")
    print("------------------------------------\n")