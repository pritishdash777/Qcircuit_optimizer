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
# 3. SAFETY VALIDATION
# ============================================================

def validate_optimization(
    original_circuit,
    candidate_circuit,
    fidelity_threshold=0.99
):
    """
    Check whether an optimized candidate circuit
    preserves the simulated quantum state.

    The ML model proposes an optimization.

    This function decides whether we trust it.

    Returns:

    safe_circuit
    fidelity
    accepted

    accepted = True
        ML optimization passed validation.

    accepted = False
        ML optimization failed validation,
        so the original circuit is returned.
    """

    # --------------------------------------------------------
    # Compare original and candidate
    # --------------------------------------------------------

    try:

        fidelity = compare_circuits(
            original_circuit,
            candidate_circuit
        )

        fidelity = float(
            fidelity
        )

    except Exception as error:

        print(
            "\nSafety validation failed:"
        )

        print(error)

        # If validation itself fails,
        # reject optimization.
        return (
            original_circuit,
            0.0,
            False
        )

    # --------------------------------------------------------
    # Accept candidate only if fidelity is high enough
    # --------------------------------------------------------

    if fidelity >= fidelity_threshold:

        return (
            candidate_circuit,
            fidelity,
            True
        )

    # --------------------------------------------------------
    # Otherwise reject ML optimization
    # --------------------------------------------------------

    return (
        original_circuit,
        fidelity,
        False
    )


# ============================================================
# 4. PRINT SAFETY REPORT
# ============================================================

def print_safety_report(
    original_circuit,
    candidate_circuit,
    safe_circuit,
    fidelity,
    accepted
):
    """
    Print a human-readable safety report.
    """

    original_gates = len(
        original_circuit.data
    )

    candidate_gates = len(
        candidate_circuit.data
    )

    final_gates = len(
        safe_circuit.data
    )

    proposed_reduction = (
        original_gates
        - candidate_gates
    )

    final_reduction = (
        original_gates
        - final_gates
    )

    print("\n====================================")
    print("ML OPTIMIZATION SAFETY CHECK")
    print("====================================")

    print(
        "Original gates:",
        original_gates
    )

    print(
        "ML proposed gates:",
        candidate_gates
    )

    print(
        "ML proposed reduction:",
        proposed_reduction
    )

    print(
        f"Measured fidelity: "
        f"{fidelity:.4f}"
    )

    if accepted:

        print(
            "Safety result: ACCEPTED"
        )

        print(
            "ML optimization preserved "
            "the required state fidelity."
        )

    else:

        print(
            "Safety result: REJECTED"
        )

        print(
            "Original circuit restored."
        )

    print(
        "Final circuit gates:",
        final_gates
    )

    print(
        "Final accepted reduction:",
        final_reduction
    )

    print("====================================")