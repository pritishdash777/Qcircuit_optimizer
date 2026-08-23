# ============================================================
# QUANTUM CIRCUIT BENCHMARKS
#
# Provides standard/demo quantum circuits that can be used
# to test the ML optimizer from the frontend.
# ============================================================

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
import random


# ============================================================
# 1. BELL STATE
# ============================================================

def create_bell_circuit():
    """
    Create a 2-qubit Bell-state circuit.

    |00>
      ↓
    H on q0
      ↓
    CX q0 -> q1

    Produces an entangled Bell state.
    """

    qc = QuantumCircuit(2, name="Bell")

    qc.h(0)
    qc.cx(0, 1)

    return qc


# ============================================================
# 2. MESSY BELL STATE
# ============================================================

def create_messy_bell_circuit():
    """
    Bell-state circuit with deliberately redundant gates.

    Useful for demonstrating optimization.
    """

    qc = QuantumCircuit(2, name="Messy Bell")

    # Redundant pair
    qc.x(0)
    qc.x(0)

    # Actual Bell preparation
    qc.h(0)
    qc.cx(0, 1)

    # Another redundant pair
    qc.z(1)
    qc.z(1)

    return qc


# ============================================================
# 3. GHZ STATE
# ============================================================

def create_ghz_circuit(num_qubits=3):
    """
    Create a GHZ-state circuit.

    Example for 3 qubits:

    H(q0)
       ↓
    CX(q0,q1)
       ↓
    CX(q1,q2)
    """

    if num_qubits < 2:
        raise ValueError(
            "GHZ circuit requires at least 2 qubits."
        )

    qc = QuantumCircuit(
        num_qubits,
        name=f"GHZ-{num_qubits}"
    )

    qc.h(0)

    for qubit in range(
        num_qubits - 1
    ):

        qc.cx(
            qubit,
            qubit + 1
        )

    return qc


# ============================================================
# 4. MESSY GHZ CIRCUIT
# ============================================================

def create_messy_ghz_circuit(
    num_qubits=3
):
    """
    GHZ circuit with some deliberately
    redundant self-cancelling gate pairs.
    """

    if num_qubits < 2:
        raise ValueError(
            "GHZ circuit requires at least 2 qubits."
        )

    qc = QuantumCircuit(
        num_qubits,
        name=f"Messy GHZ-{num_qubits}"
    )

    # Redundant pair
    qc.x(0)
    qc.x(0)

    # GHZ preparation
    qc.h(0)

    for qubit in range(
        num_qubits - 1
    ):

        qc.cx(
            qubit,
            qubit + 1
        )

    # Redundant pair
    qc.z(
        num_qubits - 1
    )

    qc.z(
        num_qubits - 1
    )

    return qc


# ============================================================
# 5. QFT CIRCUIT
# ============================================================

def create_qft_circuit(
    num_qubits=3
):
    """
    Create a Quantum Fourier Transform circuit.

    This is useful as a more structured
    benchmark than random X/H/Z circuits.
    """

    if num_qubits < 1:
        raise ValueError(
            "QFT requires at least 1 qubit."
        )

    qft = QFT(
        num_qubits=num_qubits,
        do_swaps=True,
        approximation_degree=0
    )

    qc = QuantumCircuit(
        num_qubits,
        name=f"QFT-{num_qubits}"
    )

    qc.compose(
        qft,
        inplace=True
    )

    return qc


# ============================================================
# 6. RANDOM BENCHMARK
# ============================================================

def create_random_benchmark(
    num_qubits=2,
    depth=10,
    redundancy_probability=0.35
):
    """
    Create a random benchmark circuit.

    Supported gates:

    X
    H
    Z
    CX

    Some gates are deliberately duplicated
    so the ML optimizer has optimization
    opportunities.
    """

    if num_qubits < 1:
        raise ValueError(
            "Number of qubits must be at least 1."
        )

    if depth < 1:
        raise ValueError(
            "Depth must be at least 1."
        )

    qc = QuantumCircuit(
        num_qubits,
        name="Random Benchmark"
    )

    single_gates = [
        "x",
        "h",
        "z"
    ]

    for _ in range(depth):

        # Occasionally add a CX gate
        if (
            num_qubits >= 2
            and random.random() < 0.20
        ):

            control = random.randrange(
                num_qubits
            )

            target = random.randrange(
                num_qubits
            )

            while target == control:

                target = random.randrange(
                    num_qubits
                )

            qc.cx(
                control,
                target
            )

            # Sometimes add duplicate CX
            if (
                random.random()
                < redundancy_probability
            ):

                qc.cx(
                    control,
                    target
                )

        else:

            gate = random.choice(
                single_gates
            )

            qubit = random.randrange(
                num_qubits
            )

            if gate == "x":

                qc.x(qubit)

                if (
                    random.random()
                    < redundancy_probability
                ):
                    qc.x(qubit)

            elif gate == "h":

                qc.h(qubit)

                if (
                    random.random()
                    < redundancy_probability
                ):
                    qc.h(qubit)

            elif gate == "z":

                qc.z(qubit)

                if (
                    random.random()
                    < redundancy_probability
                ):
                    qc.z(qubit)

    return qc


# ============================================================
# 7. BENCHMARK REGISTRY
# ============================================================

def get_benchmark_names():
    """
    Return benchmark options for the frontend.
    """

    return [
        "Messy Bell State",
        "Bell State",
        "Messy GHZ State",
        "GHZ State",
        "QFT",
        "Random Benchmark"
    ]


# ============================================================
# 8. BENCHMARK FACTORY
# ============================================================

def create_benchmark(
    benchmark_name,
    num_qubits=3
):
    """
    Create a benchmark circuit by name.

    This gives the frontend one simple
    function to call.
    """

    if benchmark_name == "Messy Bell State":

        return create_messy_bell_circuit()

    elif benchmark_name == "Bell State":

        return create_bell_circuit()

    elif benchmark_name == "Messy GHZ State":

        return create_messy_ghz_circuit(
            num_qubits=max(
                2,
                num_qubits
            )
        )

    elif benchmark_name == "GHZ State":

        return create_ghz_circuit(
            num_qubits=max(
                2,
                num_qubits
            )
        )

    elif benchmark_name == "QFT":

        return create_qft_circuit(
            num_qubits=max(
                1,
                num_qubits
            )
        )

    elif benchmark_name == "Random Benchmark":

        return create_random_benchmark(
            num_qubits=max(
                1,
                num_qubits
            ),
            depth=10
        )

    else:

        raise ValueError(
            f"Unknown benchmark: "
            f"{benchmark_name}"
        )