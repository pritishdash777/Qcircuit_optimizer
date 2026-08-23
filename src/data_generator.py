# src/data_generator.py

from qiskit import QuantumCircuit
import random

from .circuit_utils import circuit_to_features
from .optimizer import remove_redundant_gates


def random_circuit():
    """
    Generate circuits containing both normal gates
    and deliberately redundant gate pairs.
    """

    qc = QuantumCircuit(2)

    gates = ["x", "h", "z"]

    for _ in range(random.randint(5, 12)):

        gate = random.choice(gates)
        qubit = random.randint(0, 1)

        # 50% chance of deliberately creating
        # a cancelling pair
        make_redundant_pair = random.random() < 0.5

        if gate == "x":
            qc.x(qubit)

            if make_redundant_pair:
                qc.x(qubit)

        elif gate == "h":
            qc.h(qubit)

            if make_redundant_pair:
                qc.h(qubit)

        elif gate == "z":
            qc.z(qubit)

            if make_redundant_pair:
                qc.z(qubit)

    return qc


def create_dataset(samples=500):

    X = []
    y = []

    for _ in range(samples):

        circuit = random_circuit()

        original = circuit.size()

        optimized = remove_redundant_gates(circuit)

        reduced = original - optimized.size()

        X.append(
            circuit_to_features(circuit)
        )

        y.append(reduced)

    return X, y