import numpy as np

# Supported gates
GATE_MAP = {
    "x": 1,
    "h": 2,
    "z": 3,
    "cx": 4,
    "y": 5,
    "s": 6,
    "t": 7,
    "rx": 8,
    "ry": 9,
    "rz": 10
}

MAX_GATES = 50


def encode_circuit(circuit):
    """
    Convert circuit to fixed-length vector.
    """

    vector = []

    for instruction in circuit.data:

        gate = instruction.operation.name

        vector.append(
            GATE_MAP.get(gate, 0)
        )

    while len(vector) < MAX_GATES:
        vector.append(0)

    return np.array(
        vector[:MAX_GATES],
        dtype=np.float32
    )


def decode_vector(vector):

    reverse_map = {
        v: k
        for k, v in GATE_MAP.items()
    }

    return [
        reverse_map.get(int(v), "PAD")
        for v in vector
    ]