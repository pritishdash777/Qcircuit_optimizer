from .data_generator import random_circuit
from .encoding import encode_circuit

import numpy as np


def generate_autoencoder_data(
    samples=1000
):

    dataset = []

    for _ in range(samples):

        circuit = random_circuit()

        vector = encode_circuit(
            circuit
        )

        dataset.append(vector)

    return np.array(
        dataset,
        dtype=np.float32
    )