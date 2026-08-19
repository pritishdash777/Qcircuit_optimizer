from qiskit import QuantumCircuit
import torch

from src.data_generator import create_dataset
from src.ml_models import train_model, predict_redundancy
from src.circuit_utils import circuit_to_features

from src.ae_dataset import generate_autoencoder_data
from src.autoencoder import CircuitAutoencoder, train_autoencoder
from src.encoding import encode_circuit


# ============================================================
# PART 1: Train the simple Neural Network
# ============================================================

print("\n====================================")
print("TRAINING REDUNDANCY PREDICTOR")
print("====================================")

X, y = create_dataset(1000)

nn_model = train_model(
    X,
    y,
    epochs=100
)


# ============================================================
# PART 2: Test the Neural Network
# ============================================================

test_circuit = QuantumCircuit(2)

test_circuit.x(0)
test_circuit.x(0)

test_circuit.h(1)
test_circuit.h(1)

test_circuit.z(0)

print("\nTest Circuit:")
print(test_circuit)


features = circuit_to_features(
    test_circuit
)

prediction = predict_redundancy(
    nn_model,
    features
)

print(
    f"Predicted removable gates: {prediction:.2f}"
)


# ============================================================
# PART 3: Generate Autoencoder Dataset
# ============================================================

print("\n====================================")
print("GENERATING AUTOENCODER DATA")
print("====================================")

ae_data = generate_autoencoder_data(
    samples=1000
)

print(
    "Autoencoder dataset shape:",
    ae_data.shape
)


# ============================================================
# PART 4: Train Autoencoder
# ============================================================

print("\n====================================")
print("TRAINING AUTOENCODER")
print("====================================")

autoencoder = CircuitAutoencoder()

autoencoder = train_autoencoder(
    autoencoder,
    ae_data,
    epochs=100
)


# ============================================================
# PART 5: Encode Test Circuit
# ============================================================

encoded_circuit = encode_circuit(
    test_circuit
)

print("\nEncoded circuit:")
print(encoded_circuit)


# ============================================================
# PART 6: Extract Latent Representation
# ============================================================

input_tensor = torch.tensor(
    encoded_circuit,
    dtype=torch.float32
).unsqueeze(0)


with torch.no_grad():

    latent_vector = autoencoder.encoder(
        input_tensor
    )


print("\nLatent representation:")
print(latent_vector)


# ============================================================
# FINISHED
# ============================================================

print("\n====================================")
print("ML PIPELINE COMPLETED")
print("====================================")
from src.optimizer import get_redundancy_mask


test = QuantumCircuit(1)

test.x(0)
test.x(0)

test.h(0)

test.z(0)
test.z(0)


print("\nCircuit:")
print(test)

mask = get_redundancy_mask(test)

print("Removal mask:")
print(mask)