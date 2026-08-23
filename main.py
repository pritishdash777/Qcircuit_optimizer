# ============================================================
# QUANTUM CIRCUIT OPTIMIZER
# Main ML Pipeline
# ============================================================

from qiskit import QuantumCircuit
import torch

from src.data_generator import (
    create_dataset,
    random_circuit
)

from src.ml_models import (
    train_model,
    predict_redundancy
)

from src.circuit_utils import (
    circuit_to_features,
    compare_circuits
)

from src.ae_dataset import generate_autoencoder_data

from src.autoencoder import (
    CircuitAutoencoder,
    train_autoencoder
)

from src.encoding import encode_circuit

from src.optimizer import (
    get_redundancy_mask,
    validate_optimization,
    print_safety_report
)

from src.gate_optimizer import (
    GateOptimizerNN,
    prepare_gate_optimizer_data,
    train_gate_optimizer,
    predict_gate_mask,
    ml_optimize_circuit
)

from src.evaluation import (
    split_circuits,
    evaluate_gate_optimizer,
    print_evaluation_report,
    evaluate_thresholds
)

from src.model_utils import (
    save_model,
    load_model,
    model_exists
)

# ============================================================
# PROJECT SETTINGS
# ============================================================

REDUNDANCY_DATASET_SIZE = 1000

AUTOENCODER_DATASET_SIZE = 10000

CNN_DATASET_SIZE = 10000

REDUNDANCY_EPOCHS = 100

AUTOENCODER_EPOCHS = 100

CNN_EPOCHS = 200
# ============================================================
# MODEL SETTINGS
# ============================================================

TRAIN_NEW_CNN = False

CNN_MODEL_PATH = "models/gate_optimizer_cnn.pth"

# ============================================================
# HELPER: SECTION TITLE
# ============================================================

def print_section(title):
    """
    Print a clean section heading.
    """

    print("\n====================================")
    print(title)
    print("====================================")


# ============================================================
# CREATE SIMPLE TEST CIRCUIT
# ============================================================

def create_test_circuit():
    """
    Circuit used to test the simple redundancy predictor.

    Circuit:

    q0: X -> X -> Z
    q1: H -> H

    X-X cancels.
    H-H cancels.

    Therefore 4 gates are removable.
    """

    circuit = QuantumCircuit(2)

    circuit.x(0)
    circuit.x(0)

    circuit.h(1)
    circuit.h(1)

    circuit.z(0)

    return circuit


# ============================================================
# CREATE GATE-LEVEL TEST CIRCUIT
# ============================================================

def create_optimizer_test_circuit():
    """
    Test circuit for the gate-level CNN.

    X -> X -> H -> Z -> Z

    Correct optimized result:

    H
    """

    circuit = QuantumCircuit(1)

    circuit.x(0)
    circuit.x(0)

    circuit.h(0)

    circuit.z(0)
    circuit.z(0)

    return circuit


# ============================================================
# PART 1
# TRAIN SIMPLE REDUNDANCY PREDICTOR
# ============================================================

def train_redundancy_predictor():

    print_section(
        "TRAINING REDUNDANCY PREDICTOR"
    )

    # Generate training dataset
    X, y = create_dataset(
        REDUNDANCY_DATASET_SIZE
    )

    # Train neural network
    model = train_model(
        X,
        y,
        epochs=REDUNDANCY_EPOCHS
    )

    return model


# ============================================================
# PART 2
# TEST SIMPLE REDUNDANCY PREDICTOR
# ============================================================

def test_redundancy_predictor(model):

    print_section(
        "TESTING REDUNDANCY PREDICTOR"
    )

    circuit = create_test_circuit()

    print("\nTest Circuit:")
    print(circuit)

    # Convert circuit into six numerical features
    features = circuit_to_features(
        circuit
    )

    # Ask model how many gates may be removable
    prediction = predict_redundancy(
        model,
        features
    )

    print(
        f"\nPredicted removable gates: "
        f"{prediction:.2f}"
    )

    return circuit


# ============================================================
# PART 3
# TRAIN AUTOENCODER
# ============================================================

def train_autoencoder_model():

    print_section(
        "GENERATING AUTOENCODER DATA"
    )

    # Generate circuits for autoencoder training
    ae_data = generate_autoencoder_data(
        samples=AUTOENCODER_DATASET_SIZE
    )

    print(
        "Autoencoder dataset shape:",
        ae_data.shape
    )

    print_section(
        "TRAINING AUTOENCODER"
    )

    # Create model
    model = CircuitAutoencoder()

    # Train model
    model = train_autoencoder(
        model,
        ae_data,
        epochs=AUTOENCODER_EPOCHS
    )

    return model


# ============================================================
# PART 4
# SHOW AUTOENCODER LATENT REPRESENTATION
# ============================================================

def inspect_autoencoder(
    autoencoder,
    circuit
):

    print_section(
        "AUTOENCODER CIRCUIT REPRESENTATION"
    )

    # Convert circuit into numerical form
    encoded_circuit = encode_circuit(
        circuit
    )

    print("\nEncoded circuit:")
    print(encoded_circuit)

    # Convert NumPy data into PyTorch tensor
    input_tensor = torch.tensor(
        encoded_circuit,
        dtype=torch.float32
    ).unsqueeze(0)

    # Extract the compressed representation
    with torch.no_grad():

        latent_vector = autoencoder.encoder(
            input_tensor
        )

    print("\n8D latent representation:")
    print(latent_vector)

    return latent_vector


# ============================================================
# PART 5
# TEST RULE-BASED OPTIMIZER
# ============================================================

def test_rule_based_optimizer():

    print_section(
        "TESTING RULE-BASED OPTIMIZER"
    )

    circuit = create_optimizer_test_circuit()

    print("\nCircuit:")
    print(circuit)

    mask = get_redundancy_mask(
        circuit
    )

    print(
        "\nRule-based removal mask:",
        mask
    )

    return circuit, mask


# ============================================================
# PART 6
# TRAIN CNN GATE OPTIMIZER
# ============================================================

def train_cnn_optimizer():

    print_section(
        "TRAINING GATE-LEVEL CNN OPTIMIZER"
    )

    # --------------------------------------------------------
    # Generate complete dataset
    # --------------------------------------------------------

    all_circuits = [
        random_circuit()
        for _ in range(CNN_DATASET_SIZE)
    ]

    print(
        f"Generated {len(all_circuits)} total circuits"
    )

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    training_circuits, test_circuits = split_circuits(
        all_circuits,
        train_ratio=0.8,
        seed=42
    )

    print(
        f"Training circuits: {len(training_circuits)}"
    )

    print(
        f"Unseen test circuits: {len(test_circuits)}"
    )

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------

    X_gate, y_gate, valid_gate = (
        prepare_gate_optimizer_data(
            training_circuits
        )
    )

    print(
        "Input shape:",
        X_gate.shape
    )

    print(
        "Target shape:",
        y_gate.shape
    )

    print(
        "Valid mask shape:",
        valid_gate.shape
    )

    # --------------------------------------------------------
    # Create CNN model
    # --------------------------------------------------------

    model = GateOptimizerNN()

    # --------------------------------------------------------
    # Train CNN model
    # --------------------------------------------------------

    model = train_gate_optimizer(
        model,
        X_gate,
        y_gate,
        valid_gate,
        epochs=CNN_EPOCHS
    )

    # --------------------------------------------------------
    # Save trained CNN model
    # --------------------------------------------------------

    save_model(
        model,
        CNN_MODEL_PATH
    )

    print(
        "\nCNN gate optimizer training completed."
    )

    # --------------------------------------------------------
    # Return model and unseen test circuits
    # --------------------------------------------------------

    return model, test_circuits  # FIZ


# ============================================================
# LOAD SAVED CNN
# ============================================================

def load_saved_cnn():
    """
    Load the previously trained CNN model
    from the saved .pth file.
    """

    print_section(
        "LOADING SAVED CNN OPTIMIZER"
    )

    # Create the CNN architecture
    model = GateOptimizerNN()

    # Load the saved trained weights
    model = load_model(
        model,
        CNN_MODEL_PATH
    )

    print(
        "Saved CNN loaded successfully."
    )

    return model


# ============================================================
# PART 7
# TEST CNN GATE OPTIMIZER
# ============================================================

def test_cnn_optimizer(model):
    """
    Test the trained CNN optimizer on
    a hand-made quantum circuit.
    """

    print_section(
        "TESTING CNN GATE OPTIMIZER"
    )

    # --------------------------------------------------------
    # Create hand-made test circuit
    # --------------------------------------------------------

    test_circuit = create_optimizer_test_circuit()

    print("\nOriginal circuit:")
    print(test_circuit)
    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    predicted_mask, probabilities = (
        predict_gate_mask(
            model,
            test_circuit,
            threshold=0.60
        )
    )

    print(
        "ML predicted mask:",
        predicted_mask
    )


    # --------------------------------------------------------
    # SHOW ML PROBABILITIES
    # --------------------------------------------------------

    print(
        "\nGate removal probabilities:"
    )

    for index, probability in enumerate(
        probabilities
    ):

        instruction = (
            test_circuit.data[index]
        )

        gate_name = (
            instruction
            .operation
            .name
            .upper()
        )

        qubit_numbers = [
            test_circuit
            .find_bit(qubit)
            .index
            for qubit
            in instruction.qubits
        ]

        print(
            f"Gate {index}: "
            f"{gate_name}"
            f"(q{qubit_numbers}) "
            f"-> "
            f"{probability:.4f}"
        )


    # --------------------------------------------------------
    # ML PROPOSES AN OPTIMIZED CIRCUIT
    # --------------------------------------------------------

    candidate_circuit, _, _ = (
        ml_optimize_circuit(
            test_circuit,
            model,
            threshold=0.60
        )
    )

    print(
        "\nML proposed circuit:"
    )

    print(
        candidate_circuit
    )


    # --------------------------------------------------------
    # SAFETY VALIDATION
    # --------------------------------------------------------
    #
    # Important:
    #
    # ML does NOT automatically become the final result.
    #
    # The simulator checks the candidate first.
    #
    # Fidelity >= 0.99
    # → ACCEPT
    #
    # Fidelity < 0.99
    # → REJECT
    # → return original circuit
    #

    (
        safe_circuit,
        fidelity,
        accepted
    ) = validate_optimization(
        test_circuit,
        candidate_circuit,
        fidelity_threshold=0.99
    )


    # --------------------------------------------------------
    # PRINT SAFETY REPORT
    # --------------------------------------------------------

    print_safety_report(
        test_circuit,
        candidate_circuit,
        safe_circuit,
        fidelity,
        accepted
    )


    # --------------------------------------------------------
    # DISPLAY FINAL ACCEPTED CIRCUIT
    # --------------------------------------------------------

    print(
        "\nFinal accepted circuit:"
    )

    print(
        safe_circuit
    )


    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    original_gates = len(
        test_circuit.data
    )

    proposed_gates = len(
        candidate_circuit.data
    )

    final_gates = len(
        safe_circuit.data
    )


    proposed_removed = (
        original_gates
        - proposed_gates
    )

    final_removed = (
        original_gates
        - final_gates
    )


    # --------------------------------------------------------
    # Proposed gate reduction
    # --------------------------------------------------------

    if original_gates > 0:

        proposed_reduction_percentage = (
            proposed_removed
            / original_gates
        ) * 100

    else:

        proposed_reduction_percentage = 0.0


    # --------------------------------------------------------
    # Final SAFE gate reduction
    # --------------------------------------------------------

    if original_gates > 0:

        final_reduction_percentage = (
            final_removed
            / original_gates
        ) * 100

    else:

        final_reduction_percentage = 0.0


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print_section(
        "SAFE OPTIMIZATION RESULT"
    )

    print(
        "Original gates:",
        original_gates
    )

    print(
        "ML proposed gates:",
        proposed_gates
    )

    print(
        "ML proposed reduction:",
        proposed_removed
    )

    print(
        f"ML proposed reduction percentage: "
        f"{proposed_reduction_percentage:.2f}%"
    )

    print(
        f"Validation fidelity: "
        f"{fidelity:.4f}"
    )

    print(
        "Optimization accepted:",
        accepted
    )

    print(
        "Final circuit gates:",
        final_gates
    )

    print(
        "Final safe gates removed:",
        final_removed
    )

    print(
        f"Final safe gate reduction: "
        f"{final_reduction_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # CHECK ML MASK
    # --------------------------------------------------------

    actual_mask = get_redundancy_mask(
        test_circuit
    )

    if predicted_mask == actual_mask:

        print(
            "Mask prediction: CORRECT"
        )

    else:

        print(
            "Mask prediction: INCORRECT"
        )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "original_gates":
            original_gates,

        "proposed_gates":
            proposed_gates,

        "optimized_gates":
            final_gates,

        "removed_gates":
            final_removed,

        "proposed_reduction_percentage":
            proposed_reduction_percentage,

        "reduction_percentage":
            final_reduction_percentage,

        "fidelity":
            fidelity,

        "accepted":
            accepted,

        "actual_mask":
            actual_mask,

        "predicted_mask":
            predicted_mask,

        "probabilities":
            probabilities
    }
    # --------------------------------------------------------
    # Rule-based teacher answer
    # --------------------------------------------------------

    actual_mask = get_redundancy_mask(
        test_circuit
    )

    print(
        "\nRule-based correct mask:",
        actual_mask
    )

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    predicted_mask, probabilities = (
        predict_gate_mask(
            model,
            test_circuit
        )
    )

    print(
        "ML predicted mask:",
        predicted_mask
    )

    # --------------------------------------------------------
    # Show probability for every gate
    # --------------------------------------------------------

    print(
        "\nGate removal probabilities:"
    )

    for index, probability in enumerate(
        probabilities
    ):

        instruction = (
            test_circuit.data[index]
        )

        gate_name = (
            instruction
            .operation
            .name
            .upper()
        )

        qubit_numbers = [
            test_circuit
            .find_bit(qubit)
            .index
            for qubit
            in instruction.qubits
        ]

        print(
            f"Gate {index}: "
            f"{gate_name}"
            f"(q{qubit_numbers}) "
            f"-> "
            f"{probability:.4f}"
        )

    # --------------------------------------------------------
    # Build optimized circuit using ML
    # --------------------------------------------------------

    optimized_circuit, _, _ = (
        ml_optimize_circuit(
            test_circuit,
            model
        )
    )

    print("\nML Optimized circuit:")
    print(optimized_circuit)

    # --------------------------------------------------------
    # Calculate fidelity
    # --------------------------------------------------------

    fidelity = compare_circuits(
        test_circuit,
        optimized_circuit
    )

    # --------------------------------------------------------
    # Optimization statistics
    # --------------------------------------------------------

    original_gates = len(
        test_circuit.data
    )

    optimized_gates = len(
        optimized_circuit.data
    )

    removed_gates = (
        original_gates
        - optimized_gates
    )

    if original_gates > 0:

        reduction_percentage = (
            removed_gates
            / original_gates
        ) * 100

    else:

        reduction_percentage = 0.0

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print_section(
        "OPTIMIZATION RESULT"
    )

    print(
        "Original gates:",
        original_gates
    )

    print(
        "Optimized gates:",
        optimized_gates
    )

    print(
        "Removed gates:",
        removed_gates
    )

    print(
        f"Gate reduction: "
        f"{reduction_percentage:.2f}%"
    )

    print(
        f"Fidelity: "
        f"{fidelity:.4f}"
    )

    # --------------------------------------------------------
    # Simple validation message
    # --------------------------------------------------------

    if predicted_mask == actual_mask:

        print(
            "Mask prediction: CORRECT"
        )

    else:

        print(
            "Mask prediction: INCORRECT"
        )

    if fidelity >= 0.99:

        print(
            "Quantum state preserved: YES"
        )

    else:

        print(
            "Quantum state preserved: NO"
        )

    return {
        "original_gates": original_gates,
        "optimized_gates": optimized_gates,
        "removed_gates": removed_gates,
        "reduction_percentage": reduction_percentage,
        "fidelity": fidelity,
        "actual_mask": actual_mask,
        "predicted_mask": predicted_mask,
        "probabilities": probabilities
    }
def main():

    print_section(
        "QUANTUM ALGORITHM SIMULATOR "
        "WITH ML-BASED CIRCUIT OPTIMIZATION"
    )

    # --------------------------------------------------------
    # 1. Train simple redundancy predictor
    # --------------------------------------------------------

    redundancy_model = (
        train_redundancy_predictor()
    )

    test_circuit = (
        test_redundancy_predictor(
            redundancy_model
        )
    )

    # --------------------------------------------------------
    # 2. Train autoencoder
    # --------------------------------------------------------

    autoencoder = (
        train_autoencoder_model()
    )

    inspect_autoencoder(
        autoencoder,
        test_circuit
    )

    # --------------------------------------------------------
    # 3. Test rule-based optimizer
    # --------------------------------------------------------

    test_rule_based_optimizer()
    # --------------------------------------------------------
    # 4. Train OR load CNN gate optimizer
    # --------------------------------------------------------

    if TRAIN_NEW_CNN or not model_exists(
        CNN_MODEL_PATH
    ):

        # Train a new CNN
        cnn_model, test_circuits = (
            train_cnn_optimizer()
        )

    else:

        # Load the already-trained CNN
        cnn_model = load_saved_cnn()

        # Generate circuits for evaluation
        all_circuits = [
            random_circuit()
            for _ in range(CNN_DATASET_SIZE)
        ]

        # Keep only the unseen test portion
        _, test_circuits = split_circuits(
            all_circuits,
            train_ratio=0.8,
            seed=42
        )


    # --------------------------------------------------------
    # 5. Test CNN on hand-made circuit
    # --------------------------------------------------------

    results = test_cnn_optimizer(
        cnn_model
    )

    # --------------------------------------------------------
    # 6. Evaluate CNN on unseen circuits
    # --------------------------------------------------------

    print_section(
        "EVALUATING CNN ON UNSEEN CIRCUITS"
    )

    evaluation_results = evaluate_gate_optimizer(
        cnn_model,
        test_circuits,
        fidelity_threshold=0.99,
        prediction_threshold=0.60,
        verbose=False
    )

    print_evaluation_report(
        evaluation_results
    )
    # --------------------------------------------------------
    # 7. Threshold safety analysis
    # --------------------------------------------------------

    threshold_results = evaluate_thresholds(
        cnn_model,
        test_circuits,
        thresholds=[
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
        ],
        fidelity_threshold=0.99
    )

    # --------------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------------

    print_section(
        "FULL ML PIPELINE COMPLETED"
    )

    print(
        "Hand-made test fidelity:",
        f"{results['fidelity']:.4f}"
    )

    print(
        "Hand-made test gate reduction:",
        f"{results['reduction_percentage']:.2f}%"
    )

    print(
        "Unseen test gate accuracy:",
        f"{evaluation_results['gate_accuracy'] * 100:.2f}%"
    )

    print(
        "Unseen test precision:",
        f"{evaluation_results['precision'] * 100:.2f}%"
    )

    print(
        "Unseen test recall:",
        f"{evaluation_results['recall'] * 100:.2f}%"
    )

    print(
        "Unseen test F1 score:",
        f"{evaluation_results['f1_score'] * 100:.2f}%"
    )

    print(
        "Exact-mask accuracy:",
        f"{evaluation_results['exact_mask_accuracy'] * 100:.2f}%"
    )

    print(
        "Average unseen fidelity:",
        f"{evaluation_results['average_fidelity']:.4f}"
    )

    print(
        "Overall gate reduction:",
        f"{evaluation_results['overall_gate_reduction']:.2f}%"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()