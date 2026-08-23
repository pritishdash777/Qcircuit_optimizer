# ============================================================
# QUANTUM CIRCUIT ML OPTIMIZER
# Evaluation Utilities
# ============================================================

import random
import numpy as np

from .optimizer import get_redundancy_mask

from .gate_optimizer import (
    predict_gate_mask,
    ml_optimize_circuit
)

from .circuit_utils import compare_circuits


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_circuits(
    circuits,
    train_ratio=0.8,
    seed=42
):
    """
    Split circuits into training and testing sets.

    Example:

    10,000 circuits
        ↓

    8,000 training circuits
    2,000 testing circuits

    The test circuits remain unseen during training.
    """

    circuits = list(circuits)

    # Fixed seed makes the split reproducible
    rng = random.Random(seed)

    rng.shuffle(circuits)

    split_index = int(
        len(circuits) * train_ratio
    )

    train_circuits = circuits[
        :split_index
    ]

    test_circuits = circuits[
        split_index:
    ]

    return (
        train_circuits,
        test_circuits
    )


# ============================================================
# BINARY CLASSIFICATION METRICS
# ============================================================

def calculate_binary_metrics(
    actual,
    predicted
):
    """
    Calculate:

    Accuracy
    Precision
    Recall
    F1 score

    In our project:

    1 = REMOVE gate
    0 = KEEP gate
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0


    for actual_value, predicted_value in zip(
        actual,
        predicted
    ):

        # Correctly predicted REMOVE
        if (
            actual_value == 1
            and predicted_value == 1
        ):
            true_positive += 1

        # Correctly predicted KEEP
        elif (
            actual_value == 0
            and predicted_value == 0
        ):
            true_negative += 1

        # Model removed something
        # that should have been kept
        elif (
            actual_value == 0
            and predicted_value == 1
        ):
            false_positive += 1

        # Model kept something
        # that should have been removed
        elif (
            actual_value == 1
            and predicted_value == 0
        ):
            false_negative += 1


    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )


    if total == 0:
        accuracy = 0.0

    else:
        accuracy = (
            true_positive
            + true_negative
        ) / total


    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:
        precision = 0.0

    else:
        precision = (
            true_positive
            / precision_denominator
        )


    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:
        recall = 0.0

    else:
        recall = (
            true_positive
            / recall_denominator
        )


    if (
        precision + recall
    ) == 0:

        f1 = 0.0

    else:

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative
    }


# ============================================================
# EVALUATE ONE CIRCUIT
# ============================================================

def evaluate_single_circuit(
    model,
    circuit,
    fidelity_threshold=0.99,
    prediction_threshold=0.5
):
    """
    Evaluate the ML optimizer on one circuit.

    Steps:

    1. Get rule-based correct mask
    2. Get ML predicted mask
    3. Build ML optimized circuit
    4. Measure gate reduction
    5. Measure fidelity
    """

    # --------------------------------------------------------
    # Correct answer from rule-based teacher
    # --------------------------------------------------------

    actual_mask = get_redundancy_mask(
        circuit
    )


    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    predicted_mask, probabilities = (
        predict_gate_mask(
            model,
            circuit,
            threshold=prediction_threshold
        )
    )


    # --------------------------------------------------------
    # Make lengths equal
    # --------------------------------------------------------

    comparison_length = min(
        len(actual_mask),
        len(predicted_mask)
    )

    actual_mask = actual_mask[
        :comparison_length
    ]

    predicted_mask = predicted_mask[
        :comparison_length
    ]


    # --------------------------------------------------------
    # Exact mask accuracy
    # --------------------------------------------------------

    exact_match = (
        actual_mask
        == predicted_mask
    )


    # --------------------------------------------------------
    # Build optimized circuit
    # --------------------------------------------------------

    optimized_circuit, _, _ = (
        ml_optimize_circuit(
            circuit,
            model,
            threshold=prediction_threshold
        )
    )


    # --------------------------------------------------------
    # Gate statistics
    # --------------------------------------------------------

    original_gates = len(
        circuit.data
    )

    optimized_gates = len(
        optimized_circuit.data
    )

    removed_gates = (
        original_gates
        - optimized_gates
    )


    if original_gates == 0:

        reduction_percentage = 0.0

    else:

        reduction_percentage = (
            removed_gates
            / original_gates
        ) * 100


    # --------------------------------------------------------
    # Fidelity
    # --------------------------------------------------------

    try:

        fidelity = compare_circuits(
            circuit,
            optimized_circuit
        )

        fidelity = float(
            fidelity
        )

    except Exception:

        # If simulation fails, mark fidelity as invalid
        fidelity = np.nan


    # --------------------------------------------------------
    # State preservation check
    # --------------------------------------------------------

    if np.isnan(fidelity):

        fidelity_success = False

    else:

        fidelity_success = (
            fidelity
            >= fidelity_threshold
        )


    return {
        "actual_mask": actual_mask,
        "predicted_mask": predicted_mask,
        "probabilities": probabilities,

        "exact_match": exact_match,

        "original_gates": original_gates,
        "optimized_gates": optimized_gates,
        "removed_gates": removed_gates,

        "reduction_percentage": reduction_percentage,

        "fidelity": fidelity,
        "fidelity_success": fidelity_success
    }


# ============================================================
# EVALUATE MODEL ON MANY UNSEEN CIRCUITS
# ============================================================

def evaluate_gate_optimizer(
    model,
    test_circuits,
    fidelity_threshold=0.99,
    prediction_threshold=0.5,
    verbose=False
):
    """
    Evaluate the trained CNN on many unseen circuits.

    Calculates:

    - Gate-level accuracy
    - Precision
    - Recall
    - F1 score
    - Exact-mask accuracy
    - Average gate reduction
    - Average fidelity
    - Fidelity success rate
    - Safe optimization rate
    """

    all_actual = []
    all_predicted = []

    exact_matches = 0

    total_original_gates = 0
    total_optimized_gates = 0
    total_removed_gates = 0

    reduction_percentages = []
    fidelities = []

    fidelity_successes = 0

    safe_optimizations = 0

    evaluated_circuits = 0


    # ========================================================
    # TEST EVERY CIRCUIT
    # ========================================================

    for index, circuit in enumerate(
        test_circuits
    ):

        result = evaluate_single_circuit(
            model,
            circuit,
            fidelity_threshold=fidelity_threshold,
            prediction_threshold=prediction_threshold
        )

        evaluated_circuits += 1


        # ----------------------------------------------------
        # Collect gate-level labels
        # ----------------------------------------------------

        all_actual.extend(
            result["actual_mask"]
        )

        all_predicted.extend(
            result["predicted_mask"]
        )


        # ----------------------------------------------------
        # Exact mask
        # ----------------------------------------------------

        if result["exact_match"]:

            exact_matches += 1


        # ----------------------------------------------------
        # Gate statistics
        # ----------------------------------------------------

        total_original_gates += (
            result["original_gates"]
        )

        total_optimized_gates += (
            result["optimized_gates"]
        )

        total_removed_gates += (
            result["removed_gates"]
        )

        reduction_percentages.append(
            result[
                "reduction_percentage"
            ]
        )


        # ----------------------------------------------------
        # Fidelity
        # ----------------------------------------------------

        if not np.isnan(
            result["fidelity"]
        ):

            fidelities.append(
                result["fidelity"]
            )


        if result["fidelity_success"]:

            fidelity_successes += 1


        # ----------------------------------------------------
        # Safe optimization
        # ----------------------------------------------------
        #
        # We define a safe optimization as:
        #
        # 1. At least one gate was removed
        # AND
        # 2. Fidelity remained above threshold
        #

        if (
            result["removed_gates"] > 0
            and result["fidelity_success"]
        ):

            safe_optimizations += 1


        # ----------------------------------------------------
        # Optional debugging output
        # ----------------------------------------------------

        if verbose:

            print(
                f"\nCircuit {index + 1}"
            )

            print(
                "Actual:",
                result["actual_mask"]
            )

            print(
                "Predicted:",
                result["predicted_mask"]
            )

            print(
                "Removed:",
                result["removed_gates"]
            )

            print(
                "Fidelity:",
                result["fidelity"]
            )


    # ========================================================
    # CALCULATE CLASSIFICATION METRICS
    # ========================================================

    classification_metrics = (
        calculate_binary_metrics(
            all_actual,
            all_predicted
        )
    )


    # ========================================================
    # EXACT MASK ACCURACY
    # ========================================================

    if evaluated_circuits == 0:

        exact_mask_accuracy = 0.0

    else:

        exact_mask_accuracy = (
            exact_matches
            / evaluated_circuits
        )


    # ========================================================
    # AVERAGE GATE REDUCTION
    # ========================================================

    if len(
        reduction_percentages
    ) == 0:

        average_gate_reduction = 0.0

    else:

        average_gate_reduction = float(
            np.mean(
                reduction_percentages
            )
        )


    # ========================================================
    # OVERALL GATE REDUCTION
    # ========================================================

    if total_original_gates == 0:

        overall_gate_reduction = 0.0

    else:

        overall_gate_reduction = (
            total_removed_gates
            / total_original_gates
        ) * 100


    # ========================================================
    # AVERAGE FIDELITY
    # ========================================================

    if len(fidelities) == 0:

        average_fidelity = 0.0

    else:

        average_fidelity = float(
            np.mean(
                fidelities
            )
        )


    # ========================================================
    # FIDELITY SUCCESS RATE
    # ========================================================

    if evaluated_circuits == 0:

        fidelity_success_rate = 0.0

    else:

        fidelity_success_rate = (
            fidelity_successes
            / evaluated_circuits
        )


    # ========================================================
    # SAFE OPTIMIZATION RATE
    # ========================================================

    if evaluated_circuits == 0:

        safe_optimization_rate = 0.0

    else:

        safe_optimization_rate = (
            safe_optimizations
            / evaluated_circuits
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    results = {

        "test_circuits": evaluated_circuits,

        # Gate classification
        "gate_accuracy":
            classification_metrics[
                "accuracy"
            ],

        "precision":
            classification_metrics[
                "precision"
            ],

        "recall":
            classification_metrics[
                "recall"
            ],

        "f1_score":
            classification_metrics[
                "f1"
            ],

        # Confusion values
        "true_positive":
            classification_metrics[
                "true_positive"
            ],

        "true_negative":
            classification_metrics[
                "true_negative"
            ],

        "false_positive":
            classification_metrics[
                "false_positive"
            ],

        "false_negative":
            classification_metrics[
                "false_negative"
            ],

        # Circuit-level
        "exact_mask_accuracy":
            exact_mask_accuracy,

        # Optimization
        "total_original_gates":
            total_original_gates,

        "total_optimized_gates":
            total_optimized_gates,

        "total_removed_gates":
            total_removed_gates,

        "average_gate_reduction":
            average_gate_reduction,

        "overall_gate_reduction":
            overall_gate_reduction,

        # Fidelity
        "average_fidelity":
            average_fidelity,

        "fidelity_success_rate":
            fidelity_success_rate,

        # Safe optimization
        "safe_optimization_rate":
            safe_optimization_rate
    }


    return results


# ============================================================
# PRINT EVALUATION REPORT
# ============================================================

def print_evaluation_report(results):
    """
    Display evaluation metrics in a clean format.
    """

    print("\n============================================")
    print("CNN GATE OPTIMIZER EVALUATION")
    print("============================================")

    print(
        "\nTest circuits:",
        results["test_circuits"]
    )


    print("\n--- GATE-LEVEL CLASSIFICATION ---")

    print(
        f"Accuracy:  "
        f"{results['gate_accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{results['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:    "
        f"{results['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score:  "
        f"{results['f1_score'] * 100:.2f}%"
    )


    print("\n--- CONFUSION COUNTS ---")

    print(
        "True Positive:",
        results["true_positive"]
    )

    print(
        "True Negative:",
        results["true_negative"]
    )

    print(
        "False Positive:",
        results["false_positive"]
    )

    print(
        "False Negative:",
        results["false_negative"]
    )


    print("\n--- CIRCUIT-LEVEL PERFORMANCE ---")

    print(
        f"Exact mask accuracy: "
        f"{results['exact_mask_accuracy'] * 100:.2f}%"
    )


    print("\n--- CIRCUIT OPTIMIZATION ---")

    print(
        "Original gates:",
        results["total_original_gates"]
    )

    print(
        "Optimized gates:",
        results["total_optimized_gates"]
    )

    print(
        "Removed gates:",
        results["total_removed_gates"]
    )

    print(
        f"Average gate reduction: "
        f"{results['average_gate_reduction']:.2f}%"
    )

    print(
        f"Overall gate reduction: "
        f"{results['overall_gate_reduction']:.2f}%"
    )


    print("\n--- QUANTUM VALIDATION ---")

    print(
        f"Average fidelity: "
        f"{results['average_fidelity']:.4f}"
    )

    print(
        f"Fidelity success rate: "
        f"{results['fidelity_success_rate'] * 100:.2f}%"
    )

    print(
        f"Safe optimization rate: "
        f"{results['safe_optimization_rate'] * 100:.2f}%"
    )


    print("\n============================================")
    print("EVALUATION COMPLETED")
    print("============================================")
    # ============================================================
# THRESHOLD SAFETY EVALUATION
# ============================================================

def evaluate_thresholds(
    model,
    test_circuits,
    thresholds=None,
    fidelity_threshold=0.99
):
    """
    Test the same trained model using different
    gate-removal probability thresholds.

    A higher threshold makes the optimizer
    more conservative.

    Example:

    threshold = 0.50
    -> remove gate if probability >= 50%

    threshold = 0.90
    -> remove gate only if probability >= 90%
    """

    if thresholds is None:

        thresholds = [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90
        ]

    all_results = []

    print("\n==============================================================")
    print("THRESHOLD SAFETY ANALYSIS")
    print("==============================================================")

    for threshold in thresholds:

        print(
            f"\nTesting threshold: {threshold:.2f}"
        )

        results = evaluate_gate_optimizer(
            model,
            test_circuits,
            fidelity_threshold=fidelity_threshold,
            prediction_threshold=threshold,
            verbose=False
        )

        results["threshold"] = threshold

        all_results.append(
            results
        )

    # --------------------------------------------------------
    # Print comparison table
    # --------------------------------------------------------

    print("\n")
    print(
        "Threshold | Accuracy | Precision | Recall | "
        "F1 | Fidelity | Fidelity Success | Reduction"
    )

    print("-" * 100)

    for result in all_results:

        print(
            f"{result['threshold']:^9.2f} | "
            f"{result['gate_accuracy'] * 100:^8.2f}% | "
            f"{result['precision'] * 100:^9.2f}% | "
            f"{result['recall'] * 100:^6.2f}% | "
            f"{result['f1_score'] * 100:^6.2f}% | "
            f"{result['average_fidelity']:^8.4f} | "
            f"{result['fidelity_success_rate'] * 100:^15.2f}% | "
            f"{result['overall_gate_reduction']:^8.2f}%"
        )

    print(
        "\n=============================================================="
    )

    return all_results