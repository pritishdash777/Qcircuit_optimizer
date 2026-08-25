# ============================================================
# QUANTUM CIRCUIT ML OPTIMIZER
# Evaluation Utilities
#
# Safety evaluation:
# Exact operator equivalence up to global phase
# ============================================================

import random

from .optimizer import get_redundancy_mask

from .gate_optimizer import (
    predict_gate_mask,
    ml_optimize_circuit
)

from .circuit_utils import (
    check_circuit_equivalence
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_circuits(
    circuits,
    train_ratio=0.8,
    seed=42
):
    """
    Split circuits into reproducible training and testing sets.

    Example:

        10,000 circuits
              ↓
        8,000 training
        2,000 testing

    The supplied seed controls the shuffle.
    """

    circuits = list(circuits)

    rng = random.Random(seed)

    rng.shuffle(circuits)

    split_index = int(
        len(circuits) * train_ratio
    )

    training_circuits = circuits[
        :split_index
    ]

    test_circuits = circuits[
        split_index:
    ]

    return (
        training_circuits,
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
    Calculate gate-level classification metrics.

    Label meaning:

        1 = REMOVE gate
        0 = KEEP gate

    Metrics:

        Accuracy
        Precision
        Recall
        F1 score
        Confusion counts
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for actual_value, predicted_value in zip(
        actual,
        predicted
    ):

        # Correct REMOVE prediction
        if (
            actual_value == 1
            and predicted_value == 1
        ):
            true_positive += 1

        # Correct KEEP prediction
        elif (
            actual_value == 0
            and predicted_value == 0
        ):
            true_negative += 1

        # Removed a gate that should have been kept
        elif (
            actual_value == 0
            and predicted_value == 1
        ):
            false_positive += 1

        # Kept a gate that should have been removed
        elif (
            actual_value == 1
            and predicted_value == 0
        ):
            false_negative += 1


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

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
        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "true_positive":
            true_positive,

        "true_negative":
            true_negative,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative
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
    Evaluate one ML-generated circuit optimization.

    IMPORTANT:

    `fidelity_threshold` is retained only for backward
    compatibility with existing main.py calls.

    It is NOT used as the safety criterion.

    Safety is determined using exact operator equivalence
    up to global phase.

    Evaluation steps:

        1. Generate the rule-based teacher mask.
        2. Generate the CNN removal mask.
        3. Compare the masks.
        4. Build the ML-proposed optimized circuit.
        5. Calculate gate reduction.
        6. Verify full operator equivalence.
        7. Mark the ML proposal as safe or unsafe.

    The ML proposal is safe ONLY when complete operator
    equivalence can be established.
    """

    # Prevent accidental assumption that this parameter
    # controls the new safety mechanism.
    _ = fidelity_threshold


    # --------------------------------------------------------
    # Rule-based teacher mask
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
    # Align mask lengths
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
    # Exact mask match
    # --------------------------------------------------------

    exact_match = (
        actual_mask
        == predicted_mask
    )


    # --------------------------------------------------------
    # Build ML-proposed optimized circuit
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
    # EXACT OPERATOR EQUIVALENCE
    # --------------------------------------------------------

    try:

        equivalence_result = (
            check_circuit_equivalence(
                circuit,
                optimized_circuit
            )
        )

        operator_equivalent = bool(
            equivalence_result.get(
                "equivalent",
                False
            )
        )

        equivalence_error = (
            equivalence_result.get(
                "equivalence_error",
                None
            )
        )

        equivalence_reason = (
            equivalence_result.get(
                "reason",
                "No reason provided."
            )
        )

    except Exception as error:

        # Fail-safe policy:
        #
        # If equivalence cannot be established,
        # the optimization is NOT considered safe.

        operator_equivalent = False

        equivalence_error = None

        equivalence_reason = (
            "Operator-equivalence verification "
            f"failed: {error}"
        )


    # --------------------------------------------------------
    # SAFE ML PROPOSAL
    # --------------------------------------------------------
    #
    # A useful optimization must:
    #
    # 1. Remove at least one gate
    # 2. Preserve the COMPLETE circuit operator
    #
    # A circuit that removes zero gates can still be
    # equivalent, but it is not counted as a successful
    # optimization.
    # --------------------------------------------------------

    safe_optimization = (
        removed_gates > 0
        and operator_equivalent
    )


    return {

        # Teacher / CNN outputs
        "actual_mask":
            actual_mask,

        "predicted_mask":
            predicted_mask,

        "probabilities":
            probabilities,

        # Classification
        "exact_match":
            exact_match,

        # Gate statistics
        "original_gates":
            original_gates,

        "optimized_gates":
            optimized_gates,

        "removed_gates":
            removed_gates,

        "reduction_percentage":
            reduction_percentage,

        # Quantum safety
        "operator_equivalent":
            operator_equivalent,

        "equivalence_error":
            equivalence_error,

        "equivalence_reason":
            equivalence_reason,

        "safe_optimization":
            safe_optimization
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
    Evaluate the trained CNN on unseen circuits.

    Calculates:

        Gate-level:
            - Accuracy
            - Precision
            - Recall
            - F1 score
            - Confusion counts

        Circuit-level:
            - Exact-mask accuracy

        Optimization:
            - Gate reduction

        Quantum correctness:
            - Operator-equivalent proposals
            - Non-equivalent proposals
            - Operator-equivalence success rate
            - Safe optimization rate

    IMPORTANT:

    Operator equivalence, not statevector fidelity from
    |0...0>, is used to evaluate quantum correctness.
    """

    # Backward compatibility only.
    _ = fidelity_threshold


    # --------------------------------------------------------
    # Classification storage
    # --------------------------------------------------------

    all_actual = []
    all_predicted = []

    exact_matches = 0


    # --------------------------------------------------------
    # Gate statistics
    # --------------------------------------------------------

    total_original_gates = 0
    total_optimized_gates = 0
    total_removed_gates = 0

    reduction_percentages = []


    # --------------------------------------------------------
    # Operator-equivalence statistics
    # --------------------------------------------------------

    equivalent_proposals = 0
    non_equivalent_proposals = 0

    verification_failures = 0

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
        # Exact mask accuracy
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
        # Operator equivalence
        # ----------------------------------------------------

        if result[
            "operator_equivalent"
        ]:

            equivalent_proposals += 1

        else:

            non_equivalent_proposals += 1


        # ----------------------------------------------------
        # Verification failures
        # ----------------------------------------------------

        reason = result[
            "equivalence_reason"
        ]

        if (
            isinstance(reason, str)
            and
            "verification failed"
            in reason.lower()
        ):

            verification_failures += 1


        # ----------------------------------------------------
        # Safe optimization
        # ----------------------------------------------------

        if result[
            "safe_optimization"
        ]:

            safe_optimizations += 1


        # ----------------------------------------------------
        # Optional debugging
        # ----------------------------------------------------

        if verbose:

            print(
                f"\nCircuit {index + 1}"
            )

            print(
                "Actual mask:",
                result["actual_mask"]
            )

            print(
                "Predicted mask:",
                result["predicted_mask"]
            )

            print(
                "Removed gates:",
                result["removed_gates"]
            )

            print(
                "Operator equivalent:",
                result["operator_equivalent"]
            )

            print(
                "Equivalence error:",
                result["equivalence_error"]
            )

            print(
                "Safe optimization:",
                result["safe_optimization"]
            )


    # ========================================================
    # CLASSIFICATION METRICS
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

        average_gate_reduction = (
            sum(
                reduction_percentages
            )
            / len(
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
    # OPERATOR-EQUIVALENCE SUCCESS RATE
    # ========================================================

    if evaluated_circuits == 0:

        operator_equivalence_rate = 0.0

    else:

        operator_equivalence_rate = (
            equivalent_proposals
            / evaluated_circuits
        )


    # ========================================================
    # NON-EQUIVALENCE RATE
    # ========================================================

    if evaluated_circuits == 0:

        non_equivalence_rate = 0.0

    else:

        non_equivalence_rate = (
            non_equivalent_proposals
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
    # FINAL RESULTS
    # ========================================================

    results = {

        "test_circuits":
            evaluated_circuits,

        # ----------------------------------------------------
        # Gate classification
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Confusion values
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Circuit-level performance
        # ----------------------------------------------------

        "exact_mask_accuracy":
            exact_mask_accuracy,

        # ----------------------------------------------------
        # Gate optimization
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Operator-equivalence safety
        # ----------------------------------------------------

        "equivalent_proposals":
            equivalent_proposals,

        "non_equivalent_proposals":
            non_equivalent_proposals,

        "verification_failures":
            verification_failures,

        "operator_equivalence_rate":
            operator_equivalence_rate,

        "non_equivalence_rate":
            non_equivalence_rate,

        # ----------------------------------------------------
        # Safe useful optimization
        # ----------------------------------------------------

        "safe_optimizations":
            safe_optimizations,

        "safe_optimization_rate":
            safe_optimization_rate
    }


    return results


# ============================================================
# PRINT EVALUATION REPORT
# ============================================================

def print_evaluation_report(
    results
):
    """
    Display evaluation metrics in a clean format.
    """

    print(
        "\n============================================"
    )

    print(
        "CNN GATE OPTIMIZER EVALUATION"
    )

    print(
        "============================================"
    )


    print(
        "\nTest circuits:",
        results["test_circuits"]
    )


    # ========================================================
    # GATE CLASSIFICATION
    # ========================================================

    print(
        "\n--- GATE-LEVEL CLASSIFICATION ---"
    )

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


    # ========================================================
    # CONFUSION COUNTS
    # ========================================================

    print(
        "\n--- CONFUSION COUNTS ---"
    )

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


    # ========================================================
    # CIRCUIT-LEVEL PERFORMANCE
    # ========================================================

    print(
        "\n--- CIRCUIT-LEVEL PERFORMANCE ---"
    )

    print(
        f"Exact mask accuracy: "
        f"{results['exact_mask_accuracy'] * 100:.2f}%"
    )


    # ========================================================
    # CIRCUIT OPTIMIZATION
    # ========================================================

    print(
        "\n--- CIRCUIT OPTIMIZATION ---"
    )

    print(
        "Original gates:",
        results["total_original_gates"]
    )

    print(
        "ML proposed gates:",
        results["total_optimized_gates"]
    )

    print(
        "ML proposed removed gates:",
        results["total_removed_gates"]
    )

    print(
        f"Average proposed gate reduction: "
        f"{results['average_gate_reduction']:.2f}%"
    )

    print(
        f"Overall proposed gate reduction: "
        f"{results['overall_gate_reduction']:.2f}%"
    )


    # ========================================================
    # OPERATOR EQUIVALENCE
    # ========================================================

    print(
        "\n--- OPERATOR EQUIVALENCE VALIDATION ---"
    )

    print(
        "Equivalent ML proposals:",
        results["equivalent_proposals"]
    )

    print(
        "Non-equivalent ML proposals:",
        results["non_equivalent_proposals"]
    )

    print(
        "Verification failures:",
        results["verification_failures"]
    )

    print(
        f"Operator equivalence success rate: "
        f"{results['operator_equivalence_rate'] * 100:.2f}%"
    )


    # ========================================================
    # SAFE OPTIMIZATION
    # ========================================================

    print(
        "\n--- SAFE OPTIMIZATION PERFORMANCE ---"
    )

    print(
        "Safe useful optimizations:",
        results["safe_optimizations"]
    )

    print(
        f"Safe optimization rate: "
        f"{results['safe_optimization_rate'] * 100:.2f}%"
    )


    print(
        "\n============================================"
    )

    print(
        "EVALUATION COMPLETED"
    )

    print(
        "============================================"
    )


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

    IMPORTANT:

    The threshold controls the CNN's REMOVE decision.

    It does NOT control the operator-equivalence safety
    verification.

    A higher prediction threshold generally makes the
    ML optimizer more conservative.

    Operator equivalence is checked independently for
    every proposed circuit.
    """

    # Backward compatibility only.
    _ = fidelity_threshold


    if thresholds is None:

        thresholds = [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90
        ]


    all_results = []


    print(
        "\n=============================================================="
    )

    print(
        "THRESHOLD SAFETY ANALYSIS"
    )

    print(
        "=============================================================="
    )


    for threshold in thresholds:

        print(
            f"\nTesting threshold: "
            f"{threshold:.2f}"
        )


        results = evaluate_gate_optimizer(
            model,
            test_circuits,
            fidelity_threshold=fidelity_threshold,
            prediction_threshold=threshold,
            verbose=False
        )


        results[
            "threshold"
        ] = threshold


        all_results.append(
            results
        )


    # --------------------------------------------------------
    # Print comparison table
    # --------------------------------------------------------

    print("\n")

    print(
        "Threshold | Accuracy | Precision | Recall | "
        "F1 | Operator Eq. | Safe Opt. | Reduction"
    )

    print(
        "-" * 104
    )


    for result in all_results:

        print(
            f"{result['threshold']:^9.2f} | "
            f"{result['gate_accuracy'] * 100:^8.2f}% | "
            f"{result['precision'] * 100:^9.2f}% | "
            f"{result['recall'] * 100:^6.2f}% | "
            f"{result['f1_score'] * 100:^6.2f}% | "
            f"{result['operator_equivalence_rate'] * 100:^11.2f}% | "
            f"{result['safe_optimization_rate'] * 100:^9.2f}% | "
            f"{result['overall_gate_reduction']:^8.2f}%"
        )


    print(
        "\n=============================================================="
    )


    return all_results