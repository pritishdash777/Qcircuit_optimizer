# ============================================================
# QUANTUM ALGORITHM SIMULATOR
# WITH ML-BASED CIRCUIT OPTIMIZATION
# Streamlit Frontend
# ============================================================

import streamlit as st
from qiskit import QuantumCircuit

from src.gate_optimizer import (
    GateOptimizerNN,
    predict_gate_mask,
    ml_optimize_circuit
)

from src.optimizer import (
    validate_optimization
)

from src.model_utils import (
    load_model
)

from src.circuit_utils import (
    compare_circuits
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/gate_optimizer_cnn.pth"

ML_THRESHOLD = 0.60

FIDELITY_THRESHOLD = 0.99

DEFAULT_USER_CIRCUIT = """h 0
x 0
h 0"""


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantum Circuit Optimizer",
    page_icon="⚛️",
    layout="wide"
)


# ============================================================
# LOAD CNN MODEL
# ============================================================

@st.cache_resource
def load_cnn_model():
    """
    Load the already-trained CNN model.

    Streamlit caches it so the model is not
    loaded again every time the page refreshes.
    """

    model = GateOptimizerNN()

    model = load_model(
        model,
        MODEL_PATH
    )

    model.eval()

    return model

# ============================================================
# USER-DEFINED CIRCUIT PARSER
# ============================================================

def build_circuit_from_text(
    circuit_text,
    num_qubits
):
    """
    Convert user-written gate instructions
    into a Qiskit QuantumCircuit.

    Supported examples:

    x 0
    h 1
    z 0
    cx 0 1

    Supported gates:
    x, y, z, h, s, t, cx
    """

    circuit = QuantumCircuit(
        num_qubits
    )

    lines = circuit_text.strip().splitlines()

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        # Remove extra spaces
        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        parts = line.lower().split()

        gate = parts[0]

        try:

            # ================================================
            # SINGLE-QUBIT GATES
            # ================================================

            if gate in [
                "x",
                "y",
                "z",
                "h",
                "s",
                "t"
            ]:

                if len(parts) != 2:

                    raise ValueError(
                        f"Line {line_number}: "
                        f"{gate.upper()} requires "
                        f"exactly one qubit."
                    )

                qubit = int(
                    parts[1]
                )

                if (
                    qubit < 0
                    or qubit >= num_qubits
                ):

                    raise ValueError(
                        f"Line {line_number}: "
                        f"Qubit {qubit} does not exist."
                    )

                if gate == "x":
                    circuit.x(qubit)

                elif gate == "y":
                    circuit.y(qubit)

                elif gate == "z":
                    circuit.z(qubit)

                elif gate == "h":
                    circuit.h(qubit)

                elif gate == "s":
                    circuit.s(qubit)

                elif gate == "t":
                    circuit.t(qubit)


            # ================================================
            # TWO-QUBIT CNOT
            # ================================================

            elif gate == "cx":

                if len(parts) != 3:

                    raise ValueError(
                        f"Line {line_number}: "
                        "CX requires two qubits."
                    )

                control = int(
                    parts[1]
                )

                target = int(
                    parts[2]
                )

                if (
                    control < 0
                    or control >= num_qubits
                    or target < 0
                    or target >= num_qubits
                ):

                    raise ValueError(
                        f"Line {line_number}: "
                        "Invalid qubit index."
                    )

                if control == target:

                    raise ValueError(
                        f"Line {line_number}: "
                        "Control and target qubits "
                        "cannot be the same."
                    )

                circuit.cx(
                    control,
                    target
                )


            # ================================================
            # UNKNOWN GATE
            # ================================================

            else:

                raise ValueError(
                    f"Line {line_number}: "
                    f"Unsupported gate '{gate}'."
                )


        except ValueError:
            raise

        except Exception as error:

            raise ValueError(
                f"Line {line_number}: {error}"
            )

    return circuit

# ============================================================
# USER-DEFINED CIRCUIT
# ============================================================

st.header(
    "Build Your Own Quantum Circuit"
)

st.write(
    """
    Enter one gate per line.

    Examples:

    `x 0` → X gate on qubit 0

    `h 1` → Hadamard on qubit 1

    `cx 0 1` → CNOT with q0 as control
    and q1 as target
    """
)


# ------------------------------------------------------------
# Number of qubits
# ------------------------------------------------------------

num_qubits = st.number_input(
    "Number of qubits",
    min_value=1,
    max_value=4,
    value=1,
    step=1
)


# ------------------------------------------------------------
# Circuit instructions
# ------------------------------------------------------------

circuit_text = st.text_area(
    "Circuit instructions",
    value=DEFAULT_USER_CIRCUIT,
    height=180
)


# ------------------------------------------------------------
# Supported syntax
# ------------------------------------------------------------

with st.expander(
    "Supported gate syntax"
):

    st.code(
        """
x 0
y 0
z 0
h 0
s 0
t 0
cx 0 1
        """
    )


# ------------------------------------------------------------
# Build circuit
# ------------------------------------------------------------

try:

    user_circuit = build_circuit_from_text(
        circuit_text,
        int(num_qubits)
    )

    circuit_valid = True

except ValueError as error:

    user_circuit = None

    circuit_valid = False

    st.error(
        str(error)
    )


# ------------------------------------------------------------
# Display original circuit
# ------------------------------------------------------------

def circuit_to_text(circuit):
    """
    Convert a Qiskit circuit into printable text.
    """

    return str(
        circuit.draw(
            output="text"
        )
    )


if circuit_valid:

    st.subheader(
        "Original Circuit"
    )

    st.code(
        circuit_to_text(
            user_circuit
        )
    )

    st.write(
        f"Gate count: "
        f"{len(user_circuit.data)}"
    )

# ============================================================
# CIRCUIT TEXT

if circuit_valid:

    st.subheader(
        "Original Circuit"
    )

    st.code(
        circuit_to_text(
            user_circuit
        )
    )

    st.write(
        f"Gate count: "
        f"{len(user_circuit.data)}"
    )

# LOAD MODEL
# ============================================================

try:

    model = load_cnn_model()

    model_loaded = True

except Exception as error:

    model_loaded = False

    st.error(
        f"Could not load CNN model: {error}"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚛️ Quantum Algorithm Simulator"
)

st.subheader(
    "ML-Based Quantum Circuit Optimization"
)

st.write(
    """
    This system uses a trained 1D Convolutional Neural Network
    to identify potentially redundant quantum gates.

    The ML model proposes an optimized circuit, and a quantum
    simulator checks whether the proposed optimization preserves
    the circuit's simulated quantum state.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Optimizer Settings"
)

st.sidebar.write(
    "CNN removal threshold"
)

st.sidebar.code(
    ML_THRESHOLD
)

st.sidebar.write(
    "Minimum fidelity"
)

st.sidebar.code(
    FIDELITY_THRESHOLD
)

st.sidebar.write(
    "Model"
)

st.sidebar.code(
    MODEL_PATH
)


if model_loaded:

    st.sidebar.success(
        "CNN model loaded"
    )

else:

    st.sidebar.error(
        "CNN model unavailable"
    )


# ============================================================
# DEMO SECTION
# ============================================================

st.header(
    "Quantum Circuit Demo"
)

st.write(
    """
    Start with a deliberately messy quantum circuit.
    The trained CNN will decide which gates may be removed.
    """
)



# ============================================================
# ORIGINAL CIRCUIT DISPLAY
# ============================================================

st.subheader(
    "Original Circuit"
)

st.code(
    circuit_to_text(
        user_circuit
    )
)


original_gate_count = len(
    user_circuit.data
)


# ============================================================
# OPTIMIZATION BUTTON
# ============================================================

if st.button(
    "🚀 Optimize Circuit",
    type="primary"
):

    if not model_loaded:

        st.error(
            "The trained CNN model could not be loaded."
        )

    else:

        # ====================================================
        # ML PREDICTION
        # ====================================================

        predicted_mask, probabilities = (
            predict_gate_mask(
                model,
                user_circuit,
                threshold=ML_THRESHOLD
            )
        )


        # ====================================================
        # ML PROPOSED CIRCUIT
        # ====================================================

        candidate_circuit, _, _ = (
            ml_optimize_circuit(
                user_circuit,
                model,
                threshold=ML_THRESHOLD
            )
        )


        # ====================================================
        # SAFETY VALIDATION
        # ====================================================

        (
            final_circuit,
            fidelity,
            accepted
        ) = validate_optimization(
            user_circuit,
            candidate_circuit,
            fidelity_threshold=FIDELITY_THRESHOLD
        )


        # ====================================================
        # STATISTICS
        # ====================================================

        proposed_gate_count = len(
            candidate_circuit.data
        )

        final_gate_count = len(
            final_circuit.data
        )

        gates_removed = (
            original_gate_count
            - final_gate_count
        )


        if original_gate_count > 0:

            reduction_percentage = (
                gates_removed
                / original_gate_count
            ) * 100

        else:

            reduction_percentage = 0.0


        # ====================================================
        # RESULTS
        # ====================================================

        st.header(
            "Optimization Results"
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Original Gates",
            original_gate_count
        )

        col2.metric(
            "Proposed Gates",
            proposed_gate_count
        )

        col3.metric(
            "Final Gates",
            final_gate_count
        )

        col4.metric(
            "Gate Reduction",
            f"{reduction_percentage:.2f}%"
        )


        # ====================================================
        # ML REMOVAL MASK
        # ====================================================

        st.subheader(
            "ML Gate Decisions"
        )

        st.write(
            "Removal mask:"
        )

        st.code(
            str(
                predicted_mask
            )
        )


        # ====================================================
        # PROBABILITY TABLE
        # ====================================================

        st.subheader(
            "Gate Removal Probabilities"
        )


        gate_data = []


        for index, probability in enumerate(
            probabilities
        ):

            instruction = (
                user_circuit
                .data[index]
            )

            gate_name = (
                instruction
                .operation
                .name
                .upper()
            )


            qubits = [
                user_circuit
                .find_bit(qubit)
                .index
                for qubit
                in instruction.qubits
            ]


            decision = (
                "REMOVE"
                if probability >= ML_THRESHOLD
                else "KEEP"
            )


            gate_data.append(
                {
                    "Gate Position":
                        index,

                    "Gate":
                        gate_name,

                    "Qubit":
                        str(qubits),

                    "Removal Probability":
                        round(
                            float(probability),
                            4
                        ),

                    "Decision":
                        decision
                }
            )


        st.dataframe(
            gate_data,
            use_container_width=True
        )


        # ====================================================
        # PROPOSED CIRCUIT
        # ====================================================

        st.subheader(
            "ML Proposed Circuit"
        )

        st.code(
            circuit_to_text(
                candidate_circuit
            )
        )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        st.subheader(
            "Safety Verification"
        )


        safety_col1, safety_col2 = (
            st.columns(2)
        )


        safety_col1.metric(
            "Measured Fidelity",
            f"{fidelity:.4f}"
        )


        if accepted:

            safety_col2.success(
                "✅ Optimization Accepted"
            )

        else:

            safety_col2.error(
                "❌ Optimization Rejected"
            )


        # ====================================================
        # FINAL SAFE CIRCUIT
        # ====================================================

        st.subheader(
            "Final Safe Circuit"
        )

        st.code(
            circuit_to_text(
                final_circuit
            )
        )


        # ====================================================
        # EXPLANATION
        # ====================================================

        if accepted:

            st.success(
                """
                The ML-generated optimization passed the
                fidelity safety check and was accepted.
                """
            )

        else:

            st.warning(
                """
                The ML-generated optimization did not preserve
                sufficient fidelity, so the system rejected it
                and restored the original circuit.
                """
            )


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.divider()

st.header(
    "How the System Works"
)

st.markdown(
    """
    **1. Quantum Circuit**

    The user provides or generates a quantum circuit.

    ↓

    **2. Circuit Encoding**

    Every gate is converted into ML-readable features containing
    gate type and qubit information.

    ↓

    **3. 1D CNN**

    The trained neural network analyzes local gate patterns and
    predicts a removal probability for each gate.

    ↓

    **4. ML Optimization Proposal**

    Gates with sufficiently high removal probability are removed.

    ↓

    **5. Quantum Simulation**

    The optimized circuit is compared against the original circuit.

    ↓

    **6. Safety Layer**

    If fidelity is at least 0.99, the optimization is accepted.
    Otherwise, the original circuit is restored.
    """
)


# ============================================================
# CURRENT MODEL INFORMATION
# ============================================================

st.divider()

st.header(
    "Current ML Architecture"
)

st.write(
    """
    The current prototype contains:

    - Simple Neural Network for redundancy-count prediction
    - Autoencoder for compressed circuit representations
    - 1D Convolutional Neural Network for gate-level optimization
    - Rule-based optimizer used as supervised-learning teacher
    - Quantum-state fidelity verification
    - ML safety acceptance/rejection layer
    """
)