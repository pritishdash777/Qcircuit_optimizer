# Quantum Circuit Optimizer

A small research-style project that explores whether machine learning can help identify redundant quantum gates — while still keeping the final optimization mathematically safe.

We built this as a learning and hackathon project using **Qiskit, PyTorch and Streamlit**. The idea is simple: let a neural network suggest which gates might be removable, but never trust that suggestion blindly. Every optimized circuit is checked against the original circuit using **operator-equivalence verification** before it is accepted.

---

## Why we built this

Quantum circuits can quickly become larger than they need to be. Extra gates increase circuit depth, execution cost and, on real hardware, the opportunity for noise.

A number of simple redundancies can be removed mathematically. For example:

```text
X → X = Identity
H → H = Identity
Z → Z = Identity
CX → CX = Identity
```

Our first thought was: if these patterns can be generated and labelled automatically, can a neural network learn to identify redundant gates on its own?

That became the starting point of this project.

The final system is not just an ML model. It is a **hybrid pipeline**:

```text
Quantum Circuit
      ↓
Circuit Encoding
      ↓
1D CNN
      ↓
Gate-removal probabilities
      ↓
ML-proposed circuit
      ↓
Operator-equivalence verification
      ↓
   ┌───────────────┐
   │               │
VERIFIED          FAILED
   │               │
Accept        Restore original
```

The neural network proposes. The mathematical verifier decides.

---

## What the project currently does

The application can:

* accept manually written quantum-gate instructions
* import **OpenQASM 2.0**
* load built-in benchmark circuits
* convert circuits into ML-readable numerical features
* predict a removal probability for each gate using a trained **1D CNN**
* construct an ML-proposed optimized circuit
* verify the complete circuit operator up to global phase
* reject unsafe ML proposals automatically
* display original, proposed and final circuits in a Streamlit dashboard
* show gate-level confidence values and model-performance metrics

The saved model currently supports circuits of up to **4 qubits** and **50 encoded gate positions**.

---

## A quick example

Input:

```text
x 0
x 0
h 0
z 0
z 0
```

Conceptually, this circuit contains:

```text
X  X  H  Z  Z
```

The two X gates cancel and the two Z gates cancel, leaving:

```text
H
```

The CNN predicts a removal probability for each position and produces a mask similar to:

```text
[1, 1, 0, 1, 1]
```

where:

```text
1 = remove
0 = keep
```

The proposed one-gate circuit is then checked against the original using full operator-equivalence verification. It is only returned to the user if that verification passes.

---

## Why we added an operator-equivalence safety layer

Earlier in development, we compared the final statevectors of two circuits starting from the zero state. That looked useful at first, but it is not enough to prove that two circuits are actually the same operation.

For example:

```text
Z|0> = |0>
I|0> = |0>
```

Both give the same result for `|0>`, but:

```text
Z ≠ I
```

So matching one output state does not prove that two complete circuits are equivalent.

The current version converts both circuits into their full Qiskit `Operator` representations and compares them **up to global phase**.

Conceptually:

```text
U_candidate ≡ e^(iφ) U_original
```

If the operators are equivalent, the optimization is accepted.

If they are not, the ML proposal is rejected and the original circuit is restored.

This is one of the most important design choices in the project because it keeps the ML model as a **proposal system**, rather than treating its prediction as automatically correct.

---

## Machine-learning pipeline

### 1. Synthetic circuit generation

We did not rely on a ready-made dataset containing gate-level redundancy labels, so we generated circuits programmatically.

`src/data_generator.py` creates circuits containing a mix of:

* normal single-qubit gates
* redundant self-inverse gate pairs
* CNOT interactions
* redundant CNOT pairs
* distractor patterns
* mixed multi-qubit sequences

The generator currently works with gates such as `X`, `Y`, `Z`, `H` and `CX`.

### 2. Automatic labels

The rule-based teacher in `src/optimizer.py` generates the expected removal mask.

For example:

```text
Circuit:  X  X  H  Z  Z
Label:    1  1  0  1  1
```

These masks become the supervised-learning targets for the CNN.

### 3. Circuit encoding

`src/encoding.py` converts each quantum gate into a numerical representation.

The CNN representation has:

```text
MAX_GATES       = 50
MAX_QUBITS      = 4
Features/gate   = 19
```

The 19 features describe:

* gate type
* first qubit
* second qubit, when present

A circuit therefore becomes a matrix with shape:

```text
(50, 19)
```

Unused gate positions are padded with zeros and ignored during training using a valid-gate mask.

### 4. 1D CNN

The gate optimizer is implemented in `src/gate_optimizer.py`.

Its main flow is:

```text
19 input features
      ↓
Conv1D: 19 → 32
      ↓
ReLU
      ↓
Conv1D: 32 → 64
      ↓
ReLU
      ↓
Conv1D: 64 → 32
      ↓
ReLU
      ↓
1×1 output convolution
      ↓
one logit per gate position
```

The model is trained with:

* **PyTorch**
* `BCEWithLogitsLoss`
* Adam optimizer
* a padding mask so empty positions do not affect the loss

At inference time, logits are passed through a sigmoid to obtain gate-removal probabilities.

The current decision threshold is:

```text
0.60
```

So:

```text
P(remove) >= 0.60  → REMOVE
P(remove) <  0.60  → KEEP
```

---

## Current evaluation results

The saved evaluation was run on **2,000 unseen test circuits**.

| Metric                          |     Result |
| ------------------------------- | ---------: |
| Gate-level accuracy             | **97.88%** |
| Precision                       | **97.64%** |
| Recall                          | **98.72%** |
| F1 score                        | **98.18%** |
| Exact-mask accuracy             | **59.10%** |
| Operator-equivalence rate       | **60.35%** |
| Safe-optimization rate          | **60.25%** |
| Overall proposed gate reduction | **58.50%** |

One important thing we learned during development is that **high gate-level accuracy does not automatically mean high whole-circuit correctness**.

A model can predict nearly every gate correctly and still remove one gate that changes the full unitary operation. That is why we report both gate-level metrics and operator-equivalence results separately.

---

## Streamlit interface

The frontend is built entirely in **Streamlit**.

It supports three input modes:

### Manual

Example:

```text
x 0
h 1
cx 0 1
z 1
```

Supported manual gates currently include:

```text
x
y
z
h
s
t
cx
```

### Benchmark

The project contains built-in benchmark helpers for circuits including:

* Bell-state circuits
* intentionally messy Bell circuits
* GHZ circuits
* intentionally messy GHZ circuits
* QFT-style circuits
* random benchmark circuits

### OpenQASM

OpenQASM 2.0 input can be pasted directly into the application and is parsed using Qiskit.

After optimization, the dashboard displays:

* original gate count
* ML-proposed gate count
* final safe gate count
* gate reduction
* original circuit
* ML-proposed circuit
* final verified circuit
* removal confidence for each gate
* operator-equivalence status
* saved model-performance metrics

---

## Project structure

```text
Qcircuit_optimizer/
│
├── app.py
├── main.py
├── test_equivalence.py
├── requirements.txt
│
├── models/
│   ├── gate_optimizer_cnn.pth
│   └── evaluation_metrics.json
│
└── src/
    ├── data_generator.py
    ├── encoding.py
    ├── gate_optimizer.py
    ├── optimizer.py
    ├── circuit_utils.py
    ├── evaluation.py
    ├── model_utils.py
    ├── autoencoder.py
    ├── ae_dataset.py
    ├── ml_models.py
    ├── benchmarks.py
    └── simulator.py
```

### Important files

| File                    | What it is responsible for                               |
| ----------------------- | -------------------------------------------------------- |
| `app.py`                | Streamlit frontend and user interaction                  |
| `main.py`               | training, model loading, testing and evaluation pipeline |
| `src/data_generator.py` | synthetic quantum-circuit generation                     |
| `src/encoding.py`       | circuit-to-ML numerical encoding                         |
| `src/gate_optimizer.py` | CNN architecture, training and inference                 |
| `src/optimizer.py`      | rule-based labels and optimization safety wrapper        |
| `src/circuit_utils.py`  | circuit utilities and operator-equivalence checking      |
| `src/evaluation.py`     | accuracy, F1, exact-mask and safety evaluation           |
| `src/model_utils.py`    | saving/loading model weights and metrics                 |
| `src/benchmarks.py`     | built-in example and benchmark circuits                  |
| `test_equivalence.py`   | regression tests for the safety verifier                 |

The autoencoder and simple redundancy-count neural network are experimental parts of the repository and are not required by the main Streamlit inference path.

---

## Running the project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Qcircuit_optimizer
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If Streamlit is not already included in your environment:

```bash
pip install streamlit
```

### 4. Start the frontend

```bash
python3 -m streamlit run app.py
```

Streamlit will normally open:

```text
http://localhost:8501
```

---

## Training the CNN again

The repository already contains a saved model:

```text
models/gate_optimizer_cnn.pth
```

so normal frontend use does **not** require retraining.

Training and evaluation are controlled from `main.py`.

The current project configuration uses:

```text
10,000 generated CNN circuits
80% training
20% testing
200 CNN epochs
random seed = 42
```

The saved model is loaded when training is disabled.

---

## Safety tests

The repository includes:

```texts
test_equivalence.py
```

Run it with:

```bash
python3 test_equivalence.py
```

The tests cover cases such as:

* valid X-X cancellation
* valid H-H cancellation
* valid CX-CX cancellation
* incorrect Z removal
* reversed CNOT direction
* global-phase equivalence
* cases that would have fooled the old single-state fidelity check

---

## What we know is still limited

This is a prototype, and we would rather state its limitations clearly than pretend it solves every form of quantum-circuit optimization.

### Fixed circuit size

The saved CNN currently works with:

```text
4 qubits
50 gate positions
```

### Synthetic training data

The current evaluation mainly measures performance on circuits produced by our own synthetic generator.

### Local rule-based labels

The teacher mainly identifies adjacent identical self-inverse gate pairs. It does not yet perform general commutation analysis or discover every possible circuit identity.

### Local CNN architecture

A 1D CNN is good at nearby gate patterns, but quantum circuits can contain important long-range dependencies.

A graph-based representation and GNN would be a natural future direction.

### Parameterized gates

The encoding can identify gates such as `rx`, `ry` and `rz`, but continuous rotation angles are not currently represented in the CNN features or optimized using parameter identities.

### Exact verification does not scale forever

Building the complete unitary matrix grows exponentially with the number of qubits.

That is completely manageable for the small circuits used in this prototype, but larger systems would require a more scalable equivalence-checking approach.

### No real-hardware optimization yet

The current project works with Qiskit circuits and local mathematical verification. Hardware-aware optimization, noise models and execution on real quantum hardware are future extensions.

---

## Where we would take this next

If we continue developing the project, the areas we are most interested in are:

* graph/DAG representations of quantum circuits
* Graph Neural Networks
* long-range gate dependencies
* commutation-aware labels
* parameterized gate optimization
* larger and more realistic benchmark datasets
* hardware-aware optimization
* scalable circuit-equivalence checking
* execution on real quantum hardware

---

## Tech stack

* **Python**
* **Qiskit**
* **PyTorch**
* **NumPy**
* **Streamlit**

---

## Team

### Pritish Dash

**Team Leader**

### Tarun Ku. Choudhury

**Team Member**

---

## Final note

This project changed quite a bit while we were building it.

It started as a straightforward idea of using machine learning to remove redundant quantum gates. During testing, we realised that getting a high ML accuracy was not enough — a single wrong gate removal could change the meaning of an entire quantum circuit.

That pushed us toward the part of the project we now consider the most important:

> **let machine learning propose the optimization, but let mathematics verify it.**

That is the principle behind the current version of the Quantum Circuit Optimizer.
