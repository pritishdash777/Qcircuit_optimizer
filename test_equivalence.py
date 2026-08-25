import numpy as np
from qiskit import QuantumCircuit
from src.circuit_utils import check_circuit_equivalence


def run_equivalence_test_suite():
    print("==================================================")
    print("RUNNING OPERATOR EQUIVALENCE TEST SUITE")
    print("==================================================\n")

    # TEST 1 — Valid X cancellation
    qc1_orig = QuantumCircuit(1)
    qc1_orig.x(0)
    qc1_orig.x(0)
    qc1_cand = QuantumCircuit(1)
    res1 = check_circuit_equivalence(qc1_orig, qc1_cand)
    assert res1["equivalent"], "Test 1 Failed!"
    print("TEST 1 (Valid X cancellation)         : PASSED (EQUIVALENT)")

    # TEST 2 — Invalid Z removal
    qc2_orig = QuantumCircuit(1)
    qc2_orig.z(0)
    qc2_cand = QuantumCircuit(1)
    res2 = check_circuit_equivalence(qc2_orig, qc2_cand)
    assert not res2["equivalent"], "Test 2 Failed!"
    print("TEST 2 (Invalid Z removal)            : PASSED (NOT EQUIVALENT)")

    # TEST 3 — Valid H cancellation
    qc3_orig = QuantumCircuit(1)
    qc3_orig.h(0)
    qc3_orig.h(0)
    qc3_cand = QuantumCircuit(1)
    res3 = check_circuit_equivalence(qc3_orig, qc3_cand)
    assert res3["equivalent"], "Test 3 Failed!"
    print("TEST 3 (Valid H cancellation)         : PASSED (EQUIVALENT)")

    # TEST 4 — Valid CNOT cancellation
    qc4_orig = QuantumCircuit(2)
    qc4_orig.cx(0, 1)
    qc4_orig.cx(0, 1)
    qc4_cand = QuantumCircuit(2)
    res4 = check_circuit_equivalence(qc4_orig, qc4_cand)
    assert res4["equivalent"], "Test 4 Failed!"
    print("TEST 4 (Valid CNOT cancellation)      : PASSED (EQUIVALENT)")

    # TEST 5 — Wrong CNOT orientation
    qc5_orig = QuantumCircuit(2)
    qc5_orig.cx(0, 1)
    qc5_cand = QuantumCircuit(2)
    qc5_cand.cx(1, 0)
    res5 = check_circuit_equivalence(qc5_orig, qc5_cand)
    assert not res5["equivalent"], "Test 5 Failed!"
    print("TEST 5 (Wrong CNOT orientation)       : PASSED (NOT EQUIVALENT)")

    # TEST 6 — Global phase difference
    # X gate vs X gate with global phase exp(i * pi/4)
    qc6_orig = QuantumCircuit(1)
    qc6_orig.x(0)
    qc6_cand = QuantumCircuit(1)
    qc6_cand.x(0)
    qc6_cand.global_phase = np.pi / 4
    res6 = check_circuit_equivalence(qc6_orig, qc6_cand)
    assert res6["equivalent"], "Test 6 Failed!"
    print("TEST 6 (Global phase difference)      : PASSED (EQUIVALENT)")

    # TEST 7 — Bell-state false positive test (Same output for |00>, different operators)
    # Z(0) on |00> produces |00>, matching Identity on |00>. But Z != I.
    qc7_orig = QuantumCircuit(2)
    qc7_orig.z(0)
    qc7_cand = QuantumCircuit(2) # Empty identity circuit
    res7 = check_circuit_equivalence(qc7_orig, qc7_cand)
    assert not res7["equivalent"], "Test 7 Failed!"
    print("TEST 7 (Bell/Zero state false positive): PASSED (NOT EQUIVALENT)")

    print("\n==================================================")
    print("ALL 7 SAFETY TEST CASES PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_equivalence_test_suite()