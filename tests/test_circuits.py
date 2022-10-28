import sys
sys.path.append("../code/")

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, transpile, execute
from circuits import QFT, Add, WeightCalculator, FeasibilityOracle
from knapsack import KnapsackProblem


def test_qft_adder():
    my_control = QuantumRegister(1)
    my_register = QuantumRegister(5)
    my_circuit = QuantumCircuit(my_register, my_control)
    my_qft = QFT(my_register)
    my_circuit.append(my_qft.to_instruction(), my_register)
    my_circuit.append(Add(my_register, 5).to_instruction(), my_register)
    my_circuit.x(my_control)
    my_circuit.append(Add(my_register, 6, control=my_control).to_instruction(),
                      [*my_register, my_control])
    my_circuit.append(my_qft.inverse().to_instruction(), my_register)
    my_circuit.measure_all()

    backend = Aer.get_backend("aer_simulator")
    transpiled_circuit = transpile(my_circuit, backend)
    job = execute(transpiled_circuit, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()

    assert counts == {'101011': 1024}


def test_weight_calculator():
    problem = KnapsackProblem(values=[1, 2, 3], weights=[1, 2, 3],
                              max_weight=2)
    choice_reg = QuantumRegister(3)
    weight_reg = QuantumRegister(5)
    result_reg = ClassicalRegister(5)
    circ = QuantumCircuit(choice_reg, weight_reg, result_reg)
    circ.x(choice_reg)
    circ.append(WeightCalculator(choice_reg, weight_reg, problem).to_instruction(),
                [*choice_reg, *weight_reg])
    circ.measure(weight_reg, result_reg)

    backend = Aer.get_backend("aer_simulator")
    transpiled_circuit = transpile(circ, backend)
    job = execute(transpiled_circuit, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()

    assert counts == {"00110": 1024}


def run_feasibility_oracle(choices):
    problem = KnapsackProblem(values=[1, 2, 3], weights=[1, 2, 3],
                              max_weight=2)

    choice_reg = QuantumRegister(3)
    weight_reg = QuantumRegister(5)
    flag_reg = QuantumRegister(1)
    result_reg = ClassicalRegister(6)

    circ = QuantumCircuit(choice_reg, weight_reg, flag_reg, result_reg)
    for idx, choice in enumerate(choices):
        if choice:
            circ.x(choice_reg[idx])
    circ.append(FeasibilityOracle(choice_reg, weight_reg, flag_reg, problem),
                [*choice_reg, *weight_reg, flag_reg])
    circ.measure(weight_reg, result_reg[:-1])
    circ.measure(flag_reg, result_reg[-1])

    backend = Aer.get_backend("aer_simulator")
    transpiled_circuit = transpile(circ, backend)
    job = execute(transpiled_circuit, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()

    return counts


def test_feasibility_oracle():
    feasible_dict = {"100000": 1024}
    non_feasible_dict = {"000000": 1024}
    assert run_feasibility_oracle([0, 0, 0]) == feasible_dict
    assert run_feasibility_oracle([1, 0, 0]) == feasible_dict
    assert run_feasibility_oracle([0, 1, 0]) == feasible_dict
    assert run_feasibility_oracle([1, 1, 0]) == non_feasible_dict
    assert run_feasibility_oracle([0, 0, 1]) == non_feasible_dict
    assert run_feasibility_oracle([0, 1, 1]) == non_feasible_dict
    assert run_feasibility_oracle([1, 1, 1]) == non_feasible_dict
