"""Helper functions for the quadratic penalty based qaoa implementation."""
from functools import partial

import numpy as np
from qiskit import transpile

import knapsack
import circuits
import visualization
import simulation as sim
import optimization


def bmin(a, problem):
    """Return the minimum feasible value of the penalty scaling factor b."""
    return a * max(problem.values)


def bitstring_to_bits(bitstring, problem):
    """Convert a qiskit bitstring to two numpy arrays, one for each register."""
    bits = np.array(list(map(int, list(bitstring))))[::-1]
    x = bits[:problem.N]
    y = bits[problem.N:]
    return x, y


def bitstring_to_choice(bitstring, problem):
    """Convert a qiskit bitstring to a choice numpy array."""
    bits = np.array(list(map(int, list(bitstring))))[::-1]
    choice = np.array(bits[:problem.N])
    return choice


def objective_function(bitstring, problem, a, b):
    """The objective function of the quadratic penalty based approach."""
    x, y = bitstring_to_bits(bitstring, problem)
    value = x.dot(problem.values)
    penalty = ((1 - sum(y))**2
               + (y.dot(np.arange(1, problem.max_weight + 1))
                  - x.dot(problem.weights))**2)
    return a * value - b * penalty


def to_parameter_dict(angles, a, b, circuit):
    """Create a circuit specific parameter dict from given parameters.
    
    angles = np.array([gamma0, beta0, gamma1, beta1, ...])"""
    gammas = angles[0::2]
    betas = angles[1::2]
    parameters = {}
    for parameter, value in zip(circuit.betas, betas):
        parameters[parameter] = value
    for parameter, value in zip(circuit.gammas, gammas):
        parameters[parameter] = value
    parameters[circuit.a] = float(a)
    parameters[circuit.b] = float(b)
    return parameters


def get_probs_dict(circuit, problem, angles, a, b, choices_only=True):
    """Simulate circuit for given parameters and return probability dict."""
    transpiled_circuit = transpile(circuit, sim.backend)
    parameter_dict = to_parameter_dict(angles, a, b, circuit)
    statevector = sim.get_statevector(transpiled_circuit, parameter_dict)
    if choices_only:
        probs_dict = statevector.probabilities_dict(range(problem.N))
    else:
        probs_dict = statevector.probabilities_dict()
    return probs_dict


def get_expectation_value(circuit, problem, angles, a, b):
    """Return the expectation value of the objective function for given parameters."""
    probs_dict = get_probs_dict(circuit, angles, a, b)
    obj = partial(objective_function, problem=problem, a=a, b=b)
    return optimization.average_value(probs_dict, obj)


def find_optimal_angles(circuit, problem, a, b):
    """Optimize the parameters beta, gamma for given circuit and parameters."""
    transpiled_circuit = transpile(circuit, sim.backend)
    obj = partial(objective_function, problem=problem, a=a, b=b)
    angles_to_parameters = partial(to_parameter_dict, circuit=circuit, a=a, b=b)

    def angles_to_value(angles):
        parameter_dict = angles_to_parameters(angles)
        statevector = sim.get_statevector(transpiled_circuit, parameter_dict)
        probs_dict = statevector.probabilities_dict()
        value = - optimization.average_value(probs_dict, obj)
        return value

    return optimization.optimize_angles(circuit.p, angles_to_value,
                                        circuit.gamma_range(a, b),
                                        circuit.beta_range())


def comparable_objective_function(bitstring, problem):
    """An approach independent objective function"""
    choice = bitstring_to_choice(bitstring, problem)
    if knapsack.is_choice_feasible(choice, problem):
        return knapsack.value(choice, problem)
    return 0


def comparable_expectation_value(problem, p, a, b):
    """Calculate the expectation value of the approach independent objective function for given parameters."""
    circuit = circuits.QuadQAOA(problem, p)
    angles = find_optimal_angles(circuit, problem, a, b)
    probs = get_probs_dict(circuit, problem, angles, a, b)
    obj = partial(comparable_objective_function, problem=problem)
    expectation = optimization.average_value(probs, obj)
    return expectation


def approximation_ratio(problem, p, a, b):
    """Calculate the approximation ratio of the quadaqoa approach for given problem and parameters."""
    expectation = comparable_expectation_value(problem, p, a, b)
    best_known_solutions = knapsack.best_known_solutions(problem)
    choice = best_known_solutions[0]
    best_value = knapsack.value(choice, problem)
    ratio = expectation / best_value
    return ratio


def main():
    a = 1
    b = 10
    problem = knapsack.toy_problems[0]
    print(f"Problem: {problem}")
    print("Building Circuit...")
    circuit = circuits.QuadQAOA(problem, p=3)
    print("Done!")
    print("Optimizing Angles...")
    angles = find_optimal_angles(circuit, problem, a, b)
    print("Done!")
    print(f"Optimized Angles: {angles}")
    probs = get_probs_dict(circuit, problem, angles, a, b)
    print(f"Probabilities of Bitstrings: {probs}")
    visualization.hist(probs)


if __name__ == "__main__":
    main()
