"""Helper functions for the linear penalty based qaoa implementation."""
from functools import partial

import numpy as np
from qiskit import transpile

import knapsack
import circuits
import visualization
import simulation as sim
import optimization


def amin(problem):
    """Return the minimum feasible value of the penalty scaling factor a."""
    return max(problem.values)


def bitstring_to_choice(bitstring, problem):
    """Convert a measurement bitstring to an item choice array."""
    bits = np.array(list(map(int, list(bitstring))))[::-1]
    choice = np.array(bits[:problem.N])
    return choice


def objective_function(bitstring, problem, a):
    """The objective function of the linear penalty based approach."""
    choice = bitstring_to_choice(bitstring, problem)
    value = choice.dot(problem.values)
    weight = choice.dot(problem.weights)
    if weight > problem.max_weight:
        penalty = a * (weight - problem.max_weight)
    else:
        penalty = 0
    return value - penalty


def to_parameter_dict(angles, a, circuit):
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
    return parameters


def get_probs_dict(circuit, problem, angles, a, choice_only=True):
    """Simulate circuit for given parameters and return probability dict."""
    transpiled_circuit = transpile(circuit, sim.backend)
    parameter_dict = to_parameter_dict(angles, a, circuit)
    statevector = sim.get_statevector(transpiled_circuit, parameter_dict)
    if choice_only:
        probs_dict = statevector.probabilities_dict(range(problem.N))
    else:
        probs_dict = statevector.probabilities_dict()
    return probs_dict


def get_expectation_value(circuit, problem, angles, a):
    """Return the expectation value of the objective function for given parameters."""
    probs_dict = get_probs_dict(circuit, problem, angles, a)
    obj = partial(objective_function, problem=problem, a=a)
    return optimization.average_value(probs_dict, obj)


def find_optimal_angles(circuit, problem, a):
    """Optimize the parameters beta, gamma for given circuit and parameters."""
    transpiled_circuit = transpile(circuit, sim.backend)
    obj = partial(objective_function, problem=problem, a=a)
    angles_to_parameters = partial(to_parameter_dict, circuit=circuit, a=a)

    def angles_to_value(angles):
        parameter_dict = angles_to_parameters(angles)
        statevector = sim.get_statevector(transpiled_circuit, parameter_dict)
        probs_dict = statevector.probabilities_dict()
        value = - optimization.average_value(probs_dict, obj)
        return value

    return optimization.optimize_angles(circuit.p, angles_to_value,
                                        circuit.gamma_range(a),
                                        circuit.beta_range())


def comparable_objective_function(bitstring, problem):
    """An approach independent objective function"""
    choice = bitstring_to_choice(bitstring, problem)
    if knapsack.is_choice_feasible(choice, problem):
        return knapsack.value(choice, problem)
    return 0


def comparable_expectation_value(problem, p, a):
    """Calculate the expectation value of the approach independent objective function for given parameters."""
    circuit = circuits.LinQAOA(problem, p)
    angles = find_optimal_angles(circuit, problem, a)
    probs = get_probs_dict(circuit, problem, angles, a)
    obj = partial(comparable_objective_function, problem=problem)
    expectation = optimization.average_value(probs, obj)
    return expectation


def approximation_ratio(problem, p, a):
    """Calculate the approximation ratio of the linqaoa approach for given problem and parameters."""
    expectation = comparable_expectation_value(problem, p, a)
    best_known_solutions = knapsack.best_known_solutions(problem)
    choice = best_known_solutions[0]
    best_value = knapsack.value(choice, problem)
    ratio = expectation / best_value
    return ratio


def main():
    a = 10
    problem = knapsack.toy_problems[0]
    print(problem)
    print("Building Circuit...")
    circuit = circuits.LinQAOA(problem, 2)
    print("Done!")
    print("Optimizing Angles...")
    angles = find_optimal_angles(circuit, problem, a)
    print("Done!")
    print(f"Optimized Angles: {angles}")
    probs = get_probs_dict(circuit, problem, angles, a)
    visualization.hist(probs)


if __name__ == "__main__":
    main()
