"""Definitions and helper functions for circuit simulation using qiskit."""
from qiskit import Aer


backend = Aer.get_backend("aer_simulator_statevector")


def get_statevector(transpiled_circuit, parameter_dict):
    bound_circuit = transpiled_circuit.bind_parameters(parameter_dict)
    result = backend.run(bound_circuit, shots=1).result()
    statevector = result.get_statevector()
    return statevector
