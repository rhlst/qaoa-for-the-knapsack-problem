"""Implementations of the quantum circuits described in the thesis.

This includes ...
- a quantum fourier transform based adding circuit,
- a feasibility oracle for the knapsack problem, and
- the QAOA circuits for the three approaches.

All implementations have been kept general, in the sense that they have
been defined for arbitrary instances of the knapsack problem.
"""
from functools import partial
from itertools import product
from fractions import Fraction
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, transpile, execute
from qiskit.circuit import Parameter
import numpy as np
from knapsack import KnapsackProblem
import math


class QFT(QuantumCircuit):
    """Compute the quantum fourier transform up to ordering of qubits."""

    def __init__(self, register):
        """Initialize the Circuit."""
        super().__init__(register, name="QFT")
        for idx, qubit in reversed(list(enumerate(register))):
            super().h(qubit)
            for c_idx, control_qubit in reversed(list(enumerate(register[:idx]))):
                k = idx - c_idx + 1
                super().cp(2 * np.pi / 2**k, qubit, control_qubit)


class Add(QuantumCircuit):
    """Circuit for adding n to intermediate state."""

    def __init__(self, register, n, control=None):
        """Initialize the Circuit."""
        self.register = register
        self.control = control
        qubits = [*register, *control] if control is not None else register
        super().__init__(qubits, name=f"Add {n}")
        binary = list(map(int, reversed(bin(n)[2:])))
        for idx, value in enumerate(binary):
            if value:
                self._add_power_of_two(idx)

    def _add_power_of_two(self, k):
        """Circuit for adding 2^k to intermediate state."""
        phase_gate = super().p
        if self.control is not None:
            phase_gate = partial(super().cp, target_qubit=self.control)
        for idx, qubit in enumerate(self.register):
            l = idx + 1
            if l > k:
                m = l - k
                phase_gate(2 * np.pi / 2**m, qubit)


class WeightCalculator(QuantumCircuit):
    """Circuit for calculating the weight of an item choice."""

    def __init__(self, choice_reg, weight_reg, problem):
        """Initialize the circuit."""
        super().__init__(choice_reg, weight_reg, name="Calculate Weight")
        super().append(QFT(weight_reg).to_instruction(), weight_reg)
        for qubit, weight in zip(choice_reg, problem.weights):
            adder = Add(weight_reg, weight, control=[qubit]).to_instruction()
            super().append(adder, [*weight_reg, qubit])
        super().append(QFT(weight_reg).inverse().to_instruction(), weight_reg)


class FeasibilityOracle(QuantumCircuit):
    """Circuit for checking feasibility of a choice."""

    def __init__(self, choice_reg, weight_reg, flag_qubit, problem,
                 clean_up=True):
        """Initialize the circuit."""
        c = math.floor(math.log2(problem.max_weight)) + 1
        w0 = 2**c - problem.max_weight - 1

        subcirc = QuantumCircuit(choice_reg, weight_reg, name="")
        qft = QFT(weight_reg)
        subcirc.append(qft.to_instruction(), weight_reg)
        for qubit, weight in zip(choice_reg, problem.weights):
            adder = Add(weight_reg, weight, control=[qubit]).to_instruction()
            subcirc.append(adder, [*weight_reg, qubit])
        adder = Add(weight_reg, w0)
        subcirc.append(adder.to_instruction(), weight_reg)
        subcirc.append(qft.inverse().to_instruction(), weight_reg)

        super().__init__(choice_reg, weight_reg, flag_qubit, name="U_v")
        super().append(subcirc.to_instruction(),
                       [*choice_reg, *weight_reg])
        super().x(weight_reg[c:])
        super().mcx(weight_reg[c:], flag_qubit)
        super().x(weight_reg[c:])
        if clean_up:
            super().append(subcirc.inverse().to_instruction(),
                           [*choice_reg, *weight_reg])


class DephaseValue(QuantumCircuit):
    """Dephase Value of an item choice."""

    def __init__(self, choice_reg, problem):
        """Initialize the circuit."""
        self.gamma = Parameter("gamma")
        super().__init__(choice_reg, name="Dephase Value")
        for qubit, value in zip(choice_reg, problem.values):
            super().p(- self.gamma * value, qubit)


class LinPhaseCirc(QuantumCircuit):
    """Phase seperation circuit for QAOA with linear soft constraints."""

    def __init__(self, choice_reg, weight_reg, flag_reg, problem: KnapsackProblem):
        """Initialize the circuit."""
        c = math.floor(math.log2(problem.max_weight)) + 1
        self.a = Parameter("a")
        self.gamma = Parameter("gamma")
        super().__init__(choice_reg, weight_reg, flag_reg, name="UPhase")
        # initialize flag qubit
        super().x(flag_reg)
        # dephase value
        value_circ = DephaseValue(choice_reg, problem)
        super().append(value_circ.to_instruction({value_circ.gamma: self.gamma}),
                       choice_reg)
        # dephase penalty
        feasibility_oracle = FeasibilityOracle(choice_reg, weight_reg,
                                               flag_reg, problem,
                                               clean_up=False)
        super().append(feasibility_oracle.to_instruction(),
                       [*choice_reg, *weight_reg, flag_reg])
        for idx, qubit in enumerate(weight_reg):
            super().cp(2**idx * self.a * self.gamma, flag_reg, qubit)
        super().p(-2**c * self.a * self.gamma, flag_reg)
        super().append(feasibility_oracle.inverse().to_instruction(),
                       [*choice_reg, *weight_reg, flag_reg])


class SingleQubitQuantumWalk(QuantumCircuit):
    """Circuit for single qubit quantum walk mixing."""

    def __init__(self, choice_reg, weight_reg, flag_regs,
                 problem: KnapsackProblem, j: int):
        """Initialize the circuit."""
        flag_x, flag_neighbor, flag_both = flag_regs

        self.beta = Parameter("beta")

        super().__init__(choice_reg, weight_reg, *flag_regs,
                         name=f"SingleQubitQuantumWalk_{j=}")

        feasibility_oracle = FeasibilityOracle(choice_reg, weight_reg, flag_x,
                                               problem)

        # compute flag qubits
        super().append(feasibility_oracle.to_instruction(),
                       [*choice_reg, *weight_reg, flag_x])
        super().x(choice_reg[j])
        super().append(feasibility_oracle.to_instruction(),
                       [*choice_reg, *weight_reg, flag_neighbor])
        super().x(choice_reg[j])
        super().ccx(flag_x, flag_neighbor, flag_both)
        # mix with j-th neighbor
        super().crx(2 * self.beta, flag_both, choice_reg[j])
        # uncompute flag qubits
        super().ccx(flag_x, flag_neighbor, flag_both)
        super().x(choice_reg[j])
        super().append(feasibility_oracle.to_instruction(),
                       [*choice_reg, *weight_reg, flag_neighbor])
        super().x(choice_reg[j])
        super().append(feasibility_oracle.to_instruction(),
                       [*choice_reg, *weight_reg, flag_x])


class QuantumWalkMixer(QuantumCircuit):
    """Mixing circuit for Knapsack QAOA with hard constraints."""

    def __init__(self, choice_reg, weight_reg, flag_regs,
                 problem: KnapsackProblem, m: int):
        """Initialize the circuit."""
        flag_x, flag_neighbor, flag_both = flag_regs

        self.beta = Parameter("beta")

        super().__init__(choice_reg, weight_reg, *flag_regs,
                         name=f"QuantumWalkMixer_{m=}")
        for __ in range(m):
            for j in range(problem.N):
                jwalk = SingleQubitQuantumWalk(choice_reg, weight_reg,
                                               flag_regs, problem, j)
                super().append(jwalk.to_instruction({jwalk.beta: self.beta / m}),
                               [*choice_reg, *weight_reg, *flag_regs])


class DefaultMixer(QuantumCircuit):
    """Default Mixing Circuit for QAOA."""

    def __init__(self, register):
        """Initialize the circuit."""
        self.beta = Parameter("beta")
        super().__init__(register, name="UMix")
        super().rx(2 * self.beta, register)


class LinQAOA(QuantumCircuit):
    """QAOA Circuit for Knapsack Problem with linear soft constraints."""

    def __init__(self, problem: KnapsackProblem, p: int):
        """Initialize the circuit."""
        self.p = p
        self.betas = [Parameter(f"beta{i}") for i in range(p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(p)]
        self.a = Parameter("a")

        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1

        choice_reg = QuantumRegister(problem.N, name="choices")
        weight_reg = QuantumRegister(n, name="weight")
        flag_reg = QuantumRegister(1, name="flag")

        super().__init__(choice_reg, weight_reg, flag_reg, name=f"LinQAOA {p=}")

        phase_circ = LinPhaseCirc(choice_reg, weight_reg, flag_reg, problem)
        mix_circ = DefaultMixer(choice_reg)

        # initial state
        super().h(choice_reg)

        # alternatingly apply phase seperation circuits and mixers
        for gamma, beta in zip(self.gammas, self.betas):
            # apply phase seperation circuit
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.a: self.a,
            }
            super().append(phase_circ.to_instruction(phase_params),
                           [*choice_reg, *weight_reg, flag_reg])

            # apply mixer
            super().append(mix_circ.to_instruction({mix_circ.beta: beta}),
                           choice_reg)

        # measurement
        super().save_statevector()
        super().measure_all()

    @staticmethod
    def beta_range():
        return 0, math.pi

    @staticmethod
    def gamma_range(a):
        denominator = Fraction(a).denominator
        return 0, denominator * 2 * math.pi


class QuantumWalkQAOA(QuantumCircuit):
    """QAOA Circuit for Knapsack Problem with hard constraints."""

    def __init__(self, problem: KnapsackProblem, p: int, m: int):
        """Initialize the circuit."""
        self.p = p
        self.m = m
        self.betas = [Parameter(f"beta{i}") for i in range(p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(p)]

        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.max_weight)) + 1
        if c == n:
            n += 1

        choice_reg = QuantumRegister(problem.N, name="choice")
        weight_reg = QuantumRegister(n, name="weight")
        flag_x = QuantumRegister(1, name="v(x)")
        flag_neighbor = QuantumRegister(1, name="v(n_j(x))")
        flag_both = QuantumRegister(1, name="v_j(x)")
        flag_regs = [flag_x, flag_neighbor, flag_both]

        super().__init__(choice_reg, weight_reg, *flag_regs,
                         name=f"QuantumWalkQAOA {m=},{p=}")
        phase_circ = DephaseValue(choice_reg, problem)
        mix_circ = QuantumWalkMixer(choice_reg, weight_reg, flag_regs,
                                    problem, m)
        # start in |0>
        # alternatingly apply phase seperation circuits and mixers
        for gamma, beta in zip(self.gammas, self.betas):
            # apply phase seperation circuit
            super().append(phase_circ.to_instruction({phase_circ.gamma: gamma}),
                           choice_reg)
            # apply mixer
            super().append(mix_circ.to_instruction({mix_circ.beta: beta}),
                           [*choice_reg, *weight_reg, *flag_regs])
        # measure the state
        super().save_statevector()
        super().measure_all()

    def beta_range(self):
        return 0, self.m * math.pi

    @staticmethod
    def gamma_range():
        return 0, 2 * math.pi


class QuadPhaseCirc(QuantumCircuit):
    """Phase seperation circuit for Knapsack QAOA with quadratic soft constraints."""

    def __init__(self, choice_reg, weight_reg, problem: KnapsackProblem):
        """Initialize the circuit."""
        self.gamma = Parameter("gamma")
        self.a = Parameter("a")
        self.b = Parameter("b")
        super().__init__(choice_reg, weight_reg, name="UPhase")

        # Single-qubit rotations on choice register
        for qubit, value, weight in zip(choice_reg, problem.values, problem.weights):
            angle = self.gamma * (self.a * value - self.b * (problem.total_weight - (problem.max_weight**2 + problem.max_weight) / 2) * weight)
            super().rz(angle, qubit)

        # Single-qubit rotations on weight register
        for idx, qubit in enumerate(weight_reg):
            angle = - self.gamma * self.b * (problem.max_weight - 2
                        + (idx+1) * ((problem.max_weight**2 + problem.max_weight) / 2 - problem.total_weight))
            super().rz(angle, qubit)

        super().barrier()

        # Two-qubit rotations on choice register
        for idx1, weight1 in enumerate(problem.weights):
            for idx2, weight2 in enumerate(problem.weights[:idx1]):
                angle = - self.gamma * self.b * weight1 * weight2
                super().rzz(angle, choice_reg[idx1], choice_reg[idx2])

        # Two-qubit rotations on weight register
        for idx1, qubit1 in enumerate(weight_reg):
            for idx2, qubit2 in enumerate(weight_reg[:idx1]):
                angle = - self.gamma * self.b * (1 + (idx1 + 1) * (idx2 + 1))
                super().rzz(angle, qubit1, qubit2)

        super().barrier()

        # Common choice and weight register rotations
        for (idx1, (qubit1, weight)), (idx2, qubit2) in product(enumerate(zip(choice_reg, problem.weights)), enumerate(weight_reg)):
            angle = self.gamma * self.b * (idx2 + 1) * weight
            super().rzz(angle, qubit1, qubit2)


class QuadQAOA(QuantumCircuit):
    """QAOA Circuit for Knapsack Problem with quadratic soft constraints."""

    def __init__(self, problem: KnapsackProblem, p: int):
        """Initialize the circuit."""
        self.p = p
        self.betas = [Parameter(f"beta{i}") for i in range(p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(p)]
        self.a = Parameter("a")
        self.b = Parameter("b")

        choice_reg = QuantumRegister(problem.N, name="choice")
        weight_reg = QuantumRegister(problem.max_weight, name="weight")
        super().__init__(choice_reg, weight_reg, name=f"QuadQAOA {p=}")

        phase_circ = QuadPhaseCirc(choice_reg, weight_reg, problem)
        mix_circ = DefaultMixer([*choice_reg, *weight_reg])

        # initial state
        super().h([*choice_reg, *weight_reg])

        # alternatingly apply phase seperation circuit and mixer
        for gamma, beta in zip(self.gammas, self.betas):
            # apply phase seperation circuit
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.a: self.a,
                phase_circ.b: self.b,
            }
            super().append(phase_circ.to_instruction(phase_params),
                           [*choice_reg, *weight_reg])
            # apply mixer
            super().append(mix_circ.to_instruction({mix_circ.beta: beta}),
                           [*choice_reg, *weight_reg])

        # measurement
        super().save_statevector()
        super().measure_all()

    @staticmethod
    def beta_range():
        """Return range of values for beta."""
        return 0, math.pi

    @staticmethod
    def gamma_range(a, b):
        """Return range of values for gamma."""
        gamma_min = 0
        fraca = Fraction(a)
        fracb = Fraction(b)
        a1 = fraca.numerator
        a2 = fraca.denominator
        b1 = fracb.numerator
        b2 = fracb.numerator
        # lowest common multiple lcm(a2, b2)
        lcm = abs(a2 * b2) / math.gcd(a2, b2)
        # greatest common divisor
        gcd = math.gcd(a1, b1)
        gamma_max = lcm / gcd * 2 * math.pi
        return gamma_min, gamma_max



def main():
    problem = KnapsackProblem(values=[1, 2, 3], weights=[1, 2, 3],
                              max_weight=2)
    circ = QuadQAOA(problem, 2)
    print(circ.decompose().draw())


if __name__ == "__main__":
    main()
