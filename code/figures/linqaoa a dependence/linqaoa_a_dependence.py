import sys
sys.path.append("../../")

from functools import partial
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

import linqaoa
import knapsack
import circuits
import optimization
import visualization


name = "linqaoa a dependence"
problem = knapsack.toy_problems[-1]
amin = linqaoa.amin(problem)
a_values = [Fraction(2 * x, 10) for x in range(0, 50)]

# linqaoa p-dependece
p = 3
ratios = []
for a in a_values:
    ratio = linqaoa.approximation_ratio(problem, p, a)
    ratios.append(ratio)
    print(f"{a = }: rho = {ratio}")

comment = f"""a-dependence of LinQAOA approach.

Problem: {problem}
Minimum good value of a: {amin}
Considered Values of a: {a_values}
Parameters: {p = }.
Calculated approximation ratios: {ratios}
"""

with open(f"{name}.txt", "w") as f:
    f.write(comment)

plt.scatter(a_values, ratios)
plt.xlabel(r"Penalty Scaling Factor $a$")
plt.ylabel(r"Approximation Ratio $\rho$")
plt.savefig(f"{name}.pdf")
plt.show()
