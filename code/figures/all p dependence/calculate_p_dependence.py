import sys
sys.path.append("../../")

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import linqaoa
import quadqaoa
import qwqaoa
import knapsack
import circuits
import optimization
import visualization

name = "p_dependence"
problem = knapsack.toy_problems[-1]
ps = list(range(1, 6))

# quadqaoa p-dependence
print("QuadQAOA")
a = 1
b = 2 * quadqaoa.bmin(a, problem)
ratios = []
for p in ps:
    ratio = quadqaoa.approximation_ratio(problem, p, a, b)
    ratios.append(ratio)
    print(f"{p = }: rho = {ratio}")
plt.scatter(ps, ratios, label="quad")

comment = f"""p-dependence of different QAOA approaches.

Problem: {problem}
Considered Values of p: {ps}

1. Approach with quadratic penalty.
Parameters: {a = }, {b = }.
Calculated approximation ratios: {ratios}
"""

with open(f"{name}.txt", "w") as f:
    f.write(comment)

# linqaoa p-dependece
print("LinQAOA")
a = 2 * linqaoa.amin(problem)
ratios = []
for p in ps:
    ratio = linqaoa.approximation_ratio(problem, p, a)
    ratios.append(ratio)
    print(f"{p = }: rho = {ratio}")
plt.scatter(ps, ratios, label="lin")

comment = f"""2. Approach with linear penalty.
Parameters: {a = }.
Calculated approximation ratios: {ratios}
"""

with open(f"{name}.txt", "a") as f:
    f.write(comment)

# qwqaoa p-dependence
print("QWQAOA")
m = 3
ratios = []
for p in ps:
    ratio = qwqaoa.approximation_ratio(problem, p, m)
    ratios.append(ratio)
    print(f"{p = }: rho = {ratio}")
plt.scatter(ps, ratios, label="qw")


comment = f"""3. Approach with quantum walk mixer.
Parameters: {m = }.
Calculated approximation ratios: {ratios}
"""

with open(f"{name}.txt", "a") as f:
    f.write(comment)

plt.xlabel(r"Circuit Depth $p$")
plt.ylabel(r"Approximation Ratio $\rho$")
plt.savefig(f"{name}.pdf")
plt.show()
