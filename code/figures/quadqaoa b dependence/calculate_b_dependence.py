import sys
sys.path.append("../../")

from functools import partial
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

import quadqaoa
import knapsack
import circuits
import visualization


problem = knapsack.toy_problems[0]
a = 1
bmin = quadqaoa.bmin(a, problem)
b_values = [Fraction(x, 5) for x in range(0, 50)]
p = 4
print("Problem A")
print(f"Parameters: {p = }, {a = }")
ratios = []
for b in b_values:
    ratio = quadqaoa.approximation_ratio(problem, p, a, b)
    ratios.append(ratio)
    print(f"{b = }: rho = {ratio}")

print("Summary:")
print(f"{b_values = }")
print(f"{ratios = }")
