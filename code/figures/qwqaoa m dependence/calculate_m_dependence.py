import sys
sys.path.append("../../")

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import qwqaoa
import knapsack
import circuits
import optimization
import visualization


problems = knapsack.toy_problems
problem_names = knapsack.problem_names
m_values = [1, 2, 3, 10, 20]
p = 2

print("m-dependence of QWQAOA Approach")
print(f"Parameters: {p = }")
for problem, problem_name in zip(problems, problem_names):
    print(f"Problem {problem_name}:")
    for m in m_values:
        ratio = qwqaoa.approximation_ratio(problem, p, m)
        print(f"{m = }: rho = {ratio}")
