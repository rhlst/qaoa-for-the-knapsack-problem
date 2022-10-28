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


print("Calculate approximation ratios for different problem instances and for all three approaches.")

name = "problem_dependence"
log_filename = f"{name}.txt"
plot_filename = f"{name}.pdf"

problems = knapsack.toy_problems
problem_names = knapsack.problem_names

print("1. QuadQAOA")
p = 4
a = 1
print(f"Parameters: {p = }, {a = }")
for problem, problem_name in zip(problems, problem_names):
    b = 2 * quadqaoa.bmin(a, problem)
    ratio = quadqaoa.approximation_ratio(problem, p, a, b)
    print(f"Problem {problem_name}: {b = }, rho = {ratio}")

print("2. LinQAOA")
p = 2
print(f"Parameters: {p = }")
for problem, problem_name in zip(problems, problem_names):
    a = 2 * linqaoa.amin(problem)
    ratio = linqaoa.approximation_ratio(problem, p, a)
    print(f"Problem {problem_name}: {a = }, rho = {ratio}")

print("3. QWQAOA")
p = 2
m = 3
print(f"Parameters: {p = }, {m = }")
for problem, problem_name in zip(problems, problem_names):
    ratio = qwqaoa.approximation_ratio(problem, p, m)
    print(f"Problem {problem_name}: rho = {ratio}")
