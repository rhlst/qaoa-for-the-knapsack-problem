import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

import knapsack
import quadqaoa
import circuits
import visualization


name = "hist-quadqaoa"

p=4
a = 1
b = 10
problem = knapsack.toy_problems[0]
circuit = circuits.QuadQAOA(problem, p)
angles = quadqaoa.find_optimal_angles(circuit, problem, a, b)
probs = quadqaoa.get_probs_dict(circuit, problem, angles, a, b)

comments = [
    f"Considered {problem}",
    f"QuadQAOA circuit with {p = }, {a = } and {b = }",
    f"Optimized angles: {angles}",
    f"Resulting Probabilities: {probs}",
]

with open(name + ".txt", "w") as f:
    for comment in comments:
        f.write(comment)
        f.write("\n")

fig, ax = plt.subplots()
visualization.hist(ax, probs)
plt.savefig(name + ".pdf")
plt.show()
