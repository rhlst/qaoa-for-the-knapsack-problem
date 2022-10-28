import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

import knapsack
import linqaoa
import circuits
import visualization


name = "hist-linqaoa"

p=2
a = 10
problem = knapsack.toy_problems[0]
circuit = circuits.LinQAOA(problem, p)
angles = linqaoa.find_optimal_angles(circuit, problem, a)
probs = linqaoa.get_probs_dict(circuit, problem, angles, a)

comments = [
    f"Considered {problem}",
    f"LinQAOA circuit with {p = } and {a = }",
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
