import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

import knapsack
import qwqaoa
import circuits
import visualization


name = "hist-qwqaoa"

p=2
m = 3
problem = knapsack.toy_problems[0]
circuit = circuits.QuantumWalkQAOA(problem, p, m)
angles = qwqaoa.find_optimal_angles(circuit, problem)
probs = qwqaoa.get_probs_dict(circuit, problem, angles)

comments = [
    f"Considered {problem}",
    f"QuantumWalkQAOA circuit with {p = } and {m = }",
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
