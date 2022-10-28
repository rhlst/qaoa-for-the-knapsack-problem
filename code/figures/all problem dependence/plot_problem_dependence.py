import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

import visualization


problem_names = ["A", "B", "C", "D", "E", "F", "G"]
quad_ratios = [
    0.7852390361080505,
    0.7852390351380075,
    0.6751294953429662,
    0.588899088194157,
    0.4010330726948032,
    0.2892650621955738,
    0.343872406269758,
]
lin_ratios = [
    0.9999999999999893,
    0.375,
    0.9999999999999983,
    0.7499999999999994,
    0.7318614732386289,
    0.7308602028854456,
    0.8464234094610489,
]
qw_ratios = [
    0.9999999999999969,
    0.9999999999999954,
    0.9999999999999951,
    0.666666666666658,
    0.965723297699008,
    0.9999999999999976,
    0.6666666666666479,
]

x = np.arange(len(problem_names))
width = 0.25

plt.bar(x - width, quad_ratios, width, label="quad")
plt.bar(x,  lin_ratios, width, label="lin")
plt.bar(x + width, qw_ratios, width, label="qw")

plt.ylabel(r"Approximation Ratio $\rho$")
plt.xlabel("Problem")
plt.xticks(x, problem_names)
plt.legend(loc="lower left")
plt.savefig("problem_dependence_new.pdf")
plt.show()
