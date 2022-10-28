import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

import visualization


p_values = [1, 2, 3, 4, 5]
quad_ratios = [0.40728525192889137, 0.4247448434749554, 0.4322726043725352, 0.343872406269758, 0.419224617486492]
lin_ratios = [0.6323973514991522, 0.8464234094610489, 0.9137193285291993, 0.9766984312138275, 0.999330989870212]
qw_ratios = [0.666666666666655, 0.6666666666666479, 0.6666666666666398, 0.8216196248662176, 0.8793545802797255]

plt.scatter(p_values, quad_ratios, marker="+", label="quad")
plt.scatter(p_values, lin_ratios, marker="x", label="lin")
plt.scatter(p_values, qw_ratios, marker="1", label="qw")
plt.xlabel(r"Circuit Depth $p$")
plt.ylabel(r"Approximation Ratio $\rho$")
plt.legend(loc = "center right")
plt.savefig("p_dependence_problem_g_new.pdf")
plt.show()

