import sys
sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt

import visualization


problem_names = ["A", "B", "C", "D", "E", "F", "G"]
m_values = [1, 2, 3, 10, 20]
ratios = [
    [
        0.9999999999999984,
        0.9999999999999987,
        0.9999999999999977,
        0.6666666666666647,
        0.9999999999999976,
        0.8435969761891966,
        0.9999999999999916,
    ],
    [
        0.999999999999998,
        0.9999999999999978,
        5.162960156797566e-30,
        0.9999999999999917,
        0.9999999999999956,
        0.5999999999999986,
        0.8749347634580608,
    ],
    [
        0.9999999999999969,
        0.9999999999999954,
        0.9999999999999951,
        0.666666666666658,
        0.965723297699008,
        0.9999999999999976,
        0.6666666666666479,
    ],
    [
        0.9999999999999913,
        0.9999999999999944,
        4.0669059211909415e-29,
        0.9383798801644754,
        0.9999999999999585,
        0.9383360320724783,
        0.667582318184579,
    ],
    [
        0.9999999999999871,
        0.9999999999999838,
        1.7020994391453038e-28,
        0.979741561713964,
        0.9999999999999236,
        0.8357056268321855,
        0.7536833647770674,
    ]
]

x = np.arange(len(problem_names))
width = 0.15

plt.bar(x - 2*width, ratios[0], width, label=f"m = {m_values[0]}")
plt.bar(x - width, ratios[1], width, label=f"m = {m_values[1]}")
plt.bar(x,  ratios[2], width, label=f"m = {m_values[2]}")
plt.bar(x + width, ratios[3], width, label=f"m = {m_values[3]}")
plt.bar(x + 2*width, ratios[4], width, label=f"m = {m_values[4]}")

plt.ylabel(r"Approximation Ratio $\rho$")
plt.xlabel("Problem")
plt.xticks(x, problem_names)
plt.legend(loc="lower left")
plt.savefig("m_dependence_new.pdf")
plt.show()
