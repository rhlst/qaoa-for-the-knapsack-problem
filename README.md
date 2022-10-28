# QAOA for the Knapsack Problem
This repository contains the code I wrote for my bachelor's thesis on "QAOA for the Knapsack Problem" [1].

In this code I implement three approaches to QAOA [2] for the knapsack problem in Python using the Qiskit library [3]:
1. Soft constraints using quadratic penalty ("quadqaoa") [4]
2. Soft constraints using linear penalty ("linqaoa") [5]
3. Hard constraints using a quantum walk mixer ("qwqaoa") [6]

For a detailed discussion of these approaches I would like to refer to my thesis. Further, these approaches are simulated for different problems and for different parameter values. The results are plotted using Matplotlib [7] and discussed in my thesis.

## Repository Structure
The repository is structured as follows:
- `code/` - The code basis of this project.
- `tests/` - The unit tests for the code. There are only a few unit tests due to time limitations in the creation of this project.
- `LICENSE` - The license file of this project. Be sure to read the license before using this code for your own project.

The `code/` directory:
- `figures/` - Code related to generating different figures, i.e. the numerical results presented in my thesis. 
- `knapsack.py` - Definition of a KnapsackProblem class and directly related helper functions.
- `circuits.py` - Implementations of the necessary quantum circuits. In particular, the implementation of a QFT adder based feasibility oracle for the knapsack problem and the implementations of the QAOA circuits corresponding to the different approaches mentioned above.
- `simulation.py` - Helper function for simulating circuits.
- `optimization.py` - Helper functions for optimizing the parameters $\beta$ and $\gamma$. For this the SHGO[8] algorithm from SciPy[9] is used.
- `linqaoa.py`, `quadqaoa.py`, `qwqaoa.py` - Functions for optimizing the parameters $\beta$ and $\gamma$ specific to the approaches and required helper functions such as objective functions.
- `visualization.py` - Definitions for consistent presentation of results.

## General Notes on Code Quality
This code came into existence as a necessity for my thesis and not as a project in and of itself. While I tried to write readable, maintainable and expandable code, I know that I came short of this goal. This is particularly true for the code in the `figures/` directory. Due to limited time and resources, I also presumably won't be able to change this in the (near) future.

## Contact
Feel free to write me an email if you have any questions or would like to use the code for your own projects (rholst@mailbox.org). I would be glad to help.

## References
[1] Jan Rasmus Holst. “QAOA for the Knapsack Problem”. Bachelor's Thesis. Leibniz University Hanover, 2022.

[2] Edward Farhi, Jeffrey Goldstone, and Sam Gutmann. A Quantum Approximate Optimization Algorithm. Nov. 14, 2014. arXiv: 1411.4028.

[3] M. D. Sajid Anis et al. Qiskit: An Open-source Framework for Quantum Computing. 2021. doi: 10.5281/zenodo.2573505.

[4] This approach is covered by multiple sources. The description in my thesis is heavily based on Christoph Roch et al. Cross Entropy Hyperparameter Optimization for Constrained Problem Hamiltonians Applied to QAOA. Aug. 20, 2020. arXiv: 2003.05292v2.

[5] This is an improved version of the approach described in Pierre Dupuy de la Grand’rive and Jean-Francois Hullo. Knapsack Problem variants of QAOA for battery revenue optimisation. Aug. 15, 2019. arXiv:1908.02210.

[6] This approach is an adaption for the knapsack problem based on an improved version of S. Marsh and J. B. Wang. “A quantum walk-assisted approximate algorithm for bounded NP optimisation problems”. In: Quantum Information Processing 18.3 (Mar. 2019), p. 61. doi: 10.1007/s11128-019-2171-3.

[7] J. D. Hunter. “Matplotlib: A 2D graphics environment”. In: Computing in Science & Engineering 9.3 (2007), pp. 90–95. doi: 10.1109/MCSE.2007.55.

[8] Stefan C. Endres, Carl Sandrock, and Walter W. Focke. “A simplicial homology algorithm for Lipschitz optimisation”. In: Journal of Global Optimization 72.2 (Oct. 2018), pp. 181–217. doi: 10.1007/s10898-018-0645-y.

[9] Pauli Virtanen et al. “SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python”. In: Nature Methods 17 (2020), pp. 261–272. doi: 10.1038/s41592-019-0686-2.
