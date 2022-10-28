"""Definitions related to the knapsack problem."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class KnapsackProblem:
    """
    Class for representing particular instances of the knapsack problem.
    
    Intended only for 0-1 integer knapsack problems, i.e. values and weights
    are supposed to be of same length and must contain integer values. While
    it is possible for this class to represent other versions of the knapsack
    problem, all the other code is written with this specific subtype of the
    problem in mind and will not work for non-integers.
    
    Attributes:
    values (list): the values of the items (use int values only!)
    weights (list): the weights of the items (use int weights only!)
    max_weight (int): the maximum weight (carry capacity) of the knapsack
    total_weight (int): the sum of all weights
    N (int): the number of items
    """
    
    values: list
    weights: list
    max_weight: int
        
    def __post_init__(self):
        self.total_weight = sum(self.weights)
        self.N = len(self.weights)


def value(choice, problem):
    """Return the value of an item choice.
    
    Assumes choice is a numpy array of length problem.N"""
    return choice.dot(problem.values)


def weight(choice, problem):
    """Return the weight of an item choice.
    
    Assumes choice is a numpy array of length problem.N"""
    return choice.dot(problem.weights)


def is_choice_feasible(choice, problem):
    """Returns whether an item choice is feasible.
    
    Assumes choice is a numpy array of length problem.N"""
    return weight(choice, problem) <= problem.max_weight


# The problem instances used for numerical simulation
toy_problems = [
    KnapsackProblem(values=[1, 2], weights=[1, 1], max_weight=1),
    KnapsackProblem(values=[2, 1], weights=[1, 1], max_weight=1),
    KnapsackProblem(values=[1, 2], weights=[1, 1], max_weight=2),
    KnapsackProblem(values=[1, 1, 2], weights=[1, 1, 1], max_weight=2),
    KnapsackProblem(values=[1, 2, 4], weights=[1, 2, 3], max_weight=3),
    KnapsackProblem(values=[2, 3, 5], weights=[2, 2, 2], max_weight=3),
    KnapsackProblem(values=[1, 2, 1, 3], weights=[1, 2, 2, 1], max_weight=4),
]

# The names of the problems, as used in the thesis
problem_names = ["A", "B", "C", "D", "E", "F", "G"]


def best_known_solutions(problem: KnapsackProblem):
    """Calculate the best known solutions of a problem instance.
    
    Returns a list of item choices, represented by numpy arrays of length N with entries 0 and 1.
    Uses an inefficient algorithm, but this is good enough for the small instances used here."""
    def choices_from_number(problem, number):
        return np.array(list(map(int, list(reversed(bin(number)[2:])))) + [0] * (problem.N - len(bin(number)[2:])))
    best = 0
    solutions = []
    for i in range(2**problem.N):
        choice = choices_from_number(problem, i)
        value = choice.dot(problem.values)
        weight = choice.dot(problem.weights)
        is_legal = weight <= problem.max_weight
        if is_legal and value > best:
            best = value
            solutions = [choice]
        elif is_legal and value == best:
            solutions.append(choice)
    return np.array(solutions)
