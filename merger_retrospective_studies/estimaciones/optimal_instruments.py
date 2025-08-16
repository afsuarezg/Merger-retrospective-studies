import os
import pandas as pd
import pyblp
import numpy as np
import time 
import matplotlib.pyplot as plt


def results_optimal_instruments(results: pyblp.ProblemResults):
    """
    Computes optimal instruments, updates the problem with these instruments, 
    and solves the updated problem.

    Parameters:
        results (pyblp.ProblemResults): The results object from an initial problem.

    Returns:
        pyblp.ProblemResults: The results object after solving the updated problem with optimal instruments.
    """
    # Step 1: Compute optimal instruments using the specified method
    instrument_results = results.compute_optimal_instruments(method='approximate')

    # Step 2: Create a new problem instance with the computed optimal instruments
    updated_problem = instrument_results.to_problem()

    # Step 3: Solve the updated problem with the same sigma and pi as the original results
    # Use the 'bfgs' optimization method with a gradient tolerance of 1e-5, and solve using '1s' method
    updated_results = updated_problem.solve(
        sigma=results.sigma,
        pi=results.pi,
        optimization=pyblp.Optimization('bfgs', {'gtol': 1e-10}),
        method='1s'
    )

    # Return the updated results
    return updated_results


if __name__ == '__main__':
    pass