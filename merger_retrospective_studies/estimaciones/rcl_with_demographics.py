import os
import pandas as pd
import pyblp
import numpy as np
import time 
import matplotlib.pyplot as plt

from .rcl_without_demographics import rename_instruments
from .utils import generate_random_sparse_array, generate_random_array, generate_random_floats, create_sparse_array


def rcl_with_demographics(product_data: pd.DataFrame, 
                          agent_data: pd.DataFrame, 
                          linear_formulation: pyblp.Formulation,
                          non_linear_formulation: pyblp.Formulation, 
                          agent_formulation:pyblp.Formulation, 
                          logit_results:pyblp.results.problem_results.ProblemResults, 
                          optimization_algorithm:str='l-bfgs-b',
                          gtol:float=1e-12, 
                          num_random:int=3,
                          seed:int=50):
    
    product_formulations = (linear_formulation, non_linear_formulation)

    # Optimization algorithm
    optimization = pyblp.Optimization(method=optimization_algorithm, 
                                      method_options= {'maxiter': 10000, 'gtol': gtol, 'ftol': 1e-12})

    # Iteration algorithm 
    iteration = pyblp.Iteration(method='squarem', method_options={'max_evaluations': 3000, 'atol': 1e-14})

    # Definición del problema con consumidor
    problem = pyblp.Problem(product_formulations=product_formulations,
                            product_data=product_data,
                            agent_formulation=agent_formulation,
                            agent_data=agent_data)

    # Sigma initial values
    initial_sigma = np.diag(generate_random_floats(num_floats=problem.K2, start_range=0, end_range=4))
    initial_pi = create_sparse_array(shape=(problem.K2,problem.D), num_random=num_random, seed=seed)
    
    # Sigma bounds
    sigma_lower = np.zeros(initial_sigma.shape)
    sigma_upper_matrix = np.tril(np.ones(initial_sigma.shape))
    sigma_upper=np.where(sigma_upper_matrix==1, np.inf, sigma_upper_matrix)

    # Results
    results = problem.solve(
        sigma=initial_sigma,
        pi=initial_pi,
        beta=logit_results.beta,
        optimization=optimization,
        iteration=iteration,
        sigma_bounds=(sigma_lower, sigma_upper),
        method='1s'
    )

    return results


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

    # beta bounds 
    beta_lower = -100*np.ones((1,1))
    beta_upper = np.zeros((1,1))

    # Step 3: Solve the updated problem with the same sigma and pi as the original results
    # Use the 'bfgs' optimization method with a gradient tolerance of 1e-5, and solve using '1s' method
    updated_results = updated_problem.solve(
        beta=results.beta,
        sigma=results.sigma,
        pi=results.pi,
        optimization=pyblp.Optimization('bfgs', {'gtol': 1e-12}),
        method='1s',
        beta_bounds=(beta_lower, beta_upper)
    )

    # Return the updated results
    return updated_results


if __name__ == '__main__':
    pass