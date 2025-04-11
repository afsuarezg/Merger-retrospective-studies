import os
import pandas as pd
import pyblp
import numpy as np
import time 
import matplotlib.pyplot as plt

from .rcl_without_demographics import rename_instruments
from .estimaciones_utils import generate_random_sparse_array, generate_random_array, generate_random_floats, create_sparse_array


def rcl_with_demographics(product_data: pd.DataFrame, agent_data: pd.DataFrame):

    # Formulación del problema sin interacción con información demográfica
    X1_formulation = pyblp.Formulation('0 + prices ', absorb='C(product_ids) + C(market_ids)')
    X1_formulation = pyblp.Formulation('0 + prices ')
    X2_formulation = pyblp.Formulation('0 + nicotine + tar + co + nicotine_mg_per_g + nicotine_mg_per_cig ')
    X2_formulation = pyblp.Formulation('0 + nicotine + tar + co + nicotine_mg_per_g')
    product_formulations = (X1_formulation, X2_formulation)

    # Algoritmo de optimización
    optimization = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-12})

    # Formulación del consumidor dentro del problema
    agent_formulation = pyblp.Formulation('0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed')

    # Definición del problema con consumidor
    problem = pyblp.Problem(
                    product_formulations,
                    product_data,
                    agent_formulation,
                    agent_data)
    
    # Valores iniciales
    initial_sigma = np.diag(generate_random_floats(4, 0,4))
    # np.diag([0.3302, 2.4526, 0.0163])
    initial_pi = generate_random_sparse_array((4,4), -5,5, 6)
    
    # Sigma bounds
    sigma_lower = np.zeros((3,3))
    sigma_lower = np.zeros(initial_sigma.shape)
    sigma_upper = np.tril(np.ones(initial_sigma.shape)) * np.inf

    # beta bounds 
    beta_lower = -100*np.ones((1,1))
    beta_upper = np.zeros((1,1))

    # Resultados del problema
    results = problem.solve(
        initial_sigma,
        initial_pi,
        optimization=optimization,
        sigma_bounds=(sigma_lower, sigma_upper),
        beta_bounds = (beta_lower, beta_upper), 
        method='1s'
    )

    return results


def rcl_with_demographics(product_data: pd.DataFrame, agent_data: pd.DataFrame):

    # Problem formulations without demographics
    X1_formulation = pyblp.Formulation('0 + prices ', absorb='C(product_ids)')
    X2_formulation = pyblp.Formulation('0+ nicotine + tar + co + nicotine_mg_per_g')
    product_formulations = (X1_formulation, X2_formulation)

    # Optimization algorithm
    optimization = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-12})

    # Consumer
    agent_formulation = pyblp.Formulation('0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed')

    # Definición del problema con consumidor
    problem = pyblp.Problem(product_formulations=product_formulations,
                            product_data=product_data,
                            agent_formulation=agent_formulation,
                            agent_data=agent_data)
    
    
    # Sigma initial values
    initial_sigma = np.diag(generate_random_floats(4, 0, 4))
    # initial_pi = generate_random_sparse_array((4,4), -5,5, 6)
    initial_pi = create_sparse_array((4,4), k=3, seed=50)
    
    # Sigma bounds
    sigma_lower = np.zeros((3,3))
    sigma_lower = np.zeros(initial_sigma.shape)
    sigma_upper = np.tril(np.ones(initial_sigma.shape)) * np.inf

    # beta bounds 
    beta_lower = -100*np.ones((1,1))
    beta_upper = np.zeros((1,1))

    # Results
    results = problem.solve(
        initial_sigma,
        initial_pi,
        optimization=optimization,
        sigma_bounds=(sigma_lower, sigma_upper),
        beta_bounds = (beta_lower, beta_upper), 
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