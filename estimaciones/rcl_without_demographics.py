import os
import pandas as pd
import pyblp
import numpy as np
import time 
import matplotlib.pyplot as plt


def rename_instruments(data):
    count = 0
    dict_rename = {}
    for i, column in enumerate(data.columns):
        if 'ins' in column:
            dict_rename[column] = f'demand_instruments{count}'
            count += 1
    return dict_rename


def main():
    X1_formulation = pyblp.Formulation('0 + prices ', absorb='C(product_ids)')
    X2_formulation = pyblp.Formulation('1 + nicotine + tar ')

    product_formulations = (X1_formulation, X2_formulation)
    
    # Definición del método de integración 
    mc_integration = pyblp.Integration('monte_carlo', size=100)
    pr_integration = pyblp.Integration('product', size=25)

    # BLP 
    blp_data=pd.concat([product_data,blp_inst], axis=1)
    dict_rename = rename_instruments(blp_data)
    blp_data = blp_data.rename(columns=dict_rename)

    # Algoritmo de optimización
    bfgs = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-1})
    optimization = pyblp.Optimization('trust-constr', {'gtol': 1e-8, 'xtol': 1e-8})
    optimization = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-1})

    # Problema
    mc_problem = pyblp.Problem(product_formulations, blp_data, integration=mc_integration)
    pr_problem = pyblp.Problem(product_formulations, blp_data, integration=pr_integration)

    # Bounds
    sigma_lower = np.zeros((3,3))
    sigma_upper = np.tril(np.ones((3, 3))) * np.inf
    
    # Resultados
    results1 = mc_problem.solve(sigma=np.ones((3, 3)), sigma_bounds = (sigma_lower, sigma_upper), optimization=optimization)

    elasticities = results1.compute_elasticities()
    diversions = results1.compute_diversion_ratios()

    results2 = pr_problem.solve(sigma=np.ones((3, 3)),sigma_bounds = (sigma_lower, sigma_upper), optimization=optimization)
    results3 = mc_problem.solve(sigma=np.eye(3),sigma_bounds = (sigma_lower, sigma_upper), optimization=optimization)

    # Local
    local_data=pd.concat([product_data, local_inst], axis=1)
    dict_rename = rename_instruments(local_data)
    local_data = local_data.rename(columns=dict_rename)
    mc_problem = pyblp.Problem(product_formulations, local_data, integration=mc_integration)
    pr_problem = pyblp.Problem(product_formulations, local_data, integration=pr_integration)
    results1 = mc_problem.solve(sigma=np.ones((3, 3)), sigma_bounds = (sigma_lower, sigma_upper),optimization=optimization)
    elasticities = results1.compute_elasticities()
    diversions = results1.compute_diversion_ratios()

    results2 = pr_problem.solve(sigma=np.ones((3, 3)), sigma_bounds = (sigma_lower, sigma_upper),optimization=optimization)
    results3 = mc_problem.solve(sigma=np.eye(3), sigma_bounds = (sigma_lower, sigma_upper),optimization=optimization)

    # Quadratic
    quad_data=pd.concat([product_data, quad_inst], axis=1)
    mc_problem = pyblp.Problem(product_formulations, quad_data, integration=mc_integration)
    dict_rename = rename_instruments(local_data)
    mc_problem = pyblp.Problem(product_formulations, quad_data, integration=mc_integration)
    pr_problem = pyblp.Problem(product_formulations, quad_data, integration=pr_integration)

    results1 = mc_problem.solve(sigma=np.ones((3, 3)), sigma_bounds = (sigma_lower, sigma_upper),optimization=bfgs)
    elasticities = results1.compute_elasticities()
    diversions = results1.compute_diversion_ratios()

    results2 = pr_problem.solve(sigma=np.ones((3, 3)),sigma_bounds = (sigma_lower, sigma_upper), optimization=bfgs)
    results3 = mc_problem.solve(sigma=np.eye(3), sigma_bounds = (sigma_lower, sigma_upper),optimization=bfgs)


if __name__ == '__main__':
    main()