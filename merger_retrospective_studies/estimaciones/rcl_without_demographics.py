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


def rcl_without_demographics(product_data: pd.DataFrame, 
                blp_inst: pd.DataFrame, 
                local_inst: pd.DataFrame,
                quad_inst: pd.DataFrame,
                ):
    X1_formulation = pyblp.Formulation('0 + prices ', absorb='C(product_ids)')
    X2_formulation = pyblp.Formulation('1 + nicotine + tar ')

    product_formulations = (X1_formulation, X2_formulation)
    
    # Definición del método de integración 
    mc_integration = pyblp.Integration('monte_carlo', size=100)
    pr_integration = pyblp.Integration('product', size=25)

    # BLP 
    blp_data=pd.concat([product_data, blp_inst], axis=1)
    dict_rename = rename_instruments(blp_data)
    blp_data = blp_data.rename(columns=dict_rename)

    # Algoritmo de optimización
    bfgs = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-1})
    optimization = pyblp.Optimization('trust-constr', {'gtol': 1e-8, 'xtol': 1e-8})
    optimization = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-1})

    # Bounds
    sigma_lower = np.zeros((3,3))
    sigma_upper = np.tril(np.ones((3, 3))) * np.inf

    # Problema
    try:
        mc_problem = pyblp.Problem(product_formulations, blp_data, integration=mc_integration)
        pr_problem = pyblp.Problem(product_formulations, blp_data, integration=pr_integration)    
        # Resultados
        try:
            print('BLP 1')
            results1 = mc_problem.solve(sigma=np.ones((3, 3)), sigma_bounds=(sigma_lower, sigma_upper), optimization=optimization)
            elasticities = results1.compute_elasticities()
            diversions = results1.compute_diversion_ratios()
        except Exception as e:
            print(e)

        try:
            print('BLP 2')
            results2 = pr_problem.solve(sigma=np.ones((3, 3)), sigma_bounds=(sigma_lower, sigma_upper), optimization=optimization)
        except Exception as e:
            print(e)

        try:
            print('BLP 3')
            results3 = mc_problem.solve(sigma=np.eye(3), sigma_bounds=(sigma_lower, sigma_upper), optimization=optimization)
        except Exception as e:
            print(e)
    except Exception as e:
        print(e)

    # Local
    try:
        local_data=pd.concat([product_data, local_inst], axis=1)
        dict_rename = rename_instruments(local_data)
        local_data = local_data.rename(columns=dict_rename)
        mc_problem = pyblp.Problem(product_formulations, local_data, integration=mc_integration)
        pr_problem = pyblp.Problem(product_formulations, local_data, integration=pr_integration)
        try:
            print('LOCAL 1')
            results1 = mc_problem.solve(sigma=np.ones((3, 3)), sigma_bounds = (sigma_lower, sigma_upper),optimization=optimization)
            elasticities = results1.compute_elasticities()
            diversions = results1.compute_diversion_ratios()
        except Exception as e:
            print(e)

        try:
            print('LOCAL 2')
            results2 = pr_problem.solve(sigma=np.ones((3, 3)), sigma_bounds = (sigma_lower, sigma_upper),optimization=optimization)
        except Exception as e:
            print(e)

        try:
            print('LOCAL 3')
            results3 = mc_problem.solve(sigma=np.eye(3), sigma_bounds = (sigma_lower, sigma_upper),optimization=optimization)
        except Exception as e:
            print(e)
    except Exception as e:
        print(e)

    # Quadratic
    try:
        quad_data = pd.concat([product_data, quad_inst], axis=1)
        dict_rename = rename_instruments(quad_data)
        quad_data = quad_data.rename(columns=dict_rename)
        mc_problem = pyblp.Problem(product_formulations, quad_data, integration=mc_integration)
        pr_problem = pyblp.Problem(product_formulations, quad_data, integration=pr_integration)
        
        try:
            print('QUADRATIC 1')
            results1 = mc_problem.solve(sigma=np.ones((3, 3)), sigma_bounds=(sigma_lower, sigma_upper), optimization=optimization)
            elasticities = results1.compute_elasticities()
            diversions = results1.compute_diversion_ratios()
        except Exception as e:
            print(e)

        try:
            print('QUADRATIC 2')
            results2 = pr_problem.solve(sigma=np.ones((3, 3)), sigma_bounds=(sigma_lower, sigma_upper), optimization=optimization)
        except Exception as e:
            print(e)

        try:
            print('QUADRATIC 3')
            results3 = mc_problem.solve(sigma=np.eye(3), sigma_bounds=(sigma_lower, sigma_upper), optimization=optimization)
        except Exception as e:
            print(e)
    except Exception as e:
        print(e)



if __name__ == '__main__':
    pass