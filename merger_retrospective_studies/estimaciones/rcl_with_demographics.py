import os
import pandas as pd
import pyblp
import numpy as np
import time 
import matplotlib.pyplot as plt

from .rcl_without_demographics import rename_instruments
from .estimaciones_utils import generate_random_sparse_array, generate_random_array, generate_random_floats


def rcl_with_demographics(product_data: pd.DataFrame,
                          blp_inst: pd.DataFrame, 
                          local_inst: pd.DataFrame, 
                          quad_inst: pd.DataFrame, 
                          agent_data: pd.DataFrame):

    print('-'*10+'RCL with demographics'+ '-'*10)                      
    
    consolidated_product_data=pd.concat([product_data,blp_inst], axis=1)
    dict_rename = rename_instruments(consolidated_product_data)
    consolidated_product_data=consolidated_product_data.rename(columns=dict_rename)

    # Restringe la información del consolidated_product_data a aquella que tienen información del consumidor en el agent_data
    consolidated_product_data = consolidated_product_data[consolidated_product_data['market_ids'].isin(agent_data['market_ids'].unique())]

    # Formulación del problema sin interacción con información demográfica
    X1_formulation = pyblp.Formulation('0 + prices ', absorb='C(product_ids) + C(market_ids)')
    X2_formulation = pyblp.Formulation('0 + nicotine + tar + co + nicotine_mg_per_g + nicotine_mg_per_cig ')
    product_formulations = (X1_formulation, X2_formulation)

    # Algoritmo de optimización
    # optimization = pyblp.Optimization('trust-constr', {'gtol': 1e-8, 'xtol': 1e-8})
    # optimization = pyblp.Optimization('bfgs', {'gtol': 1e-10})
    optimization = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-8})

    # Formulación del consumidor dentro del problema
    agent_formulation = pyblp.Formulation('0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed')

    # Definición del problema con consumidor
    problem = pyblp.Problem(
                    product_formulations,
                    consolidated_product_data,
                    agent_formulation,
                    agent_data)
    
    # Valores iniciales
    initial_sigma = np.diag(generate_random_floats(5, 0,4))
    # np.diag([0.3302, 2.4526, 0.0163])
    initial_pi = generate_random_sparse_array((5,4), -5,5, 6)
    
    # np.array([
    # [ 5.4819, 0, 0.2037, 0],
    # [15.8935, 0, 0, 0],
    # [-0.2506, 0, 0, 0]
    # ])

    # Sigma bounds
    sigma_lower = np.zeros((3,3))
    sigma_lower = np.zeros(initial_sigma.shape)
    sigma_upper = np.tril(np.ones(initial_sigma.shape)) * np.inf

    # Resultados del problema
    results = problem.solve(
        initial_sigma,
        initial_pi,
        optimization=optimization,
        sigma_bounds=(sigma_lower, sigma_upper),
        method='1s'
    )

    return results
    # # Post-estimation results
    # elasticities = results.compute_elasticities()
    # diversions = results.compute_diversion_ratios()
    # single_market = product_data['market_ids'] == '26115_0'
    # plt.colorbar(plt.matshow(elasticities[single_market]));  
    # means = results.extract_diagonal_means(elasticities)
    # aggregates = results.compute_aggregate_elasticities(factor=0.1)   

    # # Marginal costs and mark-ups
    # costs = results.compute_costs()

    # # Mergers
    # hhi = results.compute_hhi()
    # profits = results.compute_profits(costs=costs)
    # cs = results.compute_consumer_surpluses()

    print('rcl with dem completed')



if __name__ == '__main__':
    pass