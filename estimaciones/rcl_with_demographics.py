import os
import pandas as pd
import pyblp
import numpy as np
import time 
import matplotlib.pyplot as plt

from .rcl_without_demographics import rename_instruments




def main():
    # Bases de datos
    product_data = pd.read_csv('7.product_data_postinst_Reynolds_Lorillard_retailer_2024-12-12 04:28:28.030184.csv')
    agent_data = pd.read_csv('7.agent_data_Reynolds_Lorillard_retailer_2024-12-12 04:37:46.542282.csv')
    blp_inst = pd.read_csv('7.blp_data_Reynolds_Lorillard_retailer_2024-12-12 04:28:27.990693.csv')
    local_inst = pd.read_csv('7.local_data_Reynolds_Lorillard_retailer_2024-12-12 04:28:28.020910.csv')
    quad_inst = pd.read_csv('7.quadratic_data_Reynolds_Lorillard_retailer_2024-12-12 04:28:28.024438.csv')
    
    blp_data=pd.concat([product_data,blp_inst], axis=1)
    dict_rename = rename_instruments(blp_data)

    # Restringe la información del blp_data a aquella que tienen información del consumidor en el agent_data
    blp_data = blp_data[blp_data['market_ids'].isin(agent_data['market_ids'].unique())]

    # Formulación del problema sin interacción con información demográfica
    X1_formulation = pyblp.Formulation('0 + prices ', absorb='C(product_ids)')
    X2_formulation = pyblp.Formulation('1 + nicotine + tar ')
    product_formulations = (X1_formulation, X2_formulation)

    # Algoritmo de optimización
    optimization = pyblp.Optimization('trust-constr', {'gtol': 1e-8, 'xtol': 1e-8})
    optimization = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-1})
    optimization = pyblp.Optimization('bfgs', {'gtol': 1e-5})

    # Formulación del consumidor dentro del problema
    agent_formulation = pyblp.Formulation('0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed')

    # Definición del problema con consumidor
    problem = pyblp.Problem(
                    product_formulations,
                    blp_data,
                    agent_formulation,
                    agent_data
                )
    
    # Valores iniciales
    initial_sigma = np.diag([0.3302, 2.4526, 0.0163])
    initial_pi = np.array([
    [ 5.4819,  0,      0.2037,  0   ],
    [15.8935, 0, 0,       0],
    [-0.2506,  0,     0,  0     ]
    ])

    # Resultados del problema
    results = problem.solve(
        initial_sigma,
        initial_pi,
        optimization=optimization,
        method='1s'
    )

    # Post-estimation results
    elasticities = results.compute_elasticities()
    diversions = results.compute_diversion_ratios()
    single_market = product_data['market_ids'] == '26115_0'
    plt.colorbar(plt.matshow(elasticities[single_market]));  
    means = results.extract_diagonal_means(elasticities)
    aggregates = results.compute_aggregate_elasticities(factor=0.1)   

    # Marginal costs and mark-ups
    costs = results.compute_costs()

    # Mergers
    hhi = results.compute_hhi()
    profits = results.compute_profits(costs=costs)
    cs = results.compute_consumer_surpluses()




if __name__ == '__main__':
    main()