import os
import pandas as pd
import pyblp
import numpy as np
import time 
import matplotlib.pyplot as plt


# prompt: create a dict that returns keys equal to the value of the variable and values equal to the number of times the value appears in the variable

def count_values(variable):
    """
    Creates a dictionary that returns keys equal to the value of the variable and values equal to the number of times the value appears in the variable.

    :param variable: The variable to be counted.
    :return: A dictionary with keys equal to the value of the variable and values equal to the number of times the value appears in the variable.
    """

    # Create an empty dictionary
    value_counts = {}

    # Iterate over the variable
    for value in variable:
        # Check if the value is already in the dictionary
        if value in value_counts:
        # Increment the count for the value
            value_counts[value] += 1
        else:
        # Add the value to the dictionary with a count of 1
            value_counts[value] = 1

    # Return the dictionary
    return value_counts


def preprend_zero(row):
    if len(row['zip'])<=2:
        return '0'+ row['zip']
    return row['zip']


def create_instrument_dict(product_data):
    """
    Creates a dictionary with keys as column names containing 'quad' and values as 'demand_instruments' followed by a count.

    :param product_data: The DataFrame containing product data.
    :return: A dictionary with instrument mappings.
    """
    inst_dict = {}
    count = 0
    for elem in product_data.columns:
        if 'quad' in elem:
            inst_dict[elem] = f'demand_instruments{count}'
            count += 1
    return inst_dict


def main():
    product_data = pd.concat([product_data, quad_inst], axis=1)
    inst_dict =  create_instrument_dict(product_data)
    product_data.rename(columns=inst_dict, inplace=True)
    # Baseline 
    problem = pyblp.Problem(pyblp.Formulation('1+ prices'), product_data)
    ols_results = problem.solve(method='1s')
    logit_results = problem.solve()

    # No fixed effects
    problem = pyblp.Problem(pyblp.Formulation('1+ prices + tar + co + nicotine + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig '), product_data)
    ols_results = problem.solve(method='1s')
    logit_results = problem.solve()

    # Fixed effects
    fe_problem = pyblp.Problem(pyblp.Formulation('1 + prices', absorb='C(market_ids) + C(product_ids)'), product_data)
    fe_results = fe_problem.solve(method='1s')



if __name__ == '__main__':
    main()
