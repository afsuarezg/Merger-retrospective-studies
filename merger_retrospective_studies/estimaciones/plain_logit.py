import pandas as pd
import pyblp
# import statsmodels.formula.api as smf


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


def create_instrument_dict(product_data:pd.DataFrame):
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


def plain_logit_deprecated(product_data: pd.DataFrame, 
                inst_data: pd.DataFrame):
    product_data = pd.concat([product_data, inst_data], axis=1)
    inst_dict =  create_instrument_dict(product_data)
    
    product_data.rename(columns=inst_dict, inplace=True)    
    
    # Eliminate missing values from a specific column
    product_data = product_data.dropna(subset=['prices'])

    
    # try:
    #     print('# Plain logit without intercept')
    #     logit_formulation = pyblp.Formulation('prices', absorb='C(product_ids)')
    #     problem = pyblp.Problem(logit_formulation, product_data)
    #     logit_results = problem.solve(method='1s')
    #     print(logit_results)
    # except Exception as e:
    #     print(f"Error in plain logit without intercept: {e}")

    # try:
    #     print('# Plain logit with intercept')
    #     logit_formulation = pyblp.Formulation('1+ prices')
    #     problem = pyblp.Problem(logit_formulation, product_data)
    #     logit_results = problem.solve(method='1s')
    #     print(logit_results)
    # except Exception as e:
    #     print(f"Error in plain logit with intercept: {e}")

    try:
        print('# Logit problem with products characteristics || No fixed effects')
        logit_formulation = pyblp.Formulation('1 + prices + tar + co + nicotine')
        logit_problem = pyblp.Problem(logit_formulation, product_data)
        logit_results = logit_problem.solve(method='1s')
        print(logit_results)
    except Exception as e:
        print(f"Error in logit problem with products' characteristics: {e}")


    # Fixed effects
    # logit_formulation = pyblp.Formulation('1 + prices', absorb='C(market_ids) + C(product_ids)')
    # fe_problem = pyblp.Problem(logit_formulation, product_data)
    # fe_results = fe_problem.solve(method='1s')
    # print(fe_results)


def plain_logit(product_data=pd.DataFrame, formulation: pyblp.Formulation):
    problem= pyblp.Problem(product_formulations=formulation, 
                           product_data=product_data)
    
    logit_results=problem.solve()

    return logit_results

    




if __name__ == '__main__':
    plain_logit()
