import pandas as pd
import pyblp
import re
import json
import math

DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
PRODUCT_MODULE = 7460
NROWS = 20000000
YEAR = 2014
WEEKS = [20140125, 20140201]


def match_patterns(elements, patterns):
    return [el for el in elements if any(re.search(pattern, el) for pattern in patterns)]


def filter_module_code(row):
    return row['product_module_code'] == PRODUCT_MODULE


def filter_module_code(row):
    return row['product_module_code'] == PRODUCT_MODULE


def filter_market_ids(row):
    return row['market_ids'] in MARKET_IDS_FILTER


def filter_row_weeks(row):
    return row['week_end'] in WEEKS


def unit_price(row):
    """
    """    
    return row['price']/row['prmult']


def total_dollar_sales(row):
    return row['prices']*row['units']


def dataframe_total_sales_by_zip_and_time(df:pd.DataFrame)-> pd.DataFrame:
    """
    Crea un dataframe que agrega el tamaño de los mercados en total dollar sales por store_zip3.
    """ 
    return df.groupby(['market_ids'],as_index = False).agg({'total dollar sales':'sum'})


def market_ids_identifier(row)-> str:
    """
    Crea una nueva columna, market_ids, que combina la informacion de mercado a nivel espacial y temporal. 
    """
    return 'ZIP'+str(row['store_zip3'])+'WEEK'+str(row['week_end_ID'])


def shares(row):
    return row['total dollar sales'] / row['market size']


def get_brands_by_zip(df):
    """
    This function takes a pandas dataframe with 'zip' and 'brand' columns
    and returns a dictionary where keys are zip codes and values are lists of unique brands in that zip.

    Args:
        df (pandas.DataFrame): The dataframe containing 'zip' and 'brand' columns

    Returns:
        dict: Dictionary with zip codes as keys and lists of unique brands as values.
    """
    brands_by_zip = df.groupby('store_zip3')['brand_descr'].apply(list).reset_index()
    brands_by_zip['brand_descr'] = brands_by_zip['brand_descr'].apply(set).apply(list)
    return brands_by_zip.set_index('store_zip3')['brand_descr'].to_dict()


def get_brands_by_retail_store(df):
    """
    This function takes a pandas dataframe with 'zip' and 'brand' columns
    and returns a dictionary where keys are the combination of zip codes and retail stores identifiers and the 
    values are lists of unique brands in that zip.

    Args:
        df (pandas.DataFrame): The dataframe containing 'zip' and 'brand' columns

    Returns:
        dict: Dictionary with zip codes as keys and lists of unique brands as values.
    """
    brands_by_zip = df.groupby(['store_code_uc','week_end'])['brand_descr'].apply(list).reset_index()
    brands_by_zip['brand_descr'] = brands_by_zip['brand_descr'].apply(set).apply(list)
    return brands_by_zip.set_index(['store_code_uc','week_end'])['brand_descr'].to_dict()


def sort_by_brand_count(brand_dict):
    """
    This function sorts a dictionary where keys are zip codes and values are lists of brands
    based on the number of brands in each list (ascending order).

    Args:
        brand_dict (dict): Dictionary with zip codes as keys and lists of brands as values.

    Returns:
        dict: Sorted dictionary with zip codes as keys and lists of brands as values.
    """
    return dict(sorted(brand_dict.items(), key=lambda item: len(item[1]), reverse= True))


def store_dict_to_json(data, filename):
    """
    This function stores a dictionary in a JSON file.

    Args:
        data (dict): The dictionary to store.
        filename (str): The name of the file to store the data in.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)


def count_values(variable):
    """
    Creates a dictionary that returns keys equal to the value of the variable and values equal to the number of times the value appears in the variable.

    :param variable: The variable to be counted.
    :return: A dictionary with keys equal to the value of the variable and values equal to the number of times the value appears in the variable.
    """

    # Create an empty dictionary
    value_counts = {}
    value_counts['nan'] = 0
    # Iterate over the variable
    for value in variable:
        # Check if the value is already in the dictionary
        if math.isnan(value):
            value_counts['nan']+=1
        elif value in value_counts:
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


def main():
    product_data = pd.read_csv(list_of_files[-1])
    product_data = product_data.dropna(subset=['tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig'])
    
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    blp_instruments = pd.DataFrame(blp_instruments)
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    local_instruments = pyblp.build_differentiation_instruments(
    formulation,
    product_data)
    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data)
    local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])


if __name__ == '__main__':
    main()