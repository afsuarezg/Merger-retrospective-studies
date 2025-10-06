import pandas as pd 
from typing import List

import os

from ..nielsen_data_cleaning.product_data_creation import creating_product_data_for_comparison
from .datos_comparacion import creating_comparison_product_data_rcl


def list_retailers_with_predictions(data: pd.DataFrame)-> List:
    """
    Extracts a list of unique retailer store codes from the given DataFrame.
    Args:
        data (pd.DataFrame): A DataFrame containing a column 'store_code_uc' with retailer store codes.
    Returns:
        List: A list of unique retailer store codes.
    """
    retailers_list = set(data['store_code_uc'])
    return retailers_list


def dict_retailers_brands(data)-> dict:
    """
    Generates a dictionary mapping each retailer (store_code_uc) to a list of unique brands (brand_code_uc) they carry.
    Args:
        data (pd.DataFrame): A pandas DataFrame containing at least two columns: 'store_code_uc' and 'brand_code_uc'.
    Returns:
        dict: A dictionary where the keys are retailer codes (store_code_uc) and the values are lists of unique brand codes (brand_code_uc) associated with each retailer.
    """
    return data.groupby('store_code_uc')['brand_code_uc'].unique().to_dict()


def filter_observed_by_predicted_data(group, key, reference_dict) -> pd.DataFrame:
    """
    Filters the observed data by comparing it with the predicted data.
    This function checks if all elements in the reference dictionary's list 
    for a given key are present in the group's list for the same key.
    Args:
        group (pd.DataFrame): The DataFrame containing the observed data.
        key (str): The key to be used for comparison.
        reference_dict (dict): A dictionary containing the predicted data.
    Returns:
        pd.DataFrame: A DataFrame indicating whether the reference data is a subset of the group data.
    """

    return set(reference_dict[key]).issubset(set(group['brand_code_uc']))


def filter_observed_by_predicted_data(group, key1, key2, reference_dict) -> pd.DataFrame:
    """
    Filters the observed data by comparing it with the predicted data.
    This function checks if all elements in the reference dictionary's list 
    for a given key are present in the group's list for the same key.
    Args:
        group (pd.DataFrame): The DataFrame containing the observed data.
        key (str): The key to be used for comparison.
        reference_dict (dict): A dictionary containing the predicted data.
    Returns:
        pd.DataFrame: A DataFrame indicating whether the reference data is a subset of the group data.
    """
    if (key1 not in reference_dict.keys()):
        return False
    else:
        if key2 in reference_dict[key1]:
            return True
        else: 
            return False


def long_to_wide(df, id_col, time_col, value_col):
    """
    Transforms a long-format DataFrame into a wide-format DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame in long format.
        id_col (str): The name of the ID column.
        time_col (str): The name of the time column.
        value_col (str): The name of the column containing the values to pivot.

    Returns:
        pd.DataFrame: The DataFrame in wide format.
    """

    wide_df = df.pivot(index=id_col, columns=time_col, values=value_col)
    wide_df = wide_df.reset_index() # make the id column a regular column
    return wide_df


def main():
    #crear la base de datos con toda la información
    num_weeks:int = 10
    year = 2014

    for first_week in range(35, 35 + num_weeks):
        print(f'Processing week: {first_week}')
        product_observed_data = creating_comparison_product_data_rcl(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                        movements_path=f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/{year}/Movement_Files/4510_{year}/7460_{year}.tsv' ,
                                        stores_path=f'Nielsen_data/{year}/Annual_Files/stores_{year}.tsv' ,
                                        products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                        first_week=first_week,
                                        num_weeks=1)
    
        week=product_observed_data['week_end'].iloc[0]

        observed_folder = f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Observed/{week}'
        os.makedirs(observed_folder, exist_ok=True)

        product_observed_data.to_json(f'{observed_folder}/product_data_completo_{week}.json', index=False)
        print(f'Product data saved to: {observed_folder}/product_data_completo_{week}.json')

