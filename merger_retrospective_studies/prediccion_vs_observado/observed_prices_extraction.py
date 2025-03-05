import pandas as pd 
from typing import List

from ..nielsen_data_cleaning.product_data_creation import creating_product_data_for_comparison

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
    iter =5
    #importar la base de las predicciones
    # price_predictions = pd.read_json('/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/20140215/l-bfgs-b/price_predictions_0.json')
    # price_predictions=pd.read_json('/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/20140111/l-bfgs-b/price_predictions_0.json')
    # path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/20140208/l-bfgs-b/price_predictions_0.json' 
    path ='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/20140201/2025-03-04/l-bfgs-b/price_predictions_0.json'
    price_predictions=pd.read_json(path)
    print(f"Path of the price_predictions file: {path}")

    #obtener los códigos de los retailers para los que se generaron las comparaciones 
    dict_retailers_predictions = dict_retailers_brands(price_predictions)
    print(dict_retailers_predictions)

    #crear la base de datos con toda la información
    first_week:int = 20
    num_weeks:int = 5
    year = 2015
    product_observed_data = creating_product_data_for_comparison(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                     movements_path=f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/{year}/Movement_Files/4510_{year}/7460_{year}.tsv' ,
                                     stores_path=f'Nielsen_data/{year}/Annual_Files/stores_{year}.tsv' ,
                                     products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                     extra_attributes_path=f'Nielsen_data/{year}/Annual_Files/products_extra_{year}.tsv', 
                                     first_week=first_week,
                                     num_weeks=num_weeks)
    
    print(product_observed_data.shape)
    product_observed_data.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Pruebas/product_data_completo_{iter}.json', index=False)
    #filtrar la base de datos solo por retailers para hacer el proceso más ágil 
    product_observed_data= product_observed_data[product_observed_data['store_code_uc'].isin(dict_retailers_predictions.keys())]
    print(product_observed_data.shape)

    #filtrar la base de datos a partir de brands por retailer
    observed_prices_data_long = product_observed_data.groupby('store_code_uc').filter(lambda group: filter_observed_by_predicted_data(group, group.name, dict_retailers_predictions))
    print(observed_prices_data_long.shape)
    print(observed_prices_data_long.columns)
    # observed_prices_data_wide = long_to_wide(observed_prices_data_long, id_col='store_code_uc', time_col='week_end', value_col='prices') #TODO: confirmar las variables para pasar la base de datos de long a wide. 
    #borrar algunas columnas de observed_prices_data que no son necesarias para la comparación


    #merge de la base de datos con predicciones con la base de datos de los precios observados 
    # price_predictions_v_observed=pd.merge(price_predictions, observed_prices_data_long[['store_code_uc', 'brand_code_uc', 'prices']], how='left', on=['store_code_uc', 'brand_code_uc']) #TODO: rellenar esta función para hacer el merge

    #Salvar información
    # observed_prices_data_wide.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Pruebas/observed_prices__wide_{iter}.json', index=False)
    observed_prices_data_long.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Pruebas/observed_prices__long_{iter}.json', index=False)
