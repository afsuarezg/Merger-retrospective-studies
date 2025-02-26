import os
import pandas as pd
import re
import json
import datetime
from functools import partial 
from typing import Callable


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
PRODUCT_MODULE = 7460
NROWS = 20000000
YEAR = 2014
WEEKS = [20140125, 20140201]
WEEKS = [20130105, 20130112]
WEEKS = [20130105]

# WEEKS = [20140125]

def test():
    return 'test test'


def match_patterns(elements, patterns):
    """
    Filters a list of elements by matching them against a list of patterns.
    Args:
        elements (list of str): The list of elements to be filtered.
        patterns (list of str): The list of regex patterns to match against the elements.
    Returns:
        list of str: A list of elements that match any of the given patterns.
    """

    return [el for el in elements if any(re.search(pattern, el) for pattern in patterns)]


def filter_module_code(row):
    """
    Filters a row based on the product module code.
    Args:
        row (dict): A dictionary representing a row of data with a 'product_module_code' key.
    Returns:
        bool: True if the 'product_module_code' in the row matches the global PRODUCT_MODULE, False otherwise.
    """

    return row['product_module_code'] == PRODUCT_MODULE


def filter_module_code(row):
    """
    Filters a row based on the product module code.
    Args:
        row (dict): A dictionary representing a row of data, which must contain a key 'product_module_code'.
    Returns:
        bool: True if the 'product_module_code' in the row matches the global variable PRODUCT_MODULE, False otherwise.
    """

    return row['product_module_code'] == PRODUCT_MODULE


def filter_market_ids(row):
    """
    Filters rows based on market IDs.
    Args:
        row (dict): A dictionary representing a row of data, which must contain a 'market_ids' key.
    Returns:
        bool: True if the 'market_ids' value in the row is in the MARKET_IDS_FILTER set, False otherwise.
    """

    return row['market_ids'] in MARKET_IDS_FILTER


def filter_row_weeks(row, weeks: list):    
    """
    Checks if the 'week_end' value in the given row is present in the specified list of weeks.
    Args:
        row (dict): A dictionary representing a row of data, which must contain a 'week_end' key.
        weeks (list): A list of week_end values to filter against.
    Returns:
        bool: True if the 'week_end' value in the row is in the list of weeks, False otherwise.   
    """

    return row['week_end'] in weeks


# def filter_row_weeks(row):
#     return row['week_end'] in WEEKS


def unit_price(row):
    """
    Calculate the unit price of an item.
    This function takes a row from a DataFrame and calculates the unit price
    by dividing the 'price' by the 'prmult' value.
    Parameters:
    row (pd.Series): A row from a DataFrame containing 'price' and 'prmult' columns.
    Returns:
    float: The unit price of the item.
    """

    return row['price']/row['prmult']


def total_dollar_sales(row):
    """
    Calculate the total dollar sales for a given row.
    Args:
        row (pd.Series): A pandas Series containing 'prices' and 'units' columns.
    Returns:
        float: The total dollar sales calculated as the product of 'prices' and 'units'.
    """

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
    return 'M'+str(row['store_zip3'])+'W'+str(row['week_end_ID'])
#     return str(row['store_zip3'])+'-'+str(row['week_end_ID'])


def retail_market_ids_identifier(row)-> str:
    """"

    Generates a new column, market_ids, that combines spatial and temporal market information.
    Args:
        row (pd.Series): A pandas Series object representing a row of data, 
                            which must contain 'store_code_uc' and 'week_end_ID' keys.
    Returns:
        str: A string that concatenates 'R' with the store code and 'W' with the week end ID.
"""
    return 'R'+str(row['store_code_uc'])+'W'+str(row['week_end_ID'])
#     return str(row['store_zip3'])+'-'+str(row['week_end_ID'])


def market_ids_fips(row) -> str:
    """
    Generates a market ID string based on FIPS state code, FIPS county code, and week end ID.
    Args:
        row (pd.Series): A pandas Series containing the following keys:
            - 'fips_state_code': The FIPS state code.
            - 'fips_county_code': The FIPS county code.
            - 'week_end_ID': The week end ID.
    Returns:
        str: A string in the format 'M{fips_state_code}.{fips_county_code}W{week_end_ID}'.
    """

    return 'M'+str(row['fips_state_code'])+'.'+str(row['fips_county_code'])+'W'+str(row['week_end_ID'])
#     return str(row['fips_state_code'])+'.'+str(row['fips_county_code'])+'-'+str(row['week_end_ID'])


def retail_market_ids_fips(row) -> str:
    """
    Generates a retail market ID string based on the store code and week end ID from the given row.
    Args:
        row (pd.Series): A pandas Series object containing 'store_code_uc' and 'week_end_ID' keys.
    Returns:
        str: A string in the format 'R{store_code_uc}W{week_end_ID}'.
    """

    return 'R'+str(row['store_code_uc'])+'W'+str(row['week_end_ID'])
#     return str(row['fips_state_code'])+'.'+str(row['fips_county_code'])+'-'+str(row['week_end_ID'])


def shares(row):
    """
    Calculate the market share of a product.
    This function takes a row from a DataFrame and calculates the market share
    by dividing the total dollar sales by the market size.
    Args:
        row (pd.Series): A row from a DataFrame containing 'total dollar sales' 
                         and 'market size' columns.
    Returns:
        float: The market share of the product.
    """

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


def movements_file(movements_path: str, filter_row_weeks: Callable, first_week: int=0, num_weeks: int=1):
    """
    Reads a movements file, filters it based on specified weeks, and returns the filtered DataFrame.
    Args:
        movements_path (str): The file path to the movements file.
        filter_row_weeks (Callable): A function to filter rows based on weeks.
        first_week (int, optional): The starting week index for filtering. Defaults to 0.
        num_weeks (int, optional): The number of weeks to include in the filter. Defaults to 1.
    Returns:
        pd.DataFrame: The filtered movements DataFrame containing columns 'store_code_uc', 'upc', 'week_end', 'units', 'prmult', and 'price'.
    """

    # movements_file = pd.read_csv(f'Nielsen_data/2013/Movement_Files/4510_2013/7460_2013.tsv', sep  = '\t', header = 0, index_col = None)
    movements_file = pd.read_csv(filepath_or_buffer=movements_path, sep  = '\t', header = 0, index_col = None)
    print(movements_file.shape)
    movements_file = movements_file[['store_code_uc', 'upc', 'week_end', 'units', 'prmult', 'price']]
    weeks = list(sorted(set(movements_file['week_end']))[first_week:first_week+num_weeks])
    # Apply the filtering function
    movements_file =movements_file[movements_file['week_end'].isin(weeks)]
    # movements_file = movements_file[movements_file.apply(lambda row: filter_row_weeks(row, weeks), axis=1)]

    # weeks_filter_partial = partial(filter_row_weeks, weeks)
    # movements_file = movements_file[movements_file.apply(weeks_filter_partial, axis=1)]

    return movements_file


def stores_file(stores_path: str):
    """
    Reads a TSV file containing store information and returns a DataFrame with selected columns.
    Args:
        stores_path (str): The file path to the TSV file containing store data.
    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'store_code_uc': Unique store code.
            - 'store_zip3': First three digits of the store's ZIP code.
            - 'fips_state_code': FIPS state code.
            - 'fips_state_descr': Description of the FIPS state code.
            - 'fips_county_code': FIPS county code.
            - 'fips_county_descr': Description of the FIPS county code.
    """
    
    # stores_file = pd.read_csv(f'Nielsen_data/{year}/Annual_Files/stores_{year}.tsv', sep = '\t', header = 0)
    stores_file = pd.read_csv(filepath_or_buffer=stores_path, sep = '\t', header = 0)
    stores_file = stores_file[['store_code_uc', 'store_zip3', 'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr']]
    return stores_file


def products_file(products_path: str):
    """
    Reads a TSV file containing product data, filters for duplicated UPCs, and selects specific columns.
    Args:
        products_path (str): The file path to the TSV file containing product data.
    Returns:
        pandas.DataFrame: A DataFrame containing the filtered product data with the following columns:
            - 'upc': Universal Product Code
            - 'upc_descr': Description of the UPC
            - 'brand_code_uc': Brand code
            - 'brand_descr': Brand description
            - 'multi': Multipack indicator
            - 'product_module_code': Product module code
            - 'product_group_code': Product group code
    """

    # products_file = pd.read_csv(f'Nielsen_data/Master_Files/Latest/products.tsv', sep = '\t', encoding='latin1')
    products_file = pd.read_csv(filepath_or_buffer=products_path, sep = '\t', encoding='latin1')
    products_file = products_file[products_file['upc'].duplicated(keep=False)]
    products_file = products_file[['upc', 'upc_descr', 'brand_code_uc', 'brand_descr','multi', 'product_module_code', 'product_group_code']]
    return products_file


def extra_attributes_file(extra_attributes_path: str, moves_data: pd.DataFrame):
    """
    Filters and returns extra product attributes for a given set of UPCs.
    This function reads a TSV file containing extra product attributes, filters the attributes
    based on the UPCs present in the provided moves_data DataFrame, and returns a DataFrame
    containing the filtered attributes.
    Parameters:
    extra_attributes_path (str): The file path to the TSV file containing extra product attributes.
    moves_data (pd.DataFrame): A DataFrame containing product movement data, which includes a column 'upc' with unique product codes.
    Returns:
    pd.DataFrame: A DataFrame containing the filtered extra product attributes, including the following columns:
                  'upc', 'style_code', 'style_descr', 'type_code', 'type_descr', 'strength_code', 'strength_descr'.
    """

    # products_extra_attributes = pd.read_csv(f'Nielsen_data/{year}/Annual_Files/products_extra_{year}.tsv', sep  = '\t', header = 0)
    products_extra_attributes = pd.read_csv(filepath_or_buffer=extra_attributes_path, sep  = '\t', header = 0)
    unique_upcs = set(moves_data['upc'])

    products_extra_attributes = products_extra_attributes[products_extra_attributes['upc'].isin(unique_upcs)]
    products_extra_attributes = products_extra_attributes[['upc','style_code','style_descr','type_code','type_descr','strength_code','strength_descr']]
    return products_extra_attributes


def main():
    os.chdir(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Movement_Files/{DEPARTMENT_CODE}_{YEAR}/')

    movements_data = movements_file()
    stores_data = stores_file()
    products_data = products_file()
    extra_attributes_data = extra_attributes_file(movements_data)

    product_data = pd.merge(movements_data, stores_data, on='store_code_uc', how='left')
    product_data = pd.merge(product_data, products_data, on='upc', how='left')
    product_data = pd.merge(product_data, extra_attributes_data, on='upc', how='left')

    product_data['week_end_ID'] = pd.factorize(product_data['week_end'])[0]
    product_data['market_ids'] = product_data.apply(retail_market_ids_identifier, axis=1)
    product_data['market_ids_fips'] = product_data.apply(retail_market_ids_fips, axis=1)
    product_data['firm_ids'] = None

    product_data = product_data[['store_code_uc','market_ids','market_ids_fips','store_zip3','week_end','week_end_ID',#mercado tiempo y espacio
                        'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr',
                        'upc','firm_ids', 'brand_code_uc','brand_descr', ##companía y marca
                        'units', 'multi', 'price', 'prmult', #cantidades y precio 
                        'style_code','style_descr', 'type_code', 'type_descr','strength_code','strength_descr']]# características del producto

    product_data['brand_descr'] = product_data['brand_descr'].fillna('Not_identified')

    nivel_de_agregacion = 'retailer'
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/analisis/1.compiled_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)

    print('fin')


if __name__=='__main__':
    main()
