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
    return [el for el in elements if any(re.search(pattern, el) for pattern in patterns)]


def filter_module_code(row):
    return row['product_module_code'] == PRODUCT_MODULE


def filter_module_code(row):
    return row['product_module_code'] == PRODUCT_MODULE


def filter_market_ids(row):
    return row['market_ids'] in MARKET_IDS_FILTER


def filter_row_weeks(row, weeks: list):
    return row['week_end'] in weeks


# def filter_row_weeks(row):
#     return row['week_end'] in WEEKS


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
    return 'M'+str(row['store_zip3'])+'W'+str(row['week_end_ID'])
#     return str(row['store_zip3'])+'-'+str(row['week_end_ID'])


def retail_market_ids_identifier(row)-> str:
    """
    Crea una nueva columna, market_ids, que combina la informacion de mercado a nivel espacial y temporal. 
    """
    return 'R'+str(row['store_code_uc'])+'W'+str(row['week_end_ID'])
#     return str(row['store_zip3'])+'-'+str(row['week_end_ID'])


def market_ids_fips(row) -> str:
    return 'M'+str(row['fips_state_code'])+'.'+str(row['fips_county_code'])+'W'+str(row['week_end_ID'])
#     return str(row['fips_state_code'])+'.'+str(row['fips_county_code'])+'-'+str(row['week_end_ID'])


def retail_market_ids_fips(row) -> str:
    return 'R'+str(row['store_code_uc'])+'W'+str(row['week_end_ID'])
#     return str(row['fips_state_code'])+'.'+str(row['fips_county_code'])+'-'+str(row['week_end_ID'])


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


def movements_file(movements_path: str, filter_row_weeks: Callable):
    # movements_file = pd.read_csv(f'raw_data/2013/Movement_Files/4510_2013/7460_2013.tsv', sep  = '\t', header = 0, index_col = None)
    movements_file = pd.read_csv(filepath_or_buffer=movements_path, sep  = '\t', header = 0, index_col = None)
    movements_file = movements_file[['store_code_uc', 'upc', 'week_end', 'units', 'prmult', 'price']]
    weeks = list(sorted(list(set(movements_file['week_end'])))[0:2])
    # Apply the filtering function
    movements_file = movements_file[movements_file.apply(lambda row: filter_row_weeks(row, weeks), axis=1)]

    # weeks_filter_partial = partial(filter_row_weeks, weeks)
    # movements_file = movements_file[movements_file.apply(weeks_filter_partial, axis=1)]

    return movements_file


def stores_file(stores_path: str, year: int):
    # stores_file = pd.read_csv(f'raw_data/{year}/Annual_Files/stores_{year}.tsv', sep = '\t', header = 0)
    stores_file = pd.read_csv(filepath_or_buffer=stores_path, sep = '\t', header = 0)
    stores_file = stores_file[['store_code_uc', 'store_zip3', 'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr']]
    return stores_file


def products_file(products_path: str):
    # products_file = pd.read_csv(f'raw_data/Master_Files/Latest/products.tsv', sep = '\t', encoding='latin1')
    products_file = pd.read_csv(filepath_or_buffer=products_path, sep = '\t', encoding='latin1')
    products_file = products_file[products_file['upc'].duplicated(keep=False)]
    products_file = products_file[['upc', 'upc_descr', 'brand_code_uc', 'brand_descr','multi', 'product_module_code', 'product_group_code']]
    return products_file


def extra_attributes_file(extra_attributes_path: str, moves_data: pd.DataFrame, year: int):
    # products_extra_attributes = pd.read_csv(f'raw_data/{year}/Annual_Files/products_extra_{year}.tsv', sep  = '\t', header = 0)
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
