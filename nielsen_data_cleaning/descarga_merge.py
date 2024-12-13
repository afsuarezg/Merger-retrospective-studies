import os
import numpy as np
import pandas as pd
import pyblp
import re
import matplotlib.pyplot as plt
import json
import sys 
import collections
import datetime


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


def movements_file():
    movements_file = pd.read_csv(f'{PRODUCT_MODULE}_{YEAR}.tsv',sep  = '\t', header = 0, index_col = None, nrows = NROWS)
    movements_file = movements_file[['store_code_uc', 'upc', 'week_end', 'units', 'prmult', 'price']]
    movements_file = movements_file[movements_file.apply(filter_row_weeks, axis=1)]
    return movements_file


def stores_file():
    stores_file = pd.read_csv(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Annual_Files/stores_{YEAR}.tsv',sep = '\t', header = 0)
    stores_file = stores_file[['store_code_uc', 'store_zip3', 'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr']]
    return stores_file


def products_file():
    products_file = pd.read_csv(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/Master_Files/Latest/products.tsv', sep = '\t', encoding='latin1')
    products_file = products_file[products_file['upc'].duplicated(keep=False)]
    products_file = products_file[['upc', 'upc_descr', 'brand_code_uc', 'brand_descr','multi', 'product_module_code', 'product_group_code']]
    return products_file


def extra_attributes_file():
    unique_upcs = set(movements_file['upc'])
    products_extra_attributes = pd.read_csv('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/nielsen_extracts/RMS/2015/Annual_Files/products_extra_2015.tsv', sep  = '\t', header = 0)
    products_extra_attributes = products_extra_attributes[products_extra_attributes['upc'].isin(unique_upcs)]
    products_extra_attributes = products_extra_attributes[['upc','style_code','style_descr','type_code','type_descr','strength_code','strength_descr']]
    return products_extra_attributes


def main():
    DIRECTORY_NAME = 'Reynolds_Lorillard'
    DEPARTMENT_CODE = 4510 #aka product_group_code
    PRODUCT_MODULE = 7460
    NROWS = 10000000
    YEAR = 2014
    WEEKS = [20140125, 20140201]
    os.chdir(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Movement_Files/{DEPARTMENT_CODE}_{YEAR}/')

    movements_data = movements_file()
    stores_data = stores_file()
    products_data = products_file()
    extra_attributes_data = extra_attributes_file()

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
