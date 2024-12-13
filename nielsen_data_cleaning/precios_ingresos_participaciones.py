import os
import pandas as pd
import re
import json
import math
import time
import bisect


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
    Crea un dataframe que agrega el tamaño de los mercados en total dollar sales por zip.
    """ 
    return df.groupby(['market_ids'],as_index = False).agg({'total dollar sales':'sum'})


def market_ids_identifier(row)-> str:
    """
    Crea una nueva columna, market_ids, que combina la informacion de mercado a nivel espacial y temporal. 
    """
    return 'ZIP'+str(row['zip'])+'WEEK'+str(row['week_end_ID'])


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
    brands_by_zip = df.groupby('zip')['brand_descr'].apply(list).reset_index()
    brands_by_zip['brand_descr'] = brands_by_zip['brand_descr'].apply(set).apply(list)
    return brands_by_zip.set_index('zip')['brand_descr'].to_dict()


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

def unique(list1):
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def total_income(row):
    return (row['price']/row['prmult'])*row['units']


def total_units(row):
    return row['multi']*row['units']


def unitary_price(row):
    return row['price']/row['prmult']


def price(row):
    return row['total_income']/row['units']


def fraccion_ventas_identificadas(row):
    return row['total_income_market_known_brands']/row['total_income_market']


def count_rows_meeting_condition(df, column_name, condition):
    """
    Count the number of rows in a DataFrame where the specified column meets the given condition.

    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    column_name (str): The name of the column to apply the condition to.
    condition (callable): A function that takes a column value and returns True if the condition is met.

    Returns:
    int: The number of rows where the condition is met.
    """
    return df[column_name].apply(condition).sum()


def find_first_above_threshold(arr, threshold):
    for i, num in enumerate(arr):
        if num > threshold:
            return i
    return -1  # Return -1 if no number is above the threshold


def find_first_above_threshold(arr, threshold):
    position = bisect.bisect_right(arr, threshold)
    return position if position < len(arr) else -1


def sum_from_first_above_threshold(arr, threshold, values_to_sum):
    # Find the position of the first number in `arr` that is above the threshold
    position = bisect.bisect_right(arr, threshold)
    
    # If no element in `arr` is above the threshold, return 0 (or any other appropriate value)
    if position == len(arr):
        return 0
    
    # Sum the elements of `values_to_sum` starting from the `position`
    return sum(values_to_sum[position:])


def group_unique_strings(df, groupby_column, target_column):
    # Group the dataframe by the specified column
    grouped = df.groupby(by=groupby_column)[target_column].apply(lambda x: list(x.unique())).reset_index()
    
    # Rename the target column to indicate that it contains lists of unique strings
    grouped.rename(columns={target_column: f'{target_column}_unique_list'}, inplace=True)
    
    return grouped


def sort_by_list_length(df, list_column, ascending=True):
    # Create a new column that contains the length of the lists
    df['list_length'] = df[list_column].apply(len)
    
    # Sort the DataFrame by the length of the lists
    df_sorted = df.sort_values(by='list_length', ascending=ascending).drop(columns='list_length')
    
    return df_sorted


def group_common_elements(df, groupby_column, target_column):
    def common_elements(lists):
        # Find the common elements in all lists
        return list(set.intersection(*map(set, lists)))

    # Group the dataframe by the specified column and find common elements
    grouped = df.groupby(by=groupby_column)[target_column].apply(common_elements).reset_index()
    
    # Rename the target column to indicate that it contains common elements
    grouped.rename(columns={target_column: f'{target_column}_common_elements'}, inplace=True)
    
    return grouped


def main():
    dir_name = '/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/analisis'
    # Get list of all files only in the given directory
    list_of_files = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                            os.listdir(dir_name) )
    # Sort list of files based on last modification time in ascending order
    list_of_files = sorted( list_of_files,
                            key = lambda x: os.path.getmtime(os.path.join(dir_name, x))
                            )
    # Iterate over sorted list of files and print file path 
    # along with last modification time of file 
    for file_name in list_of_files:
        file_path = os.path.join(dir_name, file_name)
        timestamp_str = time.strftime(  '%m/%d/%Y :: %H:%M:%S',
                                    time.gmtime(os.path.getmtime(file_path))) 
        print(timestamp_str, ' -->', file_name) 

    # product_data = pd.read_csv('1.compiled_retailer_Reynolds_Lorillard_2024-11-11 22:13:10.572819.csv')
    # product_data['total_income'] = product_data.apply(total_income, axis=1)
    # product_data['total_individual_units'] = product_data.apply(total_units, axis=1)
    # product_data['unitary_price'] = product_data.apply(unitary_price, axis=1)

    # product_data = product_data[['store_code_uc', 'market_ids', 'market_ids_fips',  'store_zip3', 'week_end', 'week_end_ID',
    #                          'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr',
    #    'upc', 'firm_ids', 'brand_code_uc', 'brand_descr', 
    #    'units', 'multi', 'price', 'prmult','unitary_price', 'total_income',
    #    'total_individual_units',
    #    'style_code', 'style_descr', 'type_code','type_descr', 'strength_code', 'strength_descr']]
    
    # product_data.rename(columns={'store_zip3':'zip'}, inplace=True)

    # product_data = product_data.groupby(['market_ids', 'brand_descr', 'store_code_uc'], as_index=False).agg({
    #             'zip':'first' ,
    #             'week_end':'first' ,
    #             'week_end_ID':'first',
    #         #     'upc':'first', # se pierde al agregar a través de marcas
    #             'market_ids_fips':'first',
    #             'fips_state_code':'first', 
    #             'fips_state_descr':'first', 
    #             'fips_county_code':'first', 
    #             'fips_county_descr':'first',
    #             'firm_ids':'first', #No está definido aún. 
    #             'brand_code_uc': 'first',
    #             'brand_descr':'first',
    #             'units': 'sum',
    #             'unitary_price':'mean',#,No vale la pena agregarlo porque no se puede calcular como el promedio simple de todas las observaciones
    #             'price': 'mean',
    #             'total_individual_units': 'sum',
    #             'total_income': 'sum',
                
    #         #     'prices': 'mean'  ,
    #         #     'total dollar sales': 'sum' ,
    #             'style_code': 'mean' ,
    #             'style_descr': 'first',
    #             'type_code': 'mean' ,
    #             'type_descr': 'first' ,
    #             'strength_code': 'mean',
    #             'strength_descr': 'first', 
    #         #     'total dollar sales': 'sum',   # Summing up the 'Value1' column
    #         #     'Value2': 'mean'   # Calculating mean of the 'Value2' column
    #         })

    # product_data.rename(columns={'unitary_price':'unitary_price_x_reemplazar', 'price':'price_x_reemplazar'}, inplace=True)
    # product_data['prices'] = product_data.apply(price, axis=1)

    # total_sales_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids','store_code_uc'], as_index=False).agg({'total_income': 'sum'}))
    # total_sales_per_marketid = total_sales_per_marketid.rename(columns={'total_income':'total_income_market'})

    # total_sales_identified_per_marketid = pd.DataFrame(product_data[product_data['brand_descr']!='Not_identified'].groupby(by=['market_ids','store_code_uc'],
    #                         as_index=False).agg({'total_income': 'sum'}))
    # total_sales_identified_per_marketid = total_sales_identified_per_marketid.rename(columns={'total_income':'total_income_market_known_brands'})
    # product_data = product_data.merge(total_sales_per_marketid, 
    #                                 how ='left',
    #                                 on=['market_ids','store_code_uc'])
    # product_data = product_data.merge(total_sales_identified_per_marketid, 
    #                               how='left',
    #                               on=['market_ids','store_code_uc'])
    # product_data['total_income_market_known_brands'].fillna(0.0, inplace=True)
    # product_data['fraction_identified_earnings'] = product_data.apply(fraccion_ventas_identificadas, axis=1)
    # total_sold_units_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids',
    #                                                              'store_code_uc'], as_index=False).agg({'units': 'sum'}))
    # total_sold_units_per_marketid.rename(columns={'units':'total_units_retailer'}, inplace=True)
    # product_data = product_data.merge(total_sold_units_per_marketid, 
    #                               how ='left',
    #                               on=['market_ids','store_code_uc'])
    # product_data = product_data[product_data['brand_code_uc'].notna()]


main()

if __name__ == '__main__':
    print(main()) 