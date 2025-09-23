import os
import pandas as pd
import re
import json
import math
import time
import bisect


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
PRODUCT_MODULE = 7460
NROWS = 10000000
YEAR = 2014
WEEKS = [20140125, 20140201]


def match_patterns(elements, patterns):
    """
    Filters a list of elements by matching them against a list of regular expression patterns.
    Args:
        elements (list of str): The list of elements to be filtered.
        patterns (list of str): The list of regular expression patterns to match against the elements.
    Returns:
        list of str: A list of elements that match any of the given patterns.
    """

    return [el for el in elements if any(re.search(pattern, el) for pattern in patterns)]


def filter_module_code(row):
    """
    Filters a row based on the 'product_module_code' column.
    Args:
        row (pd.Series): A pandas Series representing a row of data.
    Returns:
        bool: True if the 'product_module_code' of the row matches the global PRODUCT_MODULE, False otherwise.
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
    This function checks if the 'market_ids' value in the given row is present
    in the predefined list of market IDs (MARKET_IDS_FILTER).
    Args:
        row (pandas.Series): A row of data containing a 'market_ids' field.
    Returns:
        bool: True if the 'market_ids' value is in MARKET_IDS_FILTER, False otherwise.
    """

    return row['market_ids'] in MARKET_IDS_FILTER


def filter_row_weeks(row):
    """
    Filters rows based on the 'week_end' column.
    Args:
        row (pd.Series): A row of data containing a 'week_end' column.
    Returns:
        bool: True if the 'week_end' value is in the predefined list WEEKS, False otherwise.
    """

    return row['week_end'] in WEEKS


def unit_price(row):
    """
    Calculate the unit price from a given row of data.
    Args:
        row (dict): A dictionary containing 'price' and 'prmult' keys.
    Returns:
        float: The unit price calculated as price divided by prmult.
    """
    
    return row['price']/row['prmult']


def total_dollar_sales(row):
    """
    Calculate the total dollar sales for a given row.
    This function multiplies the 'prices' and 'units' values from the input row to compute the total dollar sales.
    Parameters:
    row (pd.Series): A pandas Series object containing 'prices' and 'units' columns.
    Returns:
    float: The total dollar sales calculated as the product of 'prices' and 'units'.
    """

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
    """
    Calculate the market share of a product.
    This function takes a row from a DataFrame and calculates the market share
    by dividing the total dollar sales by the market size.
    Parameters:
    row (pd.Series): A pandas Series object representing a row of data. It must
                     contain 'total dollar sales' and 'market size' columns.
    Returns:
    float: The market share calculated as the ratio of total dollar sales to market size.
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
    """
    Returns a list of unique elements from the given list.
    This function traverses the input list and appends elements to a new list
    only if they are not already present in the new list, ensuring that the
    resulting list contains only unique elements.
    Parameters:
    list1 (list): The list from which to extract unique elements.
    Returns:
    list: A list containing only the unique elements from the input list.
    """

    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def total_income(row):
    """
    Calculate the total income for a given row of data.
    This function computes the total income by dividing the price by the 
    price multiplier (prmult) and then multiplying by the number of units.
    Parameters:
    row (dict): A dictionary containing the keys 'price', 'prmult', and 'units'.
    Returns:
    float: The total income calculated from the given row.
    """
    
    return (row['price']/row['prmult'])*row['units']


def total_units(row):
    """
    Calculate the total units by multiplying the 'multi' and 'units' columns of a given row.
    Parameters:
    row (pandas.Series): A row of data containing 'multi' and 'units' columns.
    Returns:
    int or float: The product of 'multi' and 'units' from the given row.
    """

    return row['multi']*row['units']


def unitary_price(row):
    """
    Calculate the unitary price from a given row of data.
    Args:
        row (pd.Series): A pandas Series object containing 'price' and 'prmult' columns.
    Returns:
        float: The unitary price calculated as price divided by prmult.
    """

    return row['price']/row['prmult']


def price(row):
    """
    Calculate the price per unit.
    This function takes a row of data and calculates the price per unit by 
    dividing the total income by the number of units.
    Parameters:
    row (dict): A dictionary containing 'total_income' and 'units' keys.
    Returns:
    float: The calculated price per unit.
    """

    return row['total_income']/row['units']


def fraccion_ventas_identificadas(row):
    """
    Calculate the fraction of identified sales in the market.
    This function takes a row of data and computes the fraction of total income
    in the market that is attributed to known brands.
    Parameters:
    row (pd.Series): A pandas Series object containing the following keys:
        - 'total_income_market_known_brands': Total income from known brands in the market.
        - 'total_income_market': Total income in the market.
    Returns:
    float: The fraction of total income in the market that is from known brands.
    """
    
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
    """
    Find the index of the first element in the array that is above the given threshold.
    Parameters:
    arr (list of int/float): The array to search through.
    threshold (int/float): The threshold value to compare against.
    Returns:
    int: The index of the first element that is above the threshold. 
         Returns -1 if no element is above the threshold.
    """

    for i, num in enumerate(arr):
        if num > threshold:
            return i
    return -1  # Return -1 if no number is above the threshold


def find_first_above_threshold(arr, threshold):
    """
    Find the first position in a sorted array where the value is above a given threshold.
    Args:
        arr (list): A sorted list of values.
        threshold (int or float): The threshold value to compare against.
    Returns:
        int: The index of the first element in the array that is greater than the threshold.
             Returns -1 if no such element is found.
    """

    position = bisect.bisect_right(arr, threshold)
    return position if position < len(arr) else -1


def sum_from_first_above_threshold(arr, threshold, values_to_sum):
    """
    Sums the elements of `values_to_sum` starting from the position of the first element in `arr` that is above the given `threshold`.
    Parameters:
    arr (list of int/float): The list in which to find the first element above the threshold.
    threshold (int/float): The threshold value to compare against elements in `arr`.
    values_to_sum (list of int/float): The list of values to sum from the identified position.
    Returns:
    int/float: The sum of the elements in `values_to_sum` starting from the position of the first element in `arr` that is above the threshold. Returns 0 if no element in `arr` is above the threshold.
    """

    # Find the position of the first number in `arr` that is above the threshold
    position = bisect.bisect_right(arr, threshold)
    
    # If no element in `arr` is above the threshold, return 0 (or any other appropriate value)
    if position == len(arr):
        return 0
    
    # Sum the elements of `values_to_sum` starting from the `position`
    return sum(values_to_sum[position:])


def group_unique_strings(df, groupby_column, target_column):
    """
    Groups a DataFrame by a specified column and creates a list of unique strings 
    from another specified column for each group.
    Parameters:
    df (pandas.DataFrame): The DataFrame to be grouped.
    groupby_column (str): The column name to group by.
    target_column (str): The column name from which to extract unique strings.
    Returns:
    pandas.DataFrame: A DataFrame with the groupby_column and a new column 
                      containing lists of unique strings from the target_column.
    """

    # Group the dataframe by the specified column
    grouped = df.groupby(by=groupby_column)[target_column].apply(lambda x: list(x.unique())).reset_index()
    
    # Rename the target column to indicate that it contains lists of unique strings
    grouped.rename(columns={target_column: f'{target_column}_unique_list'}, inplace=True)
    
    return grouped


def sort_by_list_length(df, list_column, ascending=True):
    """
    Sort a DataFrame by the length of lists in a specified column.
    Parameters:
    df (pandas.DataFrame): The DataFrame to be sorted.
    list_column (str): The name of the column containing lists.
    ascending (bool, optional): If True, sort in ascending order, otherwise in descending order. Default is True.
    Returns:
    pandas.DataFrame: The sorted DataFrame with the original order of rows preserved.
    """

    # Create a new column that contains the length of the lists
    df['list_length'] = df[list_column].apply(len)
    
    # Sort the DataFrame by the length of the lists
    df_sorted = df.sort_values(by='list_length', ascending=ascending).drop(columns='list_length')
    
    return df_sorted


def group_common_elements(df, groupby_column, target_column):
    """
    Groups a DataFrame by a specified column and finds the common elements in another specified column.
    Parameters:
    df (pandas.DataFrame): The DataFrame to be grouped.
    groupby_column (str): The column name to group by.
    target_column (str): The column name in which to find common elements.
    Returns:
    pandas.DataFrame: A DataFrame with the grouped column and a new column containing the common elements.
    """    
    def common_elements(lists):
        # Find the common elements in all lists
        return list(set.intersection(*map(set, lists)))

    # Group the dataframe by the specified column and find common elements
    grouped = df.groupby(by=groupby_column)[target_column].apply(common_elements).reset_index()
    
    # Rename the target column to indicate that it contains common elements
    grouped.rename(columns={target_column: f'{target_column}_common_elements'}, inplace=True)
    
    return grouped


def prepend_zeros(row) :
    """
    Prepend zeros to the 'fips_county_code' to ensure it is 3 digits long and concatenate it with 'fips_state_code'.
    Args:
        row (dict): A dictionary containing 'fips_state_code' and 'fips_county_code'.
    Returns:
        str: A string that concatenates 'fips_state_code' and 'fips_county_code' with leading zeros if necessary.
    """    
    return str(row['fips_state_code'])+str(row['fips_county_code']).zfill(3)


def obtain_zip(row):
    """
    Extracts a portion of the 'Geographic Area Name' field from a given row.
    Args:
        row (dict): A dictionary representing a row of data, which contains a key 'Geographic Area Name'.
    Returns:
        str: A substring of the 'Geographic Area Name' field, specifically the characters from the third-to-last to the second-to-last position.
    """

    return row['Geographic Area Name'][-5:-2]


def zip(row):
    """
    Extracts a specific part of the 'zip' field from a given row.
    Args:
        row (dict): A dictionary containing a 'zip' key with a string value.
    Returns:
        str: A substring of the 'zip' value, specifically the third and fourth characters from the end.
    """

    return row['zip'][-5:-3]


def percentage_match(list1, list2):
    """
    Calculate the percentage of elements in list1 that are also present in list2.
    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.
    Returns:
        float: The percentage of elements in list1 that are also in list2.
    """

    # Find the intersection (common elements) of the two lists
    matches = set(list1) & set(list2)
    
    return (len(matches) / len(set(list1))) * 100


def shares_with_outside_good(row):
    """
    Calculate the market share of a product including an outside good.
    This function computes the market share of a product by dividing the number of units sold by 
    the product of the fraction of identified earnings, the population from the 2020 census, 
    a constant factor (0.78), and the number of quarters in a year (4).
    Parameters:
    row (pd.Series): A pandas Series object containing the following columns:
        - 'units': The number of units sold.
        - 'fraction_identified_earnings': The fraction of identified earnings.
        - 'CENSUS_2020_POP': The population from the 2020 census.
    Returns:
    float: The market share of the product including an outside good.
    """

    return row['units']/(row['fraction_identified_earnings']*row['poblacion_census_2020']*0.78*4)


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

    fips_pop= pd.read_excel('/oak/stanford/groups/polinsky/Tamaño_mercado/PopulationEstimates.xlsx', skiprows=4)
    fips_pop=fips_pop[['FIPStxt','State','CENSUS_2020_POP']]

    fips_pop['FIPS'] = fips_pop['FIPStxt'].astype('int  ')
    fips_pop['FIPStxt']=fips_pop['FIPStxt'].astype(str)
    product_data['fip'] = product_data.apply(prepend_zeros, axis=1).astype('int')
    fips_pop = fips_pop.rename(columns={'FIPS': 'fip'})
    product_data=product_data.merge(fips_pop[['CENSUS_2020_POP','fip']], how='left', on='fip')

    product_data = product_data[['market_ids', 'store_code_uc', 'zip','fip', 'week_end', 'week_end_ID',
       'market_ids_fips',  'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr', 
       'firm_ids', 'brand_code_uc','brand_descr', 
       'units',  'prices', 'unitary_price_x_reemplazar','price_x_reemplazar',
       'total_individual_units',  'total_units_retailer',
       'style_code', 'style_descr', 'type_code', 'type_descr', 'strength_code', 'strength_descr',
       'total_income','total_income_market', 'total_income_market_known_brands',
       'fraction_identified_earnings',  
       'CENSUS_2020_POP']]
    product_data['shares']=product_data.apply(shares_with_outside_good, axis=1)
    product_data.rename(columns={'CENSUS_2020_POP':'poblacion_census_2020'}, inplace=True)


if __name__ == '__main__':
    print(main()) 