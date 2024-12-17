import pandas as pd
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


def list_unique_elements_per_group(df, group_col, value_col):
    """
    Lists the unique elements in the 'value_col' for each group defined by 'group_col'.
    
    Parameters:
    - df: DataFrame
    - group_col: str, the column name to group by
    - value_col: str, the column name containing values to list
    
    Returns:
    - A DataFrame with group_col and the list of unique elements in value_col for each group
    """
    return df.groupby(by=group_col)[value_col].apply(lambda x: list(x.unique())).reset_index()


def create_2x2_table(df, condition1, condition2):
    """
    Create a 2x2 table that groups observations based on two conditions.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    condition1 (str): The first condition (a boolean mask or query string).
    condition2 (str): The second condition (a boolean mask or query string).

    Returns:
    pd.DataFrame: A 2x2 table showing the count of observations in each group.
    """
    # Create boolean masks for the conditions
    mask1 = df.query(condition1).index
    mask2 = df.query(condition2).index

    # Define groups
    condition1_true = df.index.isin(mask1)
    condition2_true = df.index.isin(mask2)

    # Create a 2x2 contingency table
    table = pd.crosstab(condition1_true, condition2_true)
    table.index = [f"{condition1} (True)", f"{condition1} (False)"]
    table.columns = [f"{condition2} (True)", f"{condition2} (False)"]

    return table


def create_2x2_table(df, condition1, condition2):
    """
    Create a 2x2 table that groups observations based on two conditions.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    condition1 (str): The first condition (a boolean mask or query string).
    condition2 (str): The second condition (a boolean mask or query string).
    
    Returns:
    pd.DataFrame: A 2x2 DataFrame with the counts for each combination of conditions.
    """
    # Apply the conditions to the DataFrame
    df['condition1'] = df.eval(condition1)
    df['condition2'] = df.eval(condition2)
    
    # Create the 2x2 table
    table = pd.crosstab(df['condition1'], df['condition2'])
    
    # Drop the temporary columns
    df = df.drop(['condition1', 'condition2'], axis=1)
    
    return table


def unique_elements_ordered(list1, list2):
    # Elements unique to list1
    unique_in_list1 = [x for x in list1 if x not in list2]
    # Elements unique to list2
    unique_in_list2 = [x for x in list2 if x not in list1]
    return unique_in_list1, unique_in_list2 



def main():
    product_data = pd.read_csv('4.caracteristicas_product_data_retailer_Reynolds_Lorillard_2024-12-04 11:48:53.975453.csv')
    product_data.rename(columns={'market_ids_fips':'market_ids_string'}, inplace=True)
    product_data['market_ids']=product_data['market_ids_string'].factorize()[0]
    markets_characterization =product_data[['zip',
                          'market_ids_string',
                          'market_ids',
                          'total_income_market',
                          'total_income_market_known_brands',
                          'fraction_identified_earnings']].sort_values(by=['fraction_identified_earnings'],
                                                                                         axis=0,
                                                                                         ascending=False)
    product_data = product_data[(product_data['total_income_market_known_brands'] > 700) & (product_data['fraction_identified_earnings'] >0.4 )].reset_index()
    del product_data['index']
    product_data['market_ids']=product_data['market_ids_string'].factorize()[0]
    product_data['product_ids'] = pd.factorize(product_data['brand_descr'])[0]


if __name__ == '__main__':
    main()