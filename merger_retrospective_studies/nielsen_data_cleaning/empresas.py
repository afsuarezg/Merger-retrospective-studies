import os
import pandas as pd
import json

# # Get the directory of the current file (empresas.py)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Construct the absolute path to cigarettes.json
# absolute_path = os.path.join(current_dir, "../brands_by_company_pre_merger/cigarettes.json")

# # Normalize the path (handles "..")
# absolute_path = os.path.normpath(absolute_path)

with open("/oak/stanford/groups/polinsky/Mergers/Cigarettes/Firmas_marcas/cigarette_ownership_pre_merger_lower_case.json", 'r') as file:
    # brands_by_company_pre_merger = 
    # brands_by_company_pre_merger = {key: [s.lower() for s in value] for key, value in brands_by_company_pre_merger.items()}
    brands_by_company_pre_merger = {key: [s.lower() for s in value] for key, value in json.load(file).items()}


with open("/oak/stanford/groups/polinsky/Mergers/Cigarettes/Firmas_marcas/cigarette_ownership_post_merger_lower_case.json", 'r') as file:
    # brands_by_company_post_merger = json.load(file)
    brands_by_company_post_merger = {key: [s.lower() for s in value] for key, value in json.load(file).items()}


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
    return df.groupby(group_col)[value_col].apply(lambda x: list(x.unique())).reset_index()


def list_of_files():
    """
    Retrieves a sorted list of files in a specified directory.
    This function lists all files in the given directory and sorts them 
    based on their last modification time in ascending order.
    Returns:
        list: A list of filenames sorted by their last modification time.
    """

    dir_name = '/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/analisis'
    # Get list of all files only in the given directory
    list_of_files = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                            os.listdir(dir_name) )
    # Sort list of files based on last modification time in ascending order
    list_of_files = sorted( list_of_files,
                            key = lambda x: os.path.getmtime(os.path.join(dir_name, x))
                            )
    
    return list_of_files


# def find_company_pre_merger_discontinued(row):
#     for company in brands_by_company_pre_merger.keys():
#         if row['brand_descr'] in brands_by_company_pre_merger[company]['brands']:
#             return company
#     else:
#         return 'unidentified'


def find_company_pre_merger(row, company_brands_dict:dict):

    """
    Identifies the company associated with a brand before a merger.
    This function takes a row of data and a dictionary mapping companies to their respective brands.
    It checks if the brand is associated with any company in the dictionary.
    If a match is found, it returns the company name; otherwise, it returns 'unidentified'.
    Parameters:
    row (dict): A dictionary representing a row of data, which must contain a 'brand_descr' key.
    company_brands_dict (dict): A dictionary where keys are company names and values are lists of brand names associated with each company.
    Returns:
    str: The name of the company associated with the brand, or 'unidentified' if no match is found.
    """

    for company in company_brands_dict.keys():
        if row['brand_descr'] in company_brands_dict[company]:
            return company
    else:
        return 'unidentified'


def find_company_post_merger(row, company_brands_dict:dict):
    """
    Identifies the company associated with a brand after a merger.
    Args:
        row (dict): A dictionary representing a row of data, which must contain the key 'brand_descr'.
        company_brands_dict (dict): A dictionary where keys are company names and values are lists of brand descriptions associated with each company.
    Returns:
        str: The name of the company associated with the brand in the given row. 
             Returns 'unidentified' if the brand is not found in any company's list.
    """

    for company in company_brands_dict.keys():
        if row['brand_descr'] in company_brands_dict[company]:
            return company
    else:
        return 'unidentified'


def main():
    os.chdir('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/analisis')  
    product_data_file = list_of_files()[-1]    
    product_data = pd.read_csv(product_data_file)
    product_data['firm']=product_data.apply(find_company, axis=1)
    product_data['firm_ids']=(pd.factorize(product_data['firm']))[0]


if __name__ == '__main__':
    print(brands_by_company_pre_merger)   