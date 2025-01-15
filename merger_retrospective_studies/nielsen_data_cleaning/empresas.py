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
    dir_name = '/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/analisis'
    # Get list of all files only in the given directory
    list_of_files = filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                            os.listdir(dir_name) )
    # Sort list of files based on last modification time in ascending order
    list_of_files = sorted( list_of_files,
                            key = lambda x: os.path.getmtime(os.path.join(dir_name, x))
                            )
    
    return list_of_files


def find_company_pre_merger_discontinued(row):
    for company in brands_by_company_pre_merger.keys():
        if row['brand_descr'] in brands_by_company_pre_merger[company]['brands']:
            return company
    else:
        return 'unidentified'


def find_company_pre_merger(row):
    print( brands_by_company_pre_merger )
    for company in brands_by_company_pre_merger.keys():
        if row['brand_descr'] in brands_by_company_pre_merger[company]:
            return company
    else:
        return 'unidentified'


def find_company_post_merger(row):
    print(brands_by_company_post_merger)
    for company in brands_by_company_post_merger.keys():
        if row['brand_descr'] in brands_by_company_post_merger[company]:
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