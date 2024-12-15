import os
import pandas as pd
import json
from thefuzz import process


with open('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/cleaned_data.json', 'r') as file:
    characteristics = pd.DataFrame(json.load(file))


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


def match_brands_to_characteristics(product_data, characteristics, threshold=85):
    """
    Matches brands from product_data to names in characteristics using fuzzy matching.

    Args:
        product_data (dict or DataFrame): Contains 'brand_descr' to extract unique brands.
        characteristics (dict or DataFrame): Contains 'name' to match brands against.
        threshold (int): Minimum similarity score to consider a match (default: 85).

    Returns:
        list: A list of dictionaries with matched brands and their characteristics.
    """
    i = 0
    brands = list(set(product_data['brand_descr']))
    output = []

    while len(brands) >= 1:
        elem = brands.pop()
        result = process.extract(elem, list(characteristics['name']), limit=1)[0]
        if result[1] >= threshold:
            add_output = {'from_nielsen': elem, 'from_characteristics': result[0]}
            output.append(add_output)
        i += 1

    return output


def main():
    os.chdir('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/analisis')

    product_data_file = list_of_files()[-1]    
    product_data = pd.read_csv(product_data_file)
    product_data['brand_descr']=product_data['brand_descr'].str.lower()
    match_bases_datos = match_brands_to_characteristics(product_data, characteristics, threshold=85)
    characteristics_matches=pd.DataFrame.from_dict(match_bases_datos)
    product_data = product_data.merge(characteristics_matches, how='left', left_on='brand_descr', right_on='from_nielsen')
    product_data = product_data.merge(characteristics, how='left', left_on='from_characteristics', right_on='name')
    product_data = product_data[['market_ids', 'market_ids_fips',
                             #variables relativas a la ubicacion
                             'store_code_uc', 'zip', 'fip', 'fips_state_code',  'fips_county_code', #'fips_county_descr',  'fips_state_descr',
                             #variables relativas al tiempo
                             'week_end', 'week_end_ID', 
                             #variables relativas a la compania y a la marca
                             'firm', 'firm_ids', 'brand_code_uc', 'brand_descr',
                             #varibles relativas a cantidades
                             'units', 'total_individual_units', 'total_units_retailer',
                             #variables relativas a partipación del mercado 
                             'shares', 'poblacion_census_2020',
                             #variables relativas a ingresos totales
                             'total_income', 'total_income_market_known_brands','total_income_market', 'fraction_identified_earnings',
                             #variablers relagtivas a los precios 
                             'prices',
                             #variables relativas a caracteristicas del producto obtenidas de Nielsen      
#                            'style_code', 'style_descr', 'type_code', 'type_descr', 'strength_code', 'strength_descr', 
                             #variables para hacer el merge con las características de los productos
                            'from_nielsen', 'from_characteristics', 'name', 
                             #relativas a características del producto obtenidas de otras fuentes, incluyendo la necesaria para hacer el merge con la base de las características  
                             'tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig']]
    product_data = product_data[product_data['name'].notna()]
    product_data.to_csv(f'4.caracteristicas_product_data_{nivel}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)


if __name__ == '__main__':
    main()