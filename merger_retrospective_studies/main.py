import os
import pandas as pd
import datetime
import pyblp
import numpy as np
import sys
import json
from itertools import chain
from scipy.linalg import svd

# from dotenv import load_dotenv

from .nielsen_data_cleaning.descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, retail_market_ids_fips, retail_market_ids_identifier, filter_row_weeks
from .nielsen_data_cleaning.caracteristicas_productos import match_brands_to_characteristics, list_of_files, characteristics
from .nielsen_data_cleaning.empresas import find_company_pre_merger, find_company_post_merger, brands_by_company_pre_merger, brands_by_company_post_merger
from .nielsen_data_cleaning.consumidores_sociodemograficas import read_file_with_guessed_encoding, process_file, get_random_samples_by_code, KNNImputer, add_random_nodes
from .nielsen_data_cleaning.precios_ingresos_participaciones import total_income, total_units, unitary_price, price, fraccion_ventas_identificadas, prepend_zeros, shares_with_outside_good
from .nielsen_data_cleaning.product_data_creation import creating_product_data_for_comparison, creating_product_data_rcl
from .estimaciones.plain_logit import plain_logit
from .estimaciones.rcl_without_demographics import rcl_without_demographics, rename_instruments
from .estimaciones.rcl_with_demographics import rcl_with_demographics
from .estimaciones.estimaciones_utils import save_dict_json
from .estimaciones.post_estimation_merger_simulation import predict_prices, original_prices
from .estimaciones.optimal_instruments import results_optimal_instruments
from .nielsen_data_cleaning.utils import create_output_directories, create_agent_data_from_cps, create_agent_data_sample


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
# PRODUCT_MODULE = 7460
# NROWS = 10000000
YEAR = 2014
# WEEKS = [20140125, 20140201]

#TODO: organizar las funciones de Nielsen_data_cleaning dependiendo de la base que está procesando. Las que se usen en diferentes bases deberían ir en un archivo más general. 

def creating_agent_data(product_data: pd.DataFrame, 
                        record_layout_path: str, 
                        agent_data_path: str):
    """
    Creates agent data by processing product data and agent data files.
    Parameters:
    product_data (pd.DataFrame): DataFrame containing product data with columns 'market_ids', 'market_ids_string', and 'GESTFIPS'.
    record_layout_path (str): Path to the record layout file.
    agent_data_path (str): Path to the agent data file.
    Returns:
    pd.DataFrame: DataFrame containing the merged and processed agent data with columns:
                    'FIPS', 'GESTFIPS', 'weights', 'nodes0', 'nodes1', 'nodes2', 'nodes3', 'nodes4',
                    'hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 'ptdtrace_imputed', 'peeduca_imputed'.
    """


    output = process_file(record_layout_path)
    agent_data_pop = pd.read_fwf(agent_data_path, widths= [int(elem) for elem in output.values()] )

    column_names = output.keys()
    agent_data_pop.columns = column_names
    agent_data_pop=agent_data_pop[agent_data_pop['GTCO']!=0]
    agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']
    agent_data_pop.reset_index(inplace=True, drop=True)

    # product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})
    
    demographic_sample = get_random_samples_by_code(agent_data_pop, product_data['GESTFIPS'], 200)[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]
    demographic_sample.replace(-1, np.nan, inplace=True)

    knn_imputer = KNNImputer(n_neighbors=2)
    demographic_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(demographic_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]),
                                columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 
                                        'ptdtrace_imputed', 'peeduca_imputed'])

    grouped = demographic_sample.groupby('GESTFIPS').size()

    demographic_sample['weights'] = demographic_sample['GESTFIPS'].map(1 / grouped)
    demographic_sample = pd.concat([demographic_sample[['FIPS', 'GESTFIPS','weights']],demographic_sample_knn_imputed], axis=1)
    demographic_sample = add_random_nodes(demographic_sample)

    demographic_sample = demographic_sample[['FIPS', 'GESTFIPS', 'weights',
                                            'nodes0', 'nodes1', 'nodes2', 'nodes3','nodes4',
                                            'hefaminc_imputed', 'prtage_imputed','hrnumhou_imputed', 
                                            'ptdtrace_imputed', 'peeduca_imputed']]
    
    agent_data = pd.merge(product_data[['market_ids', 'market_ids_string', 'GESTFIPS']].drop_duplicates(),
                                      demographic_sample, 
                                      how='inner', 
                                      left_on='GESTFIPS',
                                      right_on='GESTFIPS')
    return agent_data


def filtering_data_by_identified_sales(product_data: pd.DataFrame,
                                       blp_instruments: pd.DataFrame, 
                                       local_instruments: pd.DataFrame,
                                       quadratic_instruments: pd.DataFrame, 
                                       threshold_identified_earnings: float=0.4):
    """
    Filters the product data and associated instruments based on a threshold for identified earnings.
    Parameters:
    product_data (pd.DataFrame): DataFrame containing product data with a column 'fraction_identified_earnings'.
    blp_instruments (pd.DataFrame): DataFrame containing BLP instruments data.
    local_instruments (pd.DataFrame): DataFrame containing local instruments data.
    quadratic_instruments (pd.DataFrame): DataFrame containing quadratic instruments data.
    threshold_identified_earnings (float, optional): The threshold for filtering based on 'fraction_identified_earnings'. Default is 0.4.
    Returns:
    tuple: A tuple containing the filtered product_data, blp_instruments, local_instruments, and quadratic_instruments DataFrames.
    """
    condition = product_data['fraction_identified_earnings']>=threshold_identified_earnings
    kept_data = product_data.loc[condition].index

    product_data = product_data.loc[kept_data]

    local_instruments = local_instruments.loc[kept_data]
    quadratic_instruments = quadratic_instruments.loc[kept_data]
    blp_instruments = blp_instruments.loc[kept_data]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    return product_data, blp_instruments, local_instruments, quadratic_instruments


def filtering_data_by_number_brands(product_data: pd.DataFrame,

                                    blp_instruments: pd.DataFrame, 
                                    local_instruments: pd.DataFrame,
                                    quadratic_instruments: pd.DataFrame, 
                                    num_brands_by_market: int=2):
    """
    Filters the product data and corresponding instruments based on the number of brands in each market.
    Parameters:
    -----------
    product_data : pd.DataFrame
        DataFrame containing product data with a 'market_ids' column indicating the market each product belongs to.
    blp_instruments : pd.DataFrame
        DataFrame containing BLP instruments corresponding to the product data.
    local_instruments : pd.DataFrame
        DataFrame containing local instruments corresponding to the product data.
    quadratic_instruments : pd.DataFrame
        DataFrame containing quadratic instruments corresponding to the product data.
    num_brands_by_market : int, optional
        Minimum number of brands required in a market for it to be considered valid (default is 2).
    Returns:
    --------
    tuple
        A tuple containing the filtered product data, BLP instruments, local instruments, and quadratic instruments as DataFrames.
    """
    market_counts = product_data['market_ids'].value_counts()
    valid_markets = market_counts[market_counts >= num_brands_by_market].index
    product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    local_instruments = local_instruments.loc[product_data.index]
    quadratic_instruments = quadratic_instruments.loc[product_data.index]
    blp_instruments = blp_instruments.loc[product_data.index]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    return product_data, blp_instruments, local_instruments, quadratic_instruments


def matching_agent_and_product_data(product_data: pd.DataFrame, 

                                    agent_data: pd.DataFrame):
    """
    Matches agent and product data based on common market IDs.
    This function filters the agent and product data to include only the entries
    that have matching market IDs. It ensures that both dataframes contain only
    the market IDs that are present in both datasets.
    Parameters:
    product_data (pd.DataFrame): DataFrame containing product data with a 'market_ids' column.
    agent_data (pd.DataFrame): DataFrame containing agent data with a 'market_ids' column.
    Returns:
    tuple: A tuple containing two DataFrames:
        - agent_data (pd.DataFrame): Filtered agent data with matching market IDs.
        - product_data (pd.DataFrame): Filtered product data with matching market IDs.
    """
    agent_data = agent_data[agent_data['market_ids'].isin(set(product_data['market_ids']))]
    product_data = product_data[product_data['market_ids'].isin(agent_data['market_ids'].unique())]

    return agent_data, product_data


def save_processed_data(product_data, blp_instruments, local_instruments, quadratic_instruments, agent_data):
    """
    Saves processed data to CSV files in a specified directory.
    Parameters:
    product_data (pd.DataFrame): DataFrame containing product data, including a 'week_end' column.
    blp_instruments (pd.DataFrame): DataFrame containing BLP instruments data.
    local_instruments (pd.DataFrame): DataFrame containing local instruments data.
    quadratic_instruments (pd.DataFrame): DataFrame containing quadratic instruments data.
    agent_data (pd.DataFrame): DataFrame containing agent data.
    The function creates a directory based on the 'week_end' value in the product_data DataFrame.
    If there is only one unique 'week_end' value, it uses that value to create the directory.
    The function then saves each of the provided DataFrames as CSV files in the created directory.
    The filenames include the DIRECTORY_NAME and the current date.
    """

    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}', exist_ok=True)
    blp_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/blp_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    local_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/local_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    quadratic_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/quadratic_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    agent_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/agent_data_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)


def save_product_data(product_data: pd.DataFrame, output_dir: str):
    """
    Save product data to a CSV file in the specified output directory.
    Parameters:
    product_data (pd.DataFrame): The product data to be saved.
    output_dir (str): The directory where the CSV file will be saved.
    Returns:
    None
    """

    os.makedirs(output_dir, exist_ok=True)
    product_data.to_csv(os.path.join(output_dir, 'product_data.csv'), index=False)


def create_directories(product_data: pd.DataFrame):
    """
    Creates directories based on the week_end column in the provided product data.
    This function checks the 'week_end' column in the provided DataFrame. If there is only one unique value in the 
    'week_end' column, it uses that value to create two directories:
    - One for storing predicted data.
    - One for storing problem results in pickle format.
    Args:
        product_data (pd.DataFrame): A DataFrame containing product data with a 'week_end' column.
    Raises:
        KeyError: If the 'week_end' column is not present in the DataFrame.
        OSError: If there is an error creating the directories.
    """

    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}', exist_ok=True)
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}', exist_ok=True)


def select_product_data_columns(product_data: pd.DataFrame) -> pd.DataFrame:
    """
    Select specific columns from the product data DataFrame.
    Parameters:
    product_data (pd.DataFrame): The input DataFrame containing product data.
    Returns:
    pd.DataFrame: A DataFrame containing only the selected columns:
        - 'market_ids'
        - 'market_ids_string'
        - 'store_code_uc'
        - 'zip'
        - 'FIPS'
        - 'GESTFIPS'
        - 'fips_county_code'
        - 'week_end'
        - 'week_end_ID'
        - 'firm'
        - 'firm_ids'
        - 'firm_post_merger'
        - 'firm_ids_post_merger'
        - 'brand_code_uc'
        - 'brand_descr'
        - 'product_ids'
        - 'units'
        - 'total_individual_units'
        - 'total_units_retailer'
        - 'shares'
        - 'poblacion_census_2020'
        - 'total_income'
        - 'total_income_market_known_brands'
        - 'total_income_market'
        - 'fraction_identified_earnings'
        - 'prices'
        - 'from Nielsen'
        - 'from characteristics'
        - 'name'
        - 'tar'
        - 'nicotine'
        - 'co'
        - 'nicotine_mg_per_g'
        - 'nicotine_mg_per_g_dry_weight_basis'
        - 'nicotine_mg_per_cig'
    """

    return product_data[['market_ids', 
                        #  'market_ids_string',
                        'store_code_uc', 'zip', 'FIPS', 'GESTFIPS', 'fips_county_code',
                        'week_end', 'week_end_ID',
                        'firm', 'firm_ids', 'firm_post_merger', 'firm_ids_post_merger', 'brand_code_uc', 'brand_descr', 'product_ids', 
                        'units', 'total_individual_units', 'total_units_market',
                        'shares', 'poblacion_census_2020',
                        'total_income', 'total_income_market_known_brands', 'total_income_market', 'fraction_identified_earnings',
                        'prices',
                        'from Nielsen', 'from characteristics', 'name',
                        'tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig']]


def check_matrix_collinearity(matrix: pd.DataFrame, tolerance=1e-10):
    """
    Checks for collinearity in a matrix using SVD.

    Args:
        matrix: The input matrix (NumPy array).
        tolerance: Threshold for determining near-zero singular values.

    Returns:
        True if collinearity is detected, False otherwise.
    """
    _, s, _ = svd(matrix)
    return np.any(s < tolerance)


def find_first_non_collinear_matrix(**dfs):
    """
    Finds the first DataFrame in the list that does not exhibit collinearity.

    Args:
        df_list: A list of pandas DataFrames.

    Returns:
        The first DataFrame in the list that does not have collinear columns, 
        or None if all DataFrames exhibit collinearity.
    """
    for key, value in dfs.items():
        # matrix = df.values  # Convert DataFrame to NumPy array
        if not check_matrix_collinearity(value):
            print(key)
            return key, value
    return None


def compile_data(product_data: pd.DataFrame,
                blp_inst: pd.DataFrame, 
                local_inst: pd.DataFrame, 
                quad_inst: pd.DataFrame, 
                agent_data: pd.DataFrame):
    """
    Compiles and consolidates product and instrument data, renames columns, and filters based on agent data.
    Args:
        product_data (pd.DataFrame): DataFrame containing product data.
        blp_inst (pd.DataFrame): DataFrame containing BLP instruments.
        local_inst (pd.DataFrame): DataFrame containing local instruments.
        quad_inst (pd.DataFrame): DataFrame containing quadratic instruments.
        agent_data (pd.DataFrame): DataFrame containing agent data.
    Returns:
        pd.DataFrame: Consolidated and filtered product data with renamed columns.
    """

    inst_name, inst_values = find_first_non_collinear_matrix(local_inst=local_inst,
                                           quad_inst=quad_inst, 
                                           blp_inst=blp_inst)
    
    print(f'Los instrumentos usados para la actual regresión son {inst_name}')

    consolidated_product_data=pd.concat([product_data, inst_values], axis=1)
    dict_rename = rename_instruments(consolidated_product_data)
    consolidated_product_data=consolidated_product_data.rename(columns=dict_rename)

    # Restringe la información del consolidated_product_data a aquella que tienen información del consumidor en el agent_data
    consolidated_product_data = consolidated_product_data[consolidated_product_data['market_ids'].isin(agent_data['market_ids'].unique())]

    # Sort del product_data
    consolidated_product_data = consolidated_product_data.sort_values(by=['market_ids', 'product_ids'], ascending=[True, True], ignore_index=True)

    return consolidated_product_data




def run():
    date = datetime.datetime.today().strftime("%Y-%m-%d")
    datetime_=datetime.datetime.today()
    year=2014
    first_week=20
    num_weeks=5
    threshold_identified_earnings = 0.35
    optimization_algorithm = 'l-bfgs-b'

    #-----------------------------------------------------------------
    product_data = creating_product_data_rcl(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                     movements_path=f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/{year}/Movement_Files/4510_{year}/7460_{year}.tsv',
                                     stores_path='Nielsen_data/2014/Annual_Files/stores_2014.tsv',
                                     products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                     extra_attributes_path='Nielsen_data/2014/Annual_Files/products_extra_2014.tsv', 
                                     first_week=first_week,
                                     num_weeks=num_weeks,
                                     lower_threshold_identified_sales=threshold_identified_earnings=0.35)
    
    #-----------------------------------------------------------------
    # product_data = product_data[product_data['fraction_identified_earnings']>=threshold_identified_earnings]
    # market_counts = product_data['market_ids'].value_counts()
    # valid_markets = market_counts[market_counts >= 2].index
    # product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    #-----------------------------------------------------------------
    product_data = select_product_data_columns(product_data=product_data)
   
    #-------------------------------------------------------------
    # Crea directorios para guardar los datos procesados, las predicciones y los resultados de la optimización.
    week_dir = create_output_directories(product_data=product_data, date=date,optimization_algorithm=optimization_algorithm)

    #-----------------------------------------------------------------
    # Agregando información sociodemográfica 
    pop_agent_data = create_agent_data_from_cps(record_layout_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt',
                                                agent_data_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat')

    # socdem_file_structure = process_file('/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt')
    # agent_data_pop = pd.read_fwf('/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat', widths= [int(elem) for elem in socdem_file_structure.values()])
    # agent_data_pop.columns  = socdem_file_structure.keys()
    
    # # Elimina observaciones de los consumidores para los que el valor de condado es igual a 0.
    # agent_data_pop=agent_data_pop[agent_data_pop['GTCO']!=0]

    # # Genera la variable FIPS a partir de las variables GESTFIPS y GTCO.
    # agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']

    # # Resetea el índice del agent_data_pop.
    # agent_data_pop.reset_index(inplace=True, drop=True)

    # product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})
    


    #-----------------------------------------------------------------
    sample_agent_data = create_agent_data_sample(agent_data_pop=pop_agent_data, product_data=product_data)
    # agent_data_sample = get_random_samples_by_code(agent_data_pop, product_data['GESTFIPS'].unique(), 400)[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]
    # agent_data_sample.replace(-1, np.nan, inplace=True)

    # knn_imputer = KNNImputer(n_neighbors=3)
    # agent_data_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(agent_data_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]), columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 'ptdtrace_imputed', 'peeduca_imputed'])

    # grouped = agent_data_sample.groupby('GESTFIPS').size()

    # agent_data_sample['weights'] = agent_data_sample['GESTFIPS'].map(1 / grouped)
    # agent_data_sample = pd.concat([agent_data_sample[['FIPS', 'GESTFIPS','weights']],agent_data_sample_knn_imputed], axis=1)
    # agent_data_sample = add_random_nodes(agent_data_sample)

    # agent_data_sample = agent_data_sample[['FIPS', 'GESTFIPS', 'weights',
    #                                         'nodes0', 'nodes1', 'nodes2', 'nodes3','nodes4',
    #                                         'hefaminc_imputed', 'prtage_imputed','hrnumhou_imputed', 
    #                                         'ptdtrace_imputed', 'peeduca_imputed']]
    
    # agent_data = pd.merge(product_data[['market_ids', 'market_ids_string', 'GESTFIPS']].drop_duplicates(),
    #                                   agent_data_sample, 
    #                                   how='inner', 
    #                                   left_on='GESTFIPS',
    #                                   right_on='GESTFIPS')
    
    #-----------------------------------------------------------------
    ##### Filtrar base a partir de ventas identificadas########//TODO: Quitar esta sección dado que la eliminación de retailers con ventas identificadas inferiores a un threshold se hará al interior de la función creating_product_data_rcl
    condition = product_data['fraction_identified_earnings']>=threshold_identified_earnings
    kept_data = product_data.loc[condition].index
    product_data = product_data.loc[kept_data]

    ####### Restringiendo la muestra a retailers que tienen 2 o más marcas identificadas ######## //TODO: La restricción de los mercados a aquellos que tengan 2 o más marcas se debería implementar con anteriordad para evitar procesar información que posteriormente será eliminada. 
    market_counts = product_data['market_ids'].value_counts()
    valid_markets = market_counts[market_counts >= 2].index
    product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    ######### Manteniendo la información en agents y data con iguales market_ids ##########
    agent_data = agent_data[agent_data['market_ids'].isin(set(product_data['market_ids']))]
    product_data = product_data[product_data['market_ids'].isin(agent_data['market_ids'].unique())]


    product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/product_data_preins_{DIRECTORY_NAME}_{datetime_}.csv', index=False)

    ########## Creación de instrumentos ########## //TODO Revisar si las variables que se usan para crear los instrumentos también deben ser usadas al momento de definir el conjunto de características de los productos a ser analizados. 
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    blp_instruments = pd.DataFrame(blp_instruments)
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='local')
    local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])

    print(type(blp_instruments))
    print(type(local_instruments))
    print(type(quadratic_instruments))

    # local_instruments = local_instruments.loc[kept_data]
    # quadratic_instruments = quadratic_instruments.loc[kept_data]
    # blp_instruments = blp_instruments.loc[kept_data]

    # product_data.reset_index(drop=True, inplace=True)
    # blp_instruments.reset_index(drop=True, inplace=True)
    # local_instruments.reset_index(drop=True, inplace=True)
    # quadratic_instruments.reset_index(drop=True, inplace=True)


    # local_instruments = local_instruments.loc[product_data.index]
    # quadratic_instruments = quadratic_instruments.loc[product_data.index]
    # blp_instruments = blp_instruments.loc[product_data.index]


    ######### Salvando instrumentos e información de los consumidores ###########
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}', exist_ok=True)
    # product_data.to_csv(os.path.join(output_dir, f'product_data_{first_week}_{num_weeks}.csv'), index=False)
    
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/product_data_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    blp_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/blp_instruments_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    local_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/local_instruments_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    quadratic_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/quadratic_instruments_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    agent_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/agent_data_{DIRECTORY_NAME}_{datetime_}.csv', index=False)

    # Print all the locations where the DataFrames were saved
    print(f"Product data saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/product_data_{DIRECTORY_NAME}_{datetime_}.csv")
    print(f"BLP instruments saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/blp_instruments_{DIRECTORY_NAME}_{datetime_}.csv")
    print(f"Local instruments saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/local_instruments_{DIRECTORY_NAME}_{datetime_}.csv")
    print(f"Quadratic instruments saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/quadratic_instruments_{DIRECTORY_NAME}_{datetime_}.csv")
    print(f"Agent data saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/agent_data_{DIRECTORY_NAME}_{datetime_}.csv")
    print(f"Compiled product data saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/compiled_data_{DIRECTORY_NAME}_{datetime_}.csv")

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)
    agent_data.reset_index(drop=True, inplace=True)

    product_data = compile_data(product_data = product_data, 
                            blp_inst = blp_instruments, 
                            local_inst = local_instruments, 
                            quad_inst = quadratic_instruments, 
                            agent_data= agent_data)
    
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/compiled_data_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    print(f'empezando optimización {datetime_}')
    iter =  0

    #logit formulation 
    linear_formulation=pyblp.Formulation('1+ prices', absorb='C(product_ids)')
    non_linear_formulation=pyblp.Formulation('1+ prices + tar')
    agent_formulation=pyblp.Formulation('0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed')

    plain_logit_results=plain_logit(product_data=product_data, formulation=linear_formulation)

    while iter <= 100:
        print('------------------------------')
        print(iter)
        print('------------------------------')
        try:
            results= rcl_with_demographics(product_data=product_data, 
                                           agent_data=agent_data,
                                           linear_formulation=linear_formulation,
                                           non_linear_formulation=non_linear_formulation,
                                           agent_formulation=agent_formulation,
                                           logit_results=plain_logit_results)
        

            results.to_pickle(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/{date}/{optimization_algorithm}/iteration_{iter}.pickle')
            print(f'------------results {iter}------------------')
            if results.converged == True:
                initial_prices = original_prices(product_data=product_data, results=results)


                #predicting the prices and appending the information to a dataframe
                predicted_prices = predict_prices(product_data = product_data, results = results, merger=[3,0])
                predicted_prices = predicted_prices.tolist()
                price_pred_df = product_data[['market_ids','market_ids_string','store_code_uc', 'week_end', 'product_ids', 'brand_code_uc', 'brand_descr']].copy()
                price_pred_df.loc[:, 'price_prediction'] = predicted_prices 


                #saving the file
                price_pred_df.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/{date}/{optimization_algorithm}/price_predictions_{iter}.json', index=False)
                print('predictions saved')

            # optimal_results = results_optimal_instruments(results)
        except Exception as e:
            print(e)
            

        iter += 1
        

    print('fin')



def run2():
    product_data = creating_product_data_rcl(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                     movements_path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/2014/Movement_Files/4510_2014/7460_2014.tsv' ,
                                     stores_path='Nielsen_data/2014/Annual_Files/stores_2014.tsv' ,
                                     products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                     extra_attributes_path='Nielsen_data/2014/Annual_Files/products_extra_2014.tsv', 
                                     first_week=16,
                                     num_weeks=1, 
                                     fractioned_identfied_earning=0.3)
    
    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None


    ######### Save product_data DataFrame to the specified directory ###########
    save_product_data(product_data, '/oak/stanford/groups/polinsky/Mergers/Cigarettes/Pruebas')

    ########## Creación de instrumentos ##########
    formulation, blp_instruments, local_instruments, quadratic_instruments = creating_instruments_data(product_data=product_data)

    ####### Agregando información sociodemográfica #########
    agent_data = creating_agent_data(product_data=product_data, 
                                    record_layout_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt',
                                    agent_data_pop='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat')

    ##### Filtrar base a partir de ventas identificadas########
    product_data, blp_instruments, local_instruments, quadratic_instruments = filtering_data_by_identified_sales(product_data= product_data,
                                                                                                                 blp_instruments=blp_instruments,
                                                                                                                 local_instruments= local_instruments,
                                                                                                                 quadratic_instruments=quadratic_instruments,
                                                                                                                 threshold_identified_earnings=0.3)

    ####### Restringiendo la muestra a retailers que tienen 2 o más marcas identificadas ######## 
    product_data, blp_instruments, local_instruments, quadratic_instruments = filtering_data_by_number_brands(product_data=product_data,
                                                                                                              blp_instruments=blp_instruments, 
                                                                                                              local_instruments=local_instruments,
                                                                                                              quadratic_instruments=quadratic_instruments,
                                                                                                              num_brands_by_market = 2)    

    ######### Manteniendo la información en agents y data con iguales market_ids ##########
    agent_data, product_data = matching_agent_and_product_data(product_data = product_data,
                                                               agent_data = agent_data)

    ######### Salvando datos ###########
    create_directories(product_data)
    save_processed_data(blp_instruments, local_instruments, quadratic_instruments, agent_data)

    ###### Random coefficients logit ########
    iter =  0
    while iter <= 20:
        print('------------------------------')
        print(iter)
        print('------------------------------')
        try:
            results, consolidated_product_data = rcl_with_demographics(product_data=product_data,
                                                                        blp_inst=blp_instruments,
                                                                        local_inst=local_instruments,
                                                                        quad_inst=quadratic_instruments,
                                                                        agent_data=agent_data)
            # if results.converged == True:
            #     iter += 1

            optimal_results = results_optimal_instruments(results=results)

            optimal_results.to_pickle(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/iteration_{iter}.pickle')
            
            predicted_prices = predict_prices(product_data = consolidated_product_data,
                                                results = results, 
                                                merger=[3,0])

            # predicted_prices_path = f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/iteration_{iter}.json'
            predicted_prices = predicted_prices.tolist()
            price_pred_df = consolidated_product_data[['market_ids','market_ids_string','store_code_uc', 'week_end', 'product_ids', 'brand_code_uc', 'brand_descr']].copy()
            price_pred_df.loc[:, 'price_prediction'] = predicted_prices 
            price_pred_df.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/price_predictions_{iter}.json', index=False)

            # optimal_results = results_optimal_instruments(results)
        except Exception as e:
            print(e)
            

        iter += 1
        

    print('fin')


if __name__=='__main__': 
    run()
