import numpy as np
import os 
import pandas as pd 
from sklearn.impute import KNNImputer

from .consumidores_sociodemograficas import process_file
from .consumidores_sociodemograficas import get_random_samples_by_code, add_random_nodes



def create_output_directories(product_data: pd.DataFrame, date: str, optimization_algorithm: str) -> str:
    """
    Creates output directories for processed data, predictions and optimization results.
    
    Args:
        product_data (pd.DataFrame): DataFrame containing product data with 'week_end' column
        date (str): Date string in YYYY-MM-DD format
        optimization_algorithm (str): Name of optimization algorithm used
        
    Returns:
        str: The week directory name created
        
    Raises:
        ValueError: If no weeks are found in the product data
    """
    weeks = sorted(set(product_data['week_end']))

    if len(weeks) < 1:
        raise ValueError('No se encontraron semanas en la base de datos')
    else:
        week_dir = weeks[0]

    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}', exist_ok=True)
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/{date}/{optimization_algorithm}', exist_ok=True)
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/{date}/{optimization_algorithm}', exist_ok=True)
    
    return week_dir


def create_agent_data_from_cps(record_layout_path: str, agent_data_path: str) -> pd.DataFrame:
    """
    Creates agent data by processing CPS data files.
    
    Args:
        record_layout_path (str): Path to the CPS record layout file
        agent_data_path (str): Path to the CPS agent data file
        
    Returns:
        pd.DataFrame: Processed agent data with FIPS codes
    """
    # Process record layout file
    socdem_file_structure = process_file(record_layout_path)
    
    # Read agent data using fixed width format
    agent_data_pop = pd.read_fwf(agent_data_path, 
                                widths=[int(elem) for elem in socdem_file_structure.values()])
    
    agent_data_pop.columns = socdem_file_structure.keys()
    
    # Remove observations where county code is 0
    agent_data_pop = agent_data_pop[agent_data_pop['GTCO']!=0]

    # Generate FIPS code from state and county codes
    agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']

    # Reset index
    agent_data_pop.reset_index(inplace=True, drop=True)
    
    return agent_data_pop



def create_agent_data_sample(agent_data_pop: pd.DataFrame, product_data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a sample of agent data with demographic information and random nodes.
    
    Args:
        agent_data_pop (pd.DataFrame): Population agent data with demographic variables
        product_data (pd.DataFrame): Product data containing GESTFIPS codes
        
    Returns:
        pd.DataFrame: Sample of agent data with demographic variables, weights and random nodes
    """
    # Get random samples by state code
    agent_data_sample = get_random_samples_by_code(
                        agent_data_pop, 
                        product_data['GESTFIPS'].unique(), 
                        400)
    
    agent_data_sample = agent_data_sample[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]

    # Replace -1 with NaN
    agent_data_sample.replace(-1, np.nan, inplace=True)

    # Impute missing values
    knn_imputer = KNNImputer(n_neighbors=3)

    agent_data_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(agent_data_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]), columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 'ptdtrace_imputed', 'peeduca_imputed'])

    # Calculate weights
    grouped = agent_data_sample.groupby('GESTFIPS').size()
    agent_data_sample['weights'] = agent_data_sample['GESTFIPS'].map(1 / grouped)
    
    # Combine data and add random nodes
    agent_data_sample = pd.concat([agent_data_sample[['FIPS', 'GESTFIPS','weights']], agent_data_sample_knn_imputed],     
                        axis=1)

    agent_data_sample = add_random_nodes(agent_data_sample, mean=0, std_dev=1, num_nodes=5)

    # Select final columns
    agent_data_sample = agent_data_sample[[
        'FIPS', 'GESTFIPS', 'weights',
        'nodes0', 'nodes1', 'nodes2', 'nodes3','nodes4',
        'hefaminc_imputed', 'prtage_imputed','hrnumhou_imputed', 
        'ptdtrace_imputed', 'peeduca_imputed'
    ]]
    
    # Merge with product data
    agent_data = pd.merge(
                product_data[['market_ids', 'GESTFIPS']].drop_duplicates(),
                agent_data_sample, 
                how='inner', 
                left_on='GESTFIPS',
                right_on='GESTFIPS')
    
    return agent_data







