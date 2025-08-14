import numpy as np
import os 
import pandas as pd 
import pyblp
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


def filter_by_identified_earnings(product_data, threshold_identified_earnings):
    """Filter product data based on fraction of identified earnings threshold.
    
    Args:
        product_data (pd.DataFrame): Product data to filter
        threshold_identified_earnings (float): Minimum threshold for fraction of identified earnings
        
    Returns:
        pd.DataFrame: Filtered product data containing only rows above the threshold
    """
    condition = product_data['fraction_identified_earnings'] >= threshold_identified_earnings
    kept_data = product_data.loc[condition].index
    return product_data.loc[kept_data]


def filter_by_market_size(product_data, min_brands=2):
    """Filter product data to keep only markets with minimum number of brands.
    
    Args:
        product_data (pd.DataFrame): Product data to filter
        min_brands (int): Minimum number of brands required per market (default 2)
        
    Returns:
        pd.DataFrame: Filtered product data containing only markets with sufficient brands
    """
    market_counts = product_data['market_ids'].value_counts()
    valid_markets = market_counts[market_counts >= min_brands].index
    return product_data[product_data['market_ids'].isin(valid_markets)]


def filter_matching_markets(agent_data, product_data):
    """Filter agent and product data to keep only matching market IDs.
    
    Args:
        agent_data (pd.DataFrame): Agent data containing market_ids
        product_data (pd.DataFrame): Product data containing market_ids
        
    Returns:
        tuple: (filtered_agent_data, filtered_product_data) containing only matching market IDs
    """
    # Keep only market_ids that exist in both datasets
    agent_data = agent_data[agent_data['market_ids'].isin(set(product_data['market_ids']))]
    product_data = product_data[product_data['market_ids'].isin(agent_data['market_ids'].unique())]
    
    return agent_data, product_data


def create_instruments(product_data: pd.DataFrame, formulation: str='0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig') -> tuple:
    """Create BLP, local and quadratic instruments from product data.
    
    Args:
        product_data (pd.DataFrame): Product data to generate instruments from
        formulation (pyblp.Formulation): PyBLP formulation specifying variables
        
    Returns:
        tuple: (blp_instruments, local_instruments, quadratic_instruments) DataFrames containing generated instruments
    """
    # Update formulation with product characteristics
    formulation = pyblp.Formulation(formulation)
    # Create BLP instruments
    blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    blp_instruments = pd.DataFrame(blp_instruments)
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    # Create local instruments
    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='local')
    local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    # Create quadratic instruments  
    quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])

    return blp_instruments, local_instruments, quadratic_instruments


def save_processed_data(product_data: pd.DataFrame, 
                       blp_instruments: pd.DataFrame,
                       local_instruments: pd.DataFrame, 
                       quadratic_instruments: pd.DataFrame,
                       agent_data: pd.DataFrame,
                       week_dir: str,
                       date: str,
                       directory_name: str,
                       datetime_str: str) -> None:
    """
    Save processed data files to specified directory structure.
    
    Args:
        product_data (pd.DataFrame): Product data to save
        blp_instruments (pd.DataFrame): BLP instruments data
        local_instruments (pd.DataFrame): Local instruments data 
        quadratic_instruments (pd.DataFrame): Quadratic instruments data
        agent_data (pd.DataFrame): Agent data
        week_dir (str): Week directory name
        date (str): Date string
        directory_name (str): Name of directory
        datetime_str (str): Datetime string for filenames
        
    Returns:
        None
    """
    # Create base output directory path
    base_dir = f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}'
    os.makedirs(base_dir, exist_ok=True)

    # Define file paths
    file_paths = {
        'product_data': f'{base_dir}/product_data_{directory_name}_{datetime_str}.csv',
        'blp_instruments': f'{base_dir}/blp_instruments_{directory_name}_{datetime_str}.csv',
        'local_instruments': f'{base_dir}/local_instruments_{directory_name}_{datetime_str}.csv',
        'quadratic_instruments': f'{base_dir}/quadratic_instruments_{directory_name}_{datetime_str}.csv',
        'agent_data': f'{base_dir}/agent_data_{directory_name}_{datetime_str}.csv'
    }

    # Save each DataFrame and print confirmation
    data_frames = {
        'product_data': product_data,
        'blp_instruments': blp_instruments,
        'local_instruments': local_instruments,
        'quadratic_instruments': quadratic_instruments,
        'agent_data': agent_data}

    # Reset index for all DataFrames before saving
    for df in data_frames.values():
        df.reset_index(drop=True, inplace=True)

    for name, df in data_frames.items():
        df.to_csv(file_paths[name], index=False)
        print(f"{name.replace('_', ' ').title()} saved to: {file_paths[name]}")

