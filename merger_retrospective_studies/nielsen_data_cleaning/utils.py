import numpy as np
import os 
import pandas as pd 
import pyblp
from scipy.linalg import svd
from sklearn.impute import KNNImputer
from typing import Union, Tuple


from .consumidores_sociodemograficas import process_file
from .consumidores_sociodemograficas import get_random_samples_by_code, add_random_nodes
from ..estimaciones.rcl_with_demographics import rcl_with_demographics
from ..estimaciones.post_estimation_merger_simulation import predict_prices



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

    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{date}/{week_dir}', exist_ok=True)
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{date}/{week_dir}/{optimization_algorithm}', exist_ok=True)
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{date}/{week_dir}/{optimization_algorithm}', exist_ok=True)

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
    
    # Keep observations where county code is not 0
    agent_data_pop = agent_data_pop[agent_data_pop['GTCO']!=0]

    # Generate FIPS code from state and county codes
    agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']

    # Reset index
    agent_data_pop.reset_index(inplace=True, drop=True)
    
    return agent_data_pop



def create_agent_data_sample(agent_data_pop: pd.DataFrame, product_data: pd.DataFrame, num_samples: int=400, n_neighbors: int=4, num_nodes: int=5) -> pd.DataFrame:
    """
    Creates a sample of agent data with demographic information and random nodes.
    
    Args:
        agent_data_pop (pd.DataFrame): Population agent data with demographic variables
        product_data (pd.DataFrame): Product data containing GESTFIPS codes
        num_samples (int): Number of samples to draw
        n_neighbors (int): Number of neighbors to use for KNN imputation
        num_nodes (int): Number of nodes to add
    Returns:
        pd.DataFrame: Sample of agent data with demographic variables, weights and random nodes
    """
    # Get random samples by state code
    agent_data_sample = get_random_samples_by_code(
                        agent_data_pop, 
                        product_data['GESTFIPS'].unique(), 
                        num_samples)
    
    agent_data_sample = agent_data_sample[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]

    # Replace -1 with NaN
    agent_data_sample.replace(-1, np.nan, inplace=True)

    # Impute missing values
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    agent_data_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(agent_data_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]), columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 'ptdtrace_imputed', 'peeduca_imputed'])

    # Calculate weights
    grouped = agent_data_sample.groupby('GESTFIPS').size() #computes the number of observations per GESTFIPS
    agent_data_sample['weights'] = agent_data_sample['GESTFIPS'].map(1 / grouped)
    
    # Combine data and add random nodes
    agent_data_sample = pd.concat([agent_data_sample[['FIPS', 'GESTFIPS','weights']], agent_data_sample_knn_imputed],     
                        axis=1)
    agent_data_sample = add_random_nodes(agent_data_sample, mean=0, std_dev=1, num_nodes=num_nodes)

    # Select final columns
    # Create dynamic list of node columns based on num_nodes parameter
    node_columns = [f'nodes{i}' for i in range(num_nodes)]
    
    # Select final columns with dynamic node columns
    final_columns = ['FIPS', 'GESTFIPS', 'weights'] + node_columns + [
        'hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 'ptdtrace_imputed', 'peeduca_imputed'
    ]
    
    agent_data_sample = agent_data_sample[final_columns]
    
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


def filtering_agent_data_match_markets(agent_data, product_data):
    """Filter agent and product data to keep only matching market IDs.
    
    Args:
        agent_data (pd.DataFrame): Agent data containing market_ids
        product_data (pd.DataFrame): Product data containing market_ids
        
    Returns:
        tuple: (filtered_agent_data, filtered_product_data) containing only matching market IDs
    """
    # Keep only market_ids that exist in both datasets
    common_market_ids = set(product_data['market_ids']).intersection(set(agent_data['market_ids']))
    agent_data = agent_data[agent_data['market_ids'].isin(common_market_ids)]
    product_data = product_data[product_data['market_ids'].isin(common_market_ids)]
    
    return agent_data


def create_instruments(product_data: pd.DataFrame, formulation: str='0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig') -> tuple:
    """Create BLP, local and quadratic instruments from product data.
    
    Args:
        product_data (pd.DataFrame): Product data to generate instruments from
        formulation (pyblp.Formulation): PyBLP formulation specifying variables
        
    Returns:
        tuple: (blp_instruments, local_instruments, quadratic_instruments) DataFrames containing generated instruments
    """
    # Update formulation with product characteristics
    # formulation = pyblp.Formulation(formula)
    # Create BLP instruments
    blp_instruments = pd.DataFrame(pyblp.build_blp_instruments(formulation, product_data))
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    # Create local instruments
    local_instruments = pd.DataFrame(pyblp.build_differentiation_instruments(formulation, product_data, version='local'))
    local_instruments.rename(columns={i:f'local_instruments{i}' for i in local_instruments.columns}, inplace=True)

    # Create quadratic instruments  
    quadratic_instruments = pd.DataFrame(pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic'))
    quadratic_instruments.rename(columns={i:f'quadratic_instruments{i}' for i in quadratic_instruments.columns}, inplace=True)

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
    base_dir = f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{date}/{week_dir}'
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


def create_formulations(linear_formula: str = '1+ prices',
                       absorb_formula: str = 'C(product_ids)',
                       nonlinear_formula: str = '1+ prices + tar',
                       agent_formula: str = '0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed') -> tuple:
    """
    Creates the standard formulations used for demand estimation.
    
    Args:
        linear_formula (str): Formula string for linear parameters. Default: '1+ prices'
        absorb_formula (str): Formula string for absorption (fixed effects). Default: 'C(product_ids)'
        nonlinear_formula (str): Formula string for non-linear parameters. Default: '1+ prices + tar'
        agent_formula (str): Formula string for agent demographics. Default: '0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed'
    
    Returns:
        tuple: A tuple containing (linear_formulation, non_linear_formulation, agent_formulation)
            - linear_formulation: pyblp.Formulation for linear parameters
            - non_linear_formulation: pyblp.Formulation for non-linear parameters  
            - agent_formulation: pyblp.Formulation for agent demographics
    """
    linear_formulation = pyblp.Formulation(linear_formula, absorb=absorb_formula)
    non_linear_formulation = pyblp.Formulation(nonlinear_formula)
    agent_formulation = pyblp.Formulation(agent_formula)
    
    return linear_formulation, non_linear_formulation, agent_formulation


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


def select_lowest_condition_dataframe(*dataframes: pd.DataFrame, 
                                      return_condition_number: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float]]:
    """
    Calculate the condition number for multiple DataFrames and return the one with the lowest value.
    
    The condition number is calculated using the L2 norm of the matrix formed by the numeric columns
    of each DataFrame. Only numeric columns are considered for the calculation.
    
    Parameters:
    -----------
    *dataframes : pd.DataFrame
        Variable number of pandas DataFrames to compare
    return_condition_number : bool, default False
        If True, returns a tuple of (dataframe, condition_number)
        If False, returns only the dataframe
    
    Returns:
    --------
    pd.DataFrame or Tuple[pd.DataFrame, float]
        The DataFrame with the lowest condition number, optionally with the condition number value
    
    Raises:
    -------
    ValueError
        If no DataFrames are provided, if any DataFrame is empty, 
        or if no numeric columns are found in any DataFrame
    """
    
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")
    
    condition_numbers = []
    valid_dataframes = []
    
    for i, df in enumerate(dataframes):
        if df.empty:
            raise ValueError(f"DataFrame at index {i} is empty")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError(f"DataFrame at index {i} has no numeric columns")
        
        # Handle missing values by dropping rows with any NaN values
        # Alternative: you could use fillna() or other imputation methods
        clean_df = numeric_df.dropna()
        
        if clean_df.empty:
            raise ValueError(f"DataFrame at index {i} has no data after removing NaN values")
        
        # Calculate condition number using numpy's linear algebra module
        try:
            condition_num = np.linalg.cond(clean_df.values)
            condition_numbers.append(condition_num)
            valid_dataframes.append(df)
        except np.linalg.LinAlgError:
            raise ValueError(f"Unable to calculate condition number for DataFrame at index {i}")
    
    # Find the index of the minimum condition number
    min_index = np.argmin(condition_numbers)
    best_dataframe = valid_dataframes[min_index]
    best_condition_number = condition_numbers[min_index]
    
    if return_condition_number:
        return best_dataframe, best_condition_number
    else:
        return best_dataframe


def select_lowest_condition_dataframe(*dataframes: pd.DataFrame, 
                                      return_condition_number: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float]]:
    """
    Calculate the condition number for multiple DataFrames and return the one with the lowest value.
    
    The condition number is calculated using the L2 norm of the matrix formed by the numeric columns
    of each DataFrame. Only numeric columns are considered for the calculation.
    
    Parameters:
    -----------
    *dataframes : pd.DataFrame
        Variable number of pandas DataFrames to compare
    return_condition_number : bool, default False
        If True, returns a tuple of (dataframe, condition_number)
        If False, returns only the dataframe
    
    Returns:
    --------
    pd.DataFrame or Tuple[pd.DataFrame, float]
        The DataFrame with the lowest condition number, optionally with the condition number value
    
    Raises:
    -------
    ValueError
        If no DataFrames are provided, if any DataFrame is empty, 
        or if no numeric columns are found in any DataFrame
    """
    
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided")
    
    condition_numbers = []
    valid_dataframes = []
    
    for i, df in enumerate(dataframes):
        if df.empty:
            raise ValueError(f"DataFrame at index {i} is empty")
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError(f"DataFrame at index {i} has no numeric columns")
        
        # Handle missing values by dropping rows with any NaN values
        # Alternative: you could use fillna() or other imputation methods
        clean_df = numeric_df.dropna()
        
        if clean_df.empty:
            raise ValueError(f"DataFrame at index {i} has no rows after removing missing values")
        
        try:
            # Calculate condition number using L2 norm
            matrix = clean_df.values
            condition_num = np.linalg.cond(matrix)
            
            # Check for infinite condition number (singular matrix)
            if np.isinf(condition_num):
                print(f"Warning: DataFrame at index {i} has infinite condition number (singular matrix)")
            
            condition_numbers.append(condition_num)
            valid_dataframes.append(df)
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Linear algebra error for DataFrame at index {i}: {str(e)}")
    
    # Find the index of the DataFrame with the lowest condition number
    min_index = np.argmin(condition_numbers)
    best_dataframe = valid_dataframes[min_index]
    lowest_condition_number = condition_numbers[min_index]
    
    return best_dataframe
 

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
        - 'total_income_per_market_id_known_brands'
        - 'total_income_per_market_id'
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
                        'total_income', 'total_income_per_market_id_known_brands', 'total_income_per_market_id', 'fraction_identified_earnings',
                        'prices',
                        'from Nielsen', 'from characteristics', 'name',
                        'tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig']]


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


def save_processed_data_old(product_data, blp_instruments, local_instruments, quadratic_instruments, agent_data):
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


def run_optimization_iterations(product_data: pd.DataFrame, 
                                filtered_sample_agent_data: pd.DataFrame, 
                                week_dir: str, 
                                date: str, 
                                optimization_algorithm: str, 
                                num_iterations: int, 
                                linear_formulation: pyblp.Formulation, 
                                non_linear_formulation: pyblp.Formulation, 
                                agent_formulation: pyblp.Formulation, 
                                plain_logit_results) -> None:
    """
    Runs the optimization iterations for demand estimation and price prediction.
    
    Args:
        product_data: DataFrame containing product data
        filtered_sample_agent_data: DataFrame containing filtered agent data
        week_dir: Directory name for the week
        date: Date string for file naming
        optimization_algorithm: Algorithm name for optimization
        num_iterations: Number of iterations to run
        linear_formulation: pyblp.Formulation for linear parameters
        non_linear_formulation: pyblp.Formulation for non-linear parameters
        agent_formulation: pyblp.Formulation for agent demographics
        plain_logit_results: Results from plain logit estimation
        
    Returns:
        None: Saves results to pickle files and price predictions to JSON files
    """
    
    iter = 1
    while iter <= num_iterations:
        print('------------------------------')
        # try:
        results = rcl_with_demographics(product_data=product_data, 
                                        agent_data=filtered_sample_agent_data,
                                        linear_formulation=linear_formulation,
                                        non_linear_formulation=non_linear_formulation,
                                        agent_formulation=agent_formulation,
                                        logit_results=plain_logit_results,
                                        week_dir=week_dir,
                                        date=date,
                                        iter=iter)
    
        # results.to_pickle(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/{date}/{optimization_algorithm}/iteration_{iter}.pickle')

        results.to_pickle(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{date}/{week_dir}/{optimization_algorithm}/iteration_{iter}.pickle')


        # predicted_prices=compute_and_save_price_predictions(product_data=product_data, 
        #                                     results=results, 
        #                                     week_dir=week_dir, 
        #                                     date=date, 
        #                                     optimization_algorithm=optimization_algorithm, 
        #                                     iter=iter)

        # optimal_results = results_optimal_instruments(results)
        # except Exception as e:
        #     print(e)
            
        iter += 1
        
        
def compute_and_save_price_predictions(product_data: pd.DataFrame, 
                                     results: pyblp.ProblemResults, 
                                     week_dir: str, 
                                     date: str, 
                                     optimization_algorithm: str, 
                                     iter: int) -> None:
    """
    Compute price predictions after merger and save to JSON file.
    
    Args:
        product_data: DataFrame containing product data
        results: Results from demand estimation
        week_dir: Directory name for the week
        date: Date string for file naming
        optimization_algorithm: Algorithm name for optimization
        iter: Iteration number for file naming
        merger: List defining merger scenario, default [3,0]
        
    Returns:
        None
    """
    #computing the initial prices
    costs = results.compute_costs()

    predicted_prices = results.compute_prices(firm_ids=product_data['firm_ids_post_merger'], costs=costs)

    predicted_prices = predicted_prices.tolist()
    price_pred_df = product_data[['market_ids','store_code_uc', 'week_end', 'product_ids', 'brand_code_uc', 'brand_descr']].copy()
    price_pred_df.loc[:, 'price_prediction'] = predicted_prices 

    #saving the file
    price_pred_df.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{date}/{week_dir}/{optimization_algorithm}/price_predictions_{iter}.json', index=False)
    print('predictions saved')
    return predicted_prices

