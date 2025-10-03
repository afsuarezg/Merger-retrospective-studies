import pickle
import pandas as pd
import numpy as np
from typing import List, Union
import os


class SimulationResultsDataFrame:
    """
    A class to process simulation results from pickle files and organize them into a pandas DataFrame.
    
    Each row represents one simulation with columns for:
    - objective function value
    - projected gradient norm
    - min/max reduced hessian eigenvalues
    - sigma, pi, beta, and beta_se parameters
    """
    
    def __init__(self):
        self.results_df = None
        
    def load_pickle_file(self, file_path: str):
        """Load a single pickle file containing simulation results."""
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    
    def extract_simulation_data(self, results) -> dict:
        """
        Extract relevant data from a single simulation result object.
        
        Parameters:
        -----------
        results : object
            The loaded pickle object containing simulation results
            
        Returns:
        --------
        dict : Dictionary containing extracted values
        """
        data = {}
        
        # Extract objective function value
        data['objective'] = results.objective[0, 0] if hasattr(results.objective, 'shape') else results.objective
        
        # Extract projected gradient norm
        data['projected_gradient_norm'] = float(results.projected_gradient_norm)
        
        # Extract min and max reduced hessian eigenvalues
        eigenvalues = results.reduced_hessian_eigenvalues
        data['min_reduced_hessian'] = float(eigenvalues[0])
        data['max_reduced_hessian'] = float(eigenvalues[-1])
        
        # Extract sigma with variable names
        sigma_labels = results.sigma_labels
        sigma_flat = results.sigma.flatten()
        idx = 0
        for i, row_label in enumerate(sigma_labels):
            for j, col_label in enumerate(sigma_labels):
                data[f'sigma_{row_label}_{col_label}'] = float(sigma_flat[idx])
                idx += 1
        
        # Extract pi with variable names
        pi_labels = results.pi_labels
        sigma_labels = results.sigma_labels
        pi_flat = results.pi.flatten()
        idx = 0
        for i, row_label in enumerate(sigma_labels):
            for j, col_label in enumerate(pi_labels):
                data[f'pi_{row_label}_{col_label}'] = float(pi_flat[idx])
                idx += 1
        
        # Extract beta with variable names (assuming beta_labels exists, otherwise use generic names)
        beta_flat = results.beta.flatten()
        if hasattr(results, 'beta_labels'):
            beta_labels = results.beta_labels
            for i, (label, val) in enumerate(zip(beta_labels, beta_flat)):
                data[f'beta_{label}'] = float(val)
        else:
            # If no labels, try to infer from the output structure
            for i, val in enumerate(beta_flat):
                data[f'beta_{i}'] = float(val)
        
        # Extract beta_se with variable names
        beta_se_flat = results.beta_se.flatten()
        if hasattr(results, 'beta_labels'):
            beta_labels = results.beta_labels
            for i, (label, val) in enumerate(zip(beta_labels, beta_se_flat)):
                data[f'beta_se_{label}'] = float(val)
        else:
            for i, val in enumerate(beta_se_flat):
                data[f'beta_se_{i}'] = float(val)
        
        return data
    
    def process_pickle_files(self, file_paths: List[str], add_file_path: bool = True) -> pd.DataFrame:
        """
        Process multiple pickle files and create a DataFrame with all simulation results.
        
        Parameters:
        -----------
        file_paths : List[str]
            List of file paths to pickle files
        add_file_path : bool, optional
            Whether to include the file path as a column (default: True)
            
        Returns:
        --------
        pd.DataFrame : DataFrame containing all simulation results
        """
        all_results = []
        
        for file_path in file_paths:
            try:
                # Load the pickle file
                results = self.load_pickle_file(file_path)
                
                # Extract simulation data
                sim_data = self.extract_simulation_data(results)
                
                # Optionally add the file path
                if add_file_path:
                    sim_data['file_path'] = file_path
                
                all_results.append(sim_data)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Create DataFrame
        self.results_df = pd.DataFrame(all_results)
        
        return self.results_df
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the stored DataFrame.
        
        Returns:
        --------
        pd.DataFrame : The results DataFrame
        """
        if self.results_df is None:
            raise ValueError("No data has been processed yet. Run process_pickle_files() first.")
        return self.results_df
    
    def save_to_csv(self, output_path: str):
        """
        Save the DataFrame to a CSV file.
        
        Parameters:
        -----------
        output_path : str
            Path where the CSV file should be saved
        """
        if self.results_df is None:
            raise ValueError("No data has been processed yet. Run process_pickle_files() first.")
        self.results_df.to_csv(output_path, index=False)
        print(f"DataFrame saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Initialize the class
    sim_results = SimulationResultsDataFrame()
    
    # Get list of pickle files
    files_full_path = []
    pickle_dir = os.getcwd()  # Update this path
    
    for file in os.listdir(pickle_dir):
        if file.endswith('.pickle'):
            files_full_path.append(os.path.join(pickle_dir, file))
    
    # Process all pickle files and create DataFrame
    df = sim_results.process_pickle_files(files_full_path)
    
    # Display the DataFrame
    print(df.head())
    
    # Optionally save to CSV
    # sim_results.save_to_csv("simulation_results.csv")