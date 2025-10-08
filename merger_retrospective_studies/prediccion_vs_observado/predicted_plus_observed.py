import os
import sys
import json
import pandas as pd
import numpy as np
import random

from .predicted_prices_df import read_pickles_from_folder, load_compiled_product_data, add_predictions_from_problem_results, map_store_to_brands
from .prediction_observation_comparison import PredictionObservationComparison



def filter_columns(df):
    """
    Filter DataFrame to include only specific columns:
    - store_code_uc
    - brand_code_uc
    - columns starting with 'price_prediction'
    - columns starting with 'product_data_completo'
    
    Args:
        df (pd.DataFrame): Input DataFrame to filter
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only the specified columns
    """
    # Get all columns that start with 'price_prediction' or 'product_data_completo'
    price_prediction_cols = [col for col in df.columns if col.startswith('price_prediction')]
    product_data_cols = [col for col in df.columns if col.startswith('product_data_completo')]
    
    # Combine all desired columns
    desired_columns = ['store_code_uc', 'brand_code_uc'] + price_prediction_cols + product_data_cols
    
    # Filter to only include columns that exist in the DataFrame
    existing_columns = [col for col in desired_columns if col in df.columns]
    
    return df[existing_columns]


def main():
    pass

	

if __name__ == "__main__":
    # Known folder with results
    folder = r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\Research\Merger retrospective studies\Codigo\repos\merger_retrospective_studies\data_samples\ProblemResults_sample"

    if not os.path.isdir(folder):
        print(f"ProblemResults_sample folder not found: {folder}")
        sys.exit(1)

    simulation_results = read_pickles_from_folder(folder)
    # print(results)

    # breakpoint()
    # Load compiled product data CSV
    base_data = load_compiled_product_data()

    base_data_plus_predictions = add_predictions_from_problem_results(base_data, simulation_results)

    #observed prices
    csv_file_path = "merger_retrospective_studies/prediccion_vs_observado/filtered_observed_prices.csv"

    print(f"Reading filtered observed prices from: {csv_file_path}")
    df = pd.read_csv(csv_file_path, index_col=[0, 1])  # Assuming first two columns are store_code and brand_code
    df = df.reset_index()
    # Display basic information about the data
    print(f"\nData loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index levels: {df.index.names}")

    merged_df = pd.merge(base_data_plus_predictions, df, left_on=['store_code_uc', 'brand_code_uc'], right_on=['store_code', 'brand_code'], how='left')

    # INSERT_YOUR_CODE
    output_path = os.path.join(os.path.dirname(__file__), "merged_predicted_plus_observed.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merged DataFrame saved to: {output_path}")
    
    # Filter merged_df with function filter_columns
    filtered_df = filter_columns(merged_df)
    print(f"Filtered DataFrame shape: {filtered_df.shape}")
    print(f"Filtered DataFrame columns: {list(filtered_df.columns)}")
    
    # Filter the resulting dataframe using randomly one of the values in brand_code_uc
    unique_brands = filtered_df['brand_code_uc'].unique()
    if len(unique_brands) > 0:
        random_brand = random.choice(unique_brands)
        brand_filtered_df = filtered_df[filtered_df['brand_code_uc'] == random_brand]
        print(f"Randomly selected brand: {random_brand}")
        print(f"Brand-filtered DataFrame shape: {brand_filtered_df.shape}")
        
        # Extract predicted prices (columns starting with 'price_prediction')
        price_prediction_cols = [col for col in brand_filtered_df.columns if col.startswith('price_prediction')]
        predicted_prices = []
        for col in price_prediction_cols:
            predicted_prices.extend(brand_filtered_df[col].dropna().tolist())
        predicted_prices = [float(x) for x in predicted_prices if not pd.isna(x)]
        
        # Extract observed prices (columns starting with 'product_data_completo')
        product_data_cols = [col for col in brand_filtered_df.columns if col.startswith('product_data_completo')]
        observed_prices = []
        for col in product_data_cols:
            observed_prices.extend(brand_filtered_df[col].dropna().tolist())
        observed_prices = [float(x) for x in observed_prices if not pd.isna(x)]
        
        print(f"Number of predicted prices: {len(predicted_prices)}")
        print(f"Number of observed prices: {len(observed_prices)}")
        
        # Initialize PredictionObservationComparison class
        if len(predicted_prices) > 0 and len(observed_prices) > 0:
            comparison = PredictionObservationComparison(
                prediction_data=predicted_prices,
                observation_data=observed_prices,
                prediction_name=f"Price Predictions for Brand {random_brand}",
                observation_name=f"Observed Prices for Brand {random_brand}",
                units="Price Units"
            )
            print("PredictionObservationComparison object created successfully!")
            
            # Optional: Run a basic analysis
            try:
                results = comparison.run_full_analysis(alpha=0.05, create_plots=True, figsize=(15, 10))
                print("Full analysis completed successfully!")
            except Exception as e:
                print(f"Error running full analysis: {e}")
        else:
            print("Warning: No valid predicted or observed prices found for the selected brand.")
    else:
        print("Warning: No unique brands found in the filtered DataFrame.")
    
    breakpoint()

    #Comparison of predicted vs observed