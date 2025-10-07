import os
import sys
import json
import pandas as pd

from .predicted_prices_df import read_pickles_from_folder, load_compiled_product_data, add_predictions_from_problem_results, map_store_to_brands



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
    breakpoint()
    # INSERT_YOUR_CODE
    output_path = os.path.join(os.path.dirname(__file__), "merged_predicted_plus_observed.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merged DataFrame saved to: {output_path}")
    breakpoint()
