import os 
import pandas as pd
import pyblp
import json
import pickle
from typing import Any
import sys


def read_pickles_from_folder(folder: str):
    """Read all .pickle/.pkl files from a folder and return a list of (path, object).

    Files that cannot be unpickled are skipped with a console message.
    """
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return []

    results = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in [".pickle", ".pkl"]:
            continue
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            results.append((path, obj))
        except Exception as exc:
            print(f"Skipping unreadable pickle: {path} ({exc})")
            continue

    return results

def load_compiled_product_data(
    file_path: str = r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\Research\Merger retrospective studies\Codigo\repos\merger_retrospective_studies\data_samples\product_data_sample\compiled_data_Reynolds_Lorillard_2025-09-29 21_16_09.173794.csv",
):
    """Load the compiled product data CSV and return a DataFrame.

    Returns an empty DataFrame if the file is not found or can't be read.
    """
    if not os.path.isfile(file_path):
        print(f"Compiled product data not found: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as exc:
        print(f"Error reading compiled product data: {exc}")
        return pd.DataFrame()


def add_predictions_from_problem_results(
    compiled_df: pd.DataFrame,
    results: list,
    firm_ids_column: str = "firm_ids_post_merger",
):
    """Attach predicted prices per problem result as separate columns on compiled_df.

    Each new column is named using the corresponding ProblemResults_sample filename.
    """
    if compiled_df is None or compiled_df.empty:
        print("compiled_df is empty; cannot add predictions.")
        return compiled_df

    if firm_ids_column not in compiled_df.columns:
        print(f"Missing required column '{firm_ids_column}' in compiled_df.")
        return compiled_df

    for path, obj in results:
        try:
            base_name = os.path.splitext(os.path.basename(path))[0]
            column_name = f"price_prediction__{base_name}"

            costs = obj.compute_costs()
            prices = obj.compute_prices(
                firm_ids=compiled_df[firm_ids_column],
                costs=costs,
            )

            compiled_df[column_name] = prices
            print(f"Added predictions column: {column_name}")
        except Exception as exc:
            print(f"Failed to compute predictions for {path}: {exc}")

    return compiled_df


def map_store_to_brands(compiled_df: pd.DataFrame):
    """Return a dict mapping each store_code_uc to its unique brand_code_uc values.

    If required columns are missing or the DataFrame is empty, returns an empty dict.
    """
    if compiled_df is None or compiled_df.empty:
        print("compiled_df is empty; cannot build store->brands map.")
        return {}

    required = {"store_code_uc", "brand_code_uc"}
    if not required.issubset(set(compiled_df.columns)):
        print("compiled_df missing required columns 'store_code_uc' and/or 'brand_code_uc'.")
        return {}

    grouped = compiled_df.groupby("store_code_uc")["brand_code_uc"].unique()
    # Convert numpy arrays to Python lists to ensure JSON-serializable values
    return {store: list(brands) for store, brands in grouped.to_dict().items()}


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
    breakpoint()
    
    base_data_plus_predictions = add_predictions_from_problem_results(base_data, simulation_results)

    # print(base_data_plus_predictions.head(5))
    breakpoint()

    if not base_data_plus_predictions.empty:
        # Example usage: build mapping from store to available brands
        store_to_brands = map_store_to_brands(base_data_plus_predictions)

    breakpoint()
    # INSERT_YOUR_CODE

    output_path = os.path.join(folder, "store_to_brands.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(store_to_brands, f, ensure_ascii=False, indent=2)
    print(f"store_to_brands mapping saved to {output_path}")
    breakpoint()
    

