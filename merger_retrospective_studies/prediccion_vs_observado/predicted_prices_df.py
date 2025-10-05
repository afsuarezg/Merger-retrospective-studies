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



if __name__ == "__main__":
    # Known folder with results
    folder = r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\2021 - 2025\Research\Merger retrospective studies\Codigo\repos\merger_retrospective_studies\data_samples\ProblemResults_sample"

    if not os.path.isdir(folder):
        print(f"ProblemResults_sample folder not found: {folder}")
        sys.exit(1)

    results = read_pickles_from_folder(folder)
    print(results)

    breakpoint()
    # Load compiled product data CSV
    compiled_df = load_compiled_product_data()
    if not compiled_df.empty:
        print("Compiled product data loaded:", compiled_df.shape)
        print(compiled_df.head(3))
    breakpoint()
    
    compiled_df['price_prediction'] = results[0][1].compute_prices(firm_ids=compiled_df['firm_ids_post_merger'], costs=results[0][1].compute_costs())
    print(compiled_df.head(3))
    breakpoint()


    