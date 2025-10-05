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
        raise FileNotFoundError(f"Folder not found: {folder}")

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




if __name__ == "__main__":
    # Create a list with the full paths to the files in ProblemResults_sample
    results = read_pickles_from_folder("ProblemResults_sample")
    print(results)


    