import pandas as pd 
import os
import json
import re
import matplotlib.pyplot as plt



def load_price_predictions(data, directory):
    pattern = 'price_predictions_'
    iteration = 1
    files = sorted(os.listdir(directory), key=lambda x: int(re.search(r'_(\d+)\.json', x).group(1)))
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            this_data = json.load(f)
        data[f'price_prediction_{iteration}'] = this_data['price_prediction']
        iteration += 1


def calculate_stats(row):
    stats = {
        "mean": row.mean(),
        "median": row.median(),
        "std": row.std(),
        "min": row.min(),
        "max": row.max(),
    }
    return pd.Series(stats)


def add_descriptive_stats(data):
    prediction_columns = [col for col in data.columns if col.startswith("price_prediction")]
    stats = data[prediction_columns].apply(calculate_stats, axis=1)
    stats.columns = [f"price_prediction_{stat}" for stat in stats.columns]
    data_descriptive = pd.concat([data, stats], axis=1)
    return data_descriptive


def plot_histogram(data_descriptive, prediction_columns):
    data_descriptive.iloc[0][prediction_columns].plot.hist(bins=20, title="Histogram of Series")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == '__main__':
    pass