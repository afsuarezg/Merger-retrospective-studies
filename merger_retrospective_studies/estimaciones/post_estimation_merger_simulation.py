import pyblp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def elasticities(results: pyblp.ProblemResults):
    """
    Compute and plot the elasticities from the given BLP problem results.
    This function calculates the mean own elasticities and aggregate elasticities
    from the provided BLP problem results. It then plots these elasticities as a histogram.
    Parameters:
    results (pyblp.ProblemResults): The results from a BLP problem, containing the necessary data
                                    to compute elasticities.
    Returns:
    matplotlib.figure.Figure: A matplotlib figure object containing the histogram of the elasticities.
    Notes:
    - The histogram will display two sets of data: mean own elasticities (in red) and aggregate elasticities (in blue).
    - The aggregate elasticities are computed with a factor of 0.1.
    """

    elasticities = results.compute_elasticities()
    means = results.extract_diagonal_means(elasticities)
    aggregates = results.compute_aggregate_elasticities(factor=0.1)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the histogram
    ax.hist(
        [means.flatten(), aggregates.flatten()], 
        color=['red', 'blue'], 
        bins=50
    )
    ax.legend(['Mean Own Elasticities', 'Aggregate Elasticities'])

    # Return the figure
    return fig


def premerger_costs_fig(results: pyblp.ProblemResults):
    """
    Creates a histogram of marginal costs and returns the corresponding figure.

    Parameters:
        results (pyblp.ProblemResults): The results object from which costs will be computed.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    # Compute costs
    costs = results.compute_costs()

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the histogram
    ax.hist(costs, bins=50)
    ax.legend(["Marginal Costs"])
    ax.set_title("Distribution of Marginal Costs")
    ax.set_xlabel("Costs")
    ax.set_ylabel("Frequency")

    # Return the figure
    return fig


def premerger_markups_fig(results: pyblp.ProblemResults):
    """
    Creates a histogram of markups and returns the corresponding figure.

    Parameters:
        results (pyblp.ProblemResults): The results object used to compute markups.
        costs (array-like): The marginal costs used to compute markups.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    # Compute costs
    costs = results.compute_costs()
    
    # Compute markups
    markups = results.compute_markups(costs=costs)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the histogram
    ax.hist(markups, bins=50)
    ax.legend(["Markups"])
    ax.set_title("Distribution of Markups")
    ax.set_xlabel("Markups")
    ax.set_ylabel("Frequency")

    # Return the figure
    return fig


def merger_impact_hhi(results: pyblp.ProblemResults, product_data: pd.DataFrame):
    """
    Computes and visualizes the impact of a merger on the Herfindahl-Hirschman Index (HHI).

    Parameters:
        results (pyblp.ProblemResults): The results object containing model data.
        product_data (pd.DataFrame): DataFrame with product data including firm IDs.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot of HHI changes.
    """
    # Compute costs, HHI, and profits
    costs = results.compute_costs()
    hhi = results.compute_hhi()
    profits = results.compute_profits(costs=costs)

    # Update firm IDs to simulate a merger
    product_data['merger_ids'] = product_data['firm_ids'].replace(2, 1)

    # Compute changed prices, shares, and HHI
    changed_prices = results.compute_prices(firm_ids=product_data['merger_ids'], costs=costs)
    changed_shares = results.compute_shares(changed_prices)
    changed_hhi = results.compute_hhi(firm_ids=product_data['merger_ids'], shares=changed_shares)

    # Calculate HHI changes
    hhi_changes = changed_hhi - hhi

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the histogram of HHI changes
    ax.hist(hhi_changes, bins=50)
    ax.legend(["HHI Changes"])
    ax.set_title("Impact of Merger on HHI")
    ax.set_xlabel("Change in HHI")
    ax.set_ylabel("Frequency")

    # Return the figure
    return fig


def merger_impact_markups(results: pyblp.ProblemResults, product_data: pd.DataFrame):
    """
    Computes and visualizes the impact of a merger on markups.

    Parameters:
        results (pyblp.ProblemResults): The results object containing model data.
        product_data (pd.DataFrame): DataFrame with product data including firm IDs.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot of markup changes.
    """
    # Compute costs and initial markups
    costs = results.compute_costs()
    markups = results.compute_markups(costs=costs)

    # Compute changed prices and markups after the merger
    changed_prices = results.compute_prices(firm_ids=product_data['merger_ids'], costs=costs)
    changed_markups = results.compute_markups(prices=changed_prices, costs=costs)

    # Calculate markup changes
    markup_changes = changed_markups - markups

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the histogram of markup changes
    ax.hist(markup_changes, bins=50)
    ax.legend(["Markup Changes"])
    ax.set_title("Impact of Merger on Markups")
    ax.set_xlabel("Change in Markups")
    ax.set_ylabel("Frequency")

    # Return the figure
    return fig


def predict_prices(product_data: pd.DataFrame, results, merger: List[int]) -> pd.Series:
    """
    Predict prices after a merger between firms.

    Parameters:
    - product_data (pd.DataFrame): A dataframe containing product and firm information.
                                   Must include a 'firm_ids' column.
    - results: An object with methods to compute costs, markups, and prices.
    - merger (List[int]): A list of two integers representing the merging firms' IDs.
                          merger[1] will be replaced with merger[0].

    Returns:
    - pd.Series: A series of changed prices after the merger.
    """

    
    # Update firm IDs to reflect the merger
    product_data['merger_ids'] = product_data['firm_ids'].replace({merger[1]: merger[0]})
    
    # Compute costs and initial markups
    costs = results.compute_costs()
    
    # Compute changed prices after the merger
    changed_prices = results.compute_prices(firm_ids=product_data['merger_ids'], costs=costs)
    
    return changed_prices



def predicted_prices(product_data: pd.DataFrame, results: pyblp.ProblemResults) -> pd.Series:
    
    # Compute costs and initial markups
    costs = results.compute_costs()
    
    # Compute changed prices after the merger
    predicted_prices = results.compute_prices(firm_ids=product_data['firm_ids_post_merger'], costs=costs)
    
    return predicted_prices


