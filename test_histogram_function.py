#!/usr/bin/env python3
"""
Test script to demonstrate the new histogram functionality for comparing 
one prediction with multiple observed values.
"""

import numpy as np
import matplotlib.pyplot as plt
from merger_retrospective_studies.prediccion_vs_observado.prediction_observation_comparison import PredictionObservationComparison

def test_histogram_functionality():
    """Test the new histogram functionality with example data."""
    
    print("Testing Histogram Functionality for Prediction vs Observations")
    print("=" * 60)
    
    # Create example data
    np.random.seed(42)
    
    # Example 1: Prediction within the distribution
    print("\n1. Example: Prediction within observed distribution")
    print("-" * 50)
    
    observed_values_1 = np.random.normal(100, 15, 50)  # Mean=100, Std=15
    prediction_1 = 105.5  # Prediction close to mean
    
    # Create comparison object
    comparison_1 = PredictionObservationComparison(
        prediction_data=prediction_1,
        observation_data=observed_values_1,
        prediction_name="Model Prediction",
        observation_name="Actual Values",
        units="USD"
    )
    
    # Test matplotlib histogram
    print("Creating matplotlib histogram...")
    fig1, ax1 = comparison_1.plot_histogram_with_prediction(
        observations=observed_values_1,
        prediction=prediction_1,
        bins='auto',
        density=False,
        include_stats=True
    )
    plt.savefig('histogram_example_1_matplotlib.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Matplotlib histogram saved as 'histogram_example_1_matplotlib.png'")
    
    # Test seaborn histogram
    print("Creating seaborn histogram...")
    fig2, ax2 = comparison_1.plot_histogram_with_prediction_seaborn(
        observations=observed_values_1,
        prediction=prediction_1,
        bins=20,
        density=True,
        include_stats=True
    )
    plt.savefig('histogram_example_1_seaborn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Seaborn histogram saved as 'histogram_example_1_seaborn.png'")
    
    # Test plotly histogram (if available)
    try:
        print("Creating plotly histogram...")
        fig3 = comparison_1.plot_histogram_with_prediction_plotly(
            observations=observed_values_1,
            prediction=prediction_1,
            bins=20,
            density=False,
            include_stats=True
        )
        fig3.write_html('histogram_example_1_plotly.html')
        print("✓ Plotly histogram saved as 'histogram_example_1_plotly.html'")
    except ImportError:
        print("⚠ Plotly not available, skipping interactive histogram")
    
    # Example 2: Prediction outside the distribution
    print("\n2. Example: Prediction outside observed distribution")
    print("-" * 50)
    
    observed_values_2 = np.random.normal(50, 8, 30)  # Mean=50, Std=8
    prediction_2 = 75.0  # Prediction well above the distribution
    
    comparison_2 = PredictionObservationComparison(
        prediction_data=prediction_2,
        observation_data=observed_values_2,
        prediction_name="Outlier Prediction",
        observation_name="Market Data",
        units="%"
    )
    
    # Create histogram for outlier case
    fig4, ax4 = comparison_2.plot_histogram_with_prediction(
        observations=observed_values_2,
        prediction=prediction_2,
        bins=15,
        density=False,
        include_stats=True,
        figsize=(12, 8)
    )
    plt.savefig('histogram_example_2_outlier.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Outlier histogram saved as 'histogram_example_2_outlier.png'")
    
    # Example 3: Test the visualize_single_prediction method
    print("\n3. Testing visualize_single_prediction method")
    print("-" * 50)
    
    # This should include histogram in the 'all' plot type
    figures = comparison_1.visualize_single_prediction(plot_type='all', backend='matplotlib')
    
    if 'histogram' in figures:
        print("✓ Histogram successfully included in visualize_single_prediction method")
        # Save the histogram from the method
        fig_hist, ax_hist = figures['histogram']
        plt.figure(fig_hist.number)
        plt.savefig('histogram_from_method.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Histogram from method saved as 'histogram_from_method.png'")
    else:
        print("✗ Histogram not found in visualize_single_prediction method")
    
    # Print summary statistics
    print("\n4. Summary Statistics")
    print("-" * 20)
    
    # Calculate some basic stats for the first example
    obs_mean = np.mean(observed_values_1)
    obs_std = np.std(observed_values_1)
    percentile_rank = (np.sum(observed_values_1 <= prediction_1) / len(observed_values_1)) * 100
    within_range = np.min(observed_values_1) <= prediction_1 <= np.max(observed_values_1)
    
    print(f"Observed values: Mean={obs_mean:.2f}, Std={obs_std:.2f}")
    print(f"Prediction: {prediction_1:.2f}")
    print(f"Percentile rank: {percentile_rank:.1f}th percentile")
    print(f"Within observed range: {'Yes' if within_range else 'No'}")
    
    print("\n" + "=" * 60)
    print("Histogram functionality test completed successfully!")
    print("Check the generated image files to see the visualizations.")

if __name__ == "__main__":
    test_histogram_functionality()
