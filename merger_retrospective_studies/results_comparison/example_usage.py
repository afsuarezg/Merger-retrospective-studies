"""
Example Usage Script for Optimization Visualizer

This script demonstrates how to use the OptimizationVisualizer class
with synthetic data and real optimization results.

Author: Generated for merger retrospective studies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimization_visualizer import OptimizationVisualizer

def create_synthetic_data(n_solutions=5, n_parameters=10, seed=42):
    """
    Create synthetic optimization data for demonstration.
    
    Parameters:
    -----------
    n_solutions : int
        Number of optimization solutions
    n_parameters : int
        Number of parameters per solution
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray
        Synthetic optimization data
    """
    np.random.seed(seed)
    
    # Create synthetic data
    data = np.zeros((n_solutions, 5 + n_parameters))
    
    # Row indices
    data[:, 0] = np.arange(n_solutions)
    
    # Objective values (decreasing with some noise)
    data[:, 1] = np.exp(-np.arange(n_solutions) * 0.5) + np.random.normal(0, 0.1, n_solutions)
    
    # Gradient norms (decreasing with some noise)
    data[:, 2] = np.exp(-np.arange(n_solutions) * 0.8) + np.random.exponential(0.1, n_solutions)
    
    # Min Hessian eigenvalues (mix of positive and negative)
    data[:, 3] = np.random.normal(0.1, 0.5, n_solutions)
    data[0, 3] = 0.2  # Ensure at least one positive
    if n_solutions > 1:
        data[1, 3] = -0.1  # Ensure at least one negative
    
    # Max Hessian eigenvalues (positive, increasing)
    data[:, 4] = np.exp(np.arange(n_solutions) * 0.3) + np.random.exponential(1, n_solutions)
    
    # Parameters (different scales and patterns)
    for i in range(n_solutions):
        # Create different parameter patterns
        if i == 0:  # Solution 0: all positive
            data[i, 5:] = np.random.uniform(0.1, 2.0, n_parameters)
        elif i == 1:  # Solution 1: mixed signs
            data[i, 5:] = np.random.normal(0, 1, n_parameters)
        elif i == 2:  # Solution 2: similar to solution 0
            data[i, 5:] = data[0, 5:] + np.random.normal(0, 0.1, n_parameters)
        else:  # Other solutions: random
            data[i, 5:] = np.random.normal(0, 0.5, n_parameters)
    
    return data

def load_real_data_example():
    """
    Example of how to load real optimization data.
    
    This function shows the expected data format and how to load
    optimization results from various sources.
    """
    print("Example of loading real optimization data:")
    print("=" * 50)
    
    # Example 1: Loading from NumPy array
    print("1. From NumPy array:")
    print("   data = np.load('optimization_results.npy')")
    print("   viz = OptimizationVisualizer(data, parameter_start_col=5)")
    print()
    
    # Example 2: Loading from CSV
    print("2. From CSV file:")
    print("   df = pd.read_csv('optimization_results.csv')")
    print("   viz = OptimizationVisualizer(df, parameter_start_col=5)")
    print()
    
    # Example 3: Loading from pickle
    print("3. From pickle file:")
    print("   import pickle")
    print("   with open('optimization_results.pkl', 'rb') as f:")
    print("       data = pickle.load(f)")
    print("   viz = OptimizationVisualizer(data, parameter_start_col=5)")
    print()

def demonstrate_basic_usage():
    """Demonstrate basic usage of the OptimizationVisualizer."""
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 40)
    
    # Create synthetic data
    data = create_synthetic_data(n_solutions=6, n_parameters=8)
    print(f"Created synthetic data with {data.shape[0]} solutions and {data.shape[1]-5} parameters")
    
    # Create visualizer
    viz = OptimizationVisualizer(data, parameter_start_col=5)
    
    # Print summary
    viz.print_summary()
    
    # Generate individual plots
    print("\nGenerating individual plots...")
    viz.plot_objective_function(save_path='example_objective.png')
    viz.plot_gradient_norm(save_path='example_gradient.png')
    viz.plot_hessian_eigenvalues(save_path='example_hessian.png')
    
    # Generate distance plots
    for metric in ['euclidean', 'manhattan', 'cosine']:
        viz.plot_pairwise_distances(metric=metric, 
                                   save_path=f'example_pairwise_{metric}.png')
        viz.plot_distance_heatmap(metric=metric, 
                                 save_path=f'example_heatmap_{metric}.png')
    
    # Create dashboard
    print("Creating comprehensive dashboard...")
    viz.create_dashboard(save_path='example_dashboard.png')
    
    print("All example plots saved to current directory!")

def demonstrate_advanced_usage():
    """Demonstrate advanced usage and customization."""
    print("\nADVANCED USAGE DEMONSTRATION")
    print("=" * 40)
    
    # Create larger dataset
    data = create_synthetic_data(n_solutions=10, n_parameters=15, seed=123)
    viz = OptimizationVisualizer(data, parameter_start_col=5)
    
    # Custom figure sizes
    print("Creating plots with custom sizes...")
    viz.plot_objective_function(figsize=(12, 8), save_path='custom_objective.png')
    viz.plot_distance_heatmap(metric='euclidean', figsize=(15, 10), 
                             save_path='custom_heatmap.png')
    
    # Generate all plots to directory
    print("Generating all plots to directory...")
    viz.generate_all_plots(output_dir='./example_plots')
    
    # Get detailed statistics
    stats = viz.get_summary_statistics()
    print(f"\nDetailed Statistics:")
    print(f"  Best objective: Row {stats['best_objective'][0]} = {stats['best_objective'][1]:.6f}")
    print(f"  Local minima found: {len(stats['local_minima'])} solutions")
    print(f"  Convergence rate: {stats['converged_solutions']}/{stats['n_solutions']} solutions")

def demonstrate_error_handling():
    """Demonstrate error handling and edge cases."""
    print("\nERROR HANDLING DEMONSTRATION")
    print("=" * 40)
    
    # Test with edge cases
    try:
        # Single solution
        single_data = create_synthetic_data(n_solutions=1, n_parameters=5)
        viz_single = OptimizationVisualizer(single_data)
        print("+ Single solution handled correctly")
    except Exception as e:
        print(f"- Single solution error: {e}")
    
    try:
        # Identical solutions
        identical_data = np.ones((3, 8))
        identical_data[:, 0] = [0, 1, 2]  # Row indices
        identical_data[:, 1] = [1.0, 1.0, 1.0]  # Same objectives
        identical_data[:, 2] = [0.001, 0.001, 0.001]  # Same gradients
        identical_data[:, 3] = [0.1, 0.1, 0.1]  # Same min hessian
        identical_data[:, 4] = [10.0, 10.0, 10.0]  # Same max hessian
        # Parameters are all 1.0
        
        viz_identical = OptimizationVisualizer(identical_data)
        print("+ Identical solutions handled correctly")
        
        # Test distance calculations with identical solutions
        viz_identical.plot_distance_heatmap(metric='euclidean', 
                                          save_path='identical_heatmap.png')
        print("+ Distance calculations work with identical solutions")
        
    except Exception as e:
        print(f"- Identical solutions error: {e}")

def main():
    """Main demonstration function."""
    print("OPTIMIZATION VISUALIZER - EXAMPLE USAGE")
    print("=" * 50)
    
    # Show data loading examples
    load_real_data_example()
    
    # Basic usage
    demonstrate_basic_usage()
    
    # Advanced usage
    demonstrate_advanced_usage()
    
    # Error handling
    demonstrate_error_handling()
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE!")
    print("Check the generated PNG files for visualization examples.")
    print("=" * 50)

if __name__ == "__main__":
    main()