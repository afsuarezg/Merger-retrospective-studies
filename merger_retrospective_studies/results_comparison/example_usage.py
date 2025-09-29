"""
Example usage of the VectorComparator module.

This script demonstrates how to use the VectorComparator class
to compare optimization result vectors.
"""

import numpy as np
from vector_run_comparator import VectorComparator

def main():
    """Demonstrate VectorComparator usage with sample data."""
    
    # Create sample optimization result vectors
    np.random.seed(42)
    
    # Simulate different optimization runs with different characteristics
    vectors = [
        np.array([1.0, 2.5, 3.2, 4.1, 5.0]),  # Baseline run
        np.array([1.1, 2.4, 3.3, 4.0, 5.1]),  # Slightly different
        np.array([0.8, 2.8, 2.9, 4.3, 4.8]),  # More different
        np.array([1.2, 2.2, 3.5, 3.9, 5.2])   # Another variation
    ]
    
    labels = ["Baseline", "Run_2", "Run_3", "Run_4"]
    
    # Sample optimization metrics
    objective_values = [0.123, 0.145, 0.134, 0.156]  # Final objective values
    gradient_norms = [1.2e-6, 2.3e-5, 8.7e-7, 4.5e-4]  # Final gradient norms
    hessian_eigenvals = [
        np.array([0.1, 0.3, 0.5, 0.8, 1.2]),  # Hessian eigenvalues for each run
        np.array([0.05, 0.2, 0.4, 0.9, 1.5]),
        np.array([0.08, 0.25, 0.45, 0.85, 1.3]),
        np.array([0.02, 0.15, 0.35, 0.7, 1.8])
    ]
    
    print("Vector Comparison Example")
    print("=" * 50)
    print(f"Comparing {len(vectors)} optimization result vectors")
    print(f"Vector length: {len(vectors[0])}")
    print()
    
    # Create comparator with optimization metrics
    comparator = VectorComparator(vectors, labels, 
                                objective_values=objective_values,
                                gradient_norms=gradient_norms,
                                hessian_eigenvals=hessian_eigenvals)
    
    # 1. Statistical Summary
    print("1. Statistical Summary:")
    print("-" * 30)
    stats = comparator.statistical_summary()
    print(stats.round(3))
    print()
    
    # 2. Magnitude Comparison
    print("2. Magnitude Comparison:")
    print("-" * 30)
    mag_comp = comparator.magnitude_comparison()
    print(f"Euclidean norms: {mag_comp['norms']}")
    print(f"Rankings (highest to lowest): {mag_comp['rankings']}")
    print(f"Norm ratio (max/min): {mag_comp['norm_ratio']:.3f}")
    print()
    
    # 3. Distance Analysis
    print("3. Distance Matrix (Euclidean):")
    print("-" * 30)
    distances = comparator.distance_matrix('euclidean')
    print(distances.round(3))
    print()
    
    # 4. Similarity Analysis
    print("4. Similarity Matrix (Cosine):")
    print("-" * 30)
    similarities = comparator.similarity_matrix('cosine')
    print(similarities.round(3))
    print()
    
    # 5. Component-wise Analysis
    print("5. Component-wise Analysis (vs Baseline):")
    print("-" * 30)
    comp_analysis = comparator.component_wise_analysis(reference_idx=0)
    for label, data in comp_analysis.items():
        if label != 'reference':
            print(f"{label}:")
            print(f"  Max difference: {data['max_difference']:.3f}")
            print(f"  Mean absolute difference: {data['mean_absolute_difference']:.3f}")
            print(f"  Max ratio: {data['max_ratio']:.3f}")
    print()
    
    # 6. Optimization Metrics Summary
    print("6. Optimization Metrics Summary:")
    print("-" * 30)
    opt_metrics = comparator.optimization_metrics_summary()
    print(opt_metrics.round(6))
    print()
    
    # 7. Convergence Analysis
    print("7. Convergence Analysis:")
    print("-" * 30)
    convergence = comparator.optimization_convergence_analysis()
    print(f"Convergence rate: {convergence['convergence_assessment']['convergence_rate']:.1%}")
    print(f"Best objective: {convergence['objective_analysis']['best_objective']:.6f}")
    print(f"Best vector: {convergence['objective_analysis']['best_vector_label']}")
    print(f"Converged runs: {convergence['gradient_analysis']['converged_runs']}")
    print()
    
    # 8. Normalization Example
    print("8. Normalization Example:")
    print("-" * 30)
    normalized = comparator.normalize_vectors('l2')
    norm_stats = normalized.statistical_summary()
    print("After L2 normalization:")
    print(norm_stats[['Vector', 'Mean', 'Std']].round(3))
    print()
    
    # 9. Generate Comprehensive Report
    print("9. Generating comprehensive report...")
    report = comparator.generate_report(include_plots=False)
    print(f"Report generated with {len(report)} sections")
    print(f"Summary: {report['summary']}")
    
    # 10. Visualization (commented out to avoid display issues in script)
    print("\n10. Visualization methods available:")
    print("   - comparator.plot_vectors()")
    print("   - comparator.plot_distance_heatmap()")
    print("   - comparator.plot_component_analysis()")
    print("   - comparator.plot_optimization_metrics()")
    print("   - comparator.plot_comprehensive_overview()")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
