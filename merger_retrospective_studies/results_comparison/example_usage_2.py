"""
Example usage of the VectorComparator module (version 2).

This script demonstrates how to use the VectorComparator class
to compare optimization result vectors with comprehensive analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from merger_retrospective_studies.results_comparison.vector_run_comparator_2 import VectorComparator, quick_compare

def main():
    """Demonstrate VectorComparator usage with sample data."""
    
    # Create sample optimization result vectors
    np.random.seed(42)
    
    # Simulate different optimization runs with different characteristics
    vectors = [
        np.array([1.0, 2.5, 3.2, 4.1, 5.0, 2.8, 1.5]),  # Baseline run
        np.array([1.1, 2.4, 3.3, 4.0, 5.1, 2.7, 1.6]),  # Slightly different
        np.array([0.8, 2.8, 2.9, 4.3, 4.8, 3.1, 1.2]),  # More different
        np.array([1.2, 2.2, 3.5, 3.9, 5.2, 2.6, 1.8])   # Another variation
    ]
    
    labels = ["Baseline", "Run_2", "Run_3", "Run_4"]
    
    # Sample optimization metrics
    objective_values = [0.123, 0.145, 0.134, 0.156]  # Final objective values
    gradient_norms = [1.2e-6, 2.3e-5, 8.7e-7, 4.5e-4]  # Final gradient norms
    hessian_min_eigenvalues = [-0.1, 0.05, -0.05, 0.02]  # Min eigenvalues
    hessian_max_eigenvalues = [10.2, 9.8, 11.3, 12.1]  # Max eigenvalues
    
    print("Vector Comparison Example (Version 2)")
    print("=" * 50)
    print(f"Comparing {len(vectors)} optimization result vectors")
    print(f"Vector length: {len(vectors[0])}")
    print()
    
    # Create comparator with optimization metrics
    comparator = VectorComparator(
        vectors, 
        labels, 
        objective_values=objective_values,
        gradient_norms=gradient_norms,
        hessian_min_eigenvalues=hessian_min_eigenvalues,
        hessian_max_eigenvalues=hessian_max_eigenvalues
    )
    
    # 1. Statistical Summary
    print("1. Statistical Summary:")
    print("-" * 30)
    stats = comparator.statistical_summary()
    print(stats.round(6))
    print()
    
    # 2. Magnitude Comparison
    print("2. Magnitude Comparison:")
    print("-" * 30)
    mag_comp = comparator.magnitude_comparison()
    print(f"Euclidean norms: {[f'{n:.3f}' for n in mag_comp['norms']]}")
    print(f"Rankings (highest to lowest): {[labels[i] for i in mag_comp['rankings']]}")
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
            print(f"  Significant positions: {data['significant_positions']}")
    print()
    
    # 6. Optimization Metrics Analysis
    print("6. Optimization Metrics Analysis:")
    print("-" * 30)
    opt_analysis = comparator._analyze_optimization_metrics()
    
    if 'objective_analysis' in opt_analysis:
        obj_analysis = opt_analysis['objective_analysis']
        print(f"Best objective: {obj_analysis['best_objective']:.6f} ({obj_analysis['best_vector_label']})")
        print(f"Objective range: {obj_analysis['objective_range']:.6f}")
    
    if 'gradient_analysis' in opt_analysis:
        grad_analysis = opt_analysis['gradient_analysis']
        print(f"Converged runs: {grad_analysis['converged_runs']}/{len(vectors)} ({grad_analysis['convergence_rate']:.1%})")
        print(f"Best gradient norm: {grad_analysis['min_gradient_norm']:.2e} ({grad_analysis['best_gradient_label']})")
    
    if 'hessian_analysis' in opt_analysis:
        hess_analysis = opt_analysis['hessian_analysis']
        print(f"Ill-conditioned runs: {hess_analysis['ill_conditioned_runs']}")
        print(f"Best conditioned: {hess_analysis['best_conditioned_label']}")
    print()
    
    # 7. Normalization Example
    print("7. Normalization Example:")
    print("-" * 30)
    normalized = comparator.normalize_vectors('l2')
    norm_stats = normalized.statistical_summary()
    print("After L2 normalization:")
    print(norm_stats[['Vector', 'Mean', 'Std', 'Euclidean_Norm']].round(6))
    print()
    
    # 8. Generate Comprehensive Report
    print("8. Generating comprehensive report...")
    report = comparator.generate_report(include_plots=False)
    print(f"Report generated with {len(report)} sections")
    print(f"Summary: {report['summary']}")
    
    # 9. Save results
    print("\n9. Saving results...")
    comparator.save_results('comparison_results.json', format='json')
    comparator.save_results('statistical_summary.csv', format='csv')
    print("Results saved to 'comparison_results.json' and 'statistical_summary.csv'")
    
    # 10. Visualization examples
    print("\n10. Visualization methods available:")
    print("   - comparator.plot_vectors()")
    print("   - comparator.plot_distance_heatmap()")
    print("   - comparator.plot_component_analysis()")
    print("   - comparator.plot_optimization_metrics()")
    
    # Demonstrate quick comparison
    print("\n11. Quick comparison example:")
    print("-" * 30)
    quick_comp = quick_compare(vectors[:2], labels[:2])
    quick_stats = quick_comp.statistical_summary()
    print("Quick comparison of first two vectors:")
    print(quick_stats[['Vector', 'Mean', 'Std', 'Euclidean_Norm']].round(3))
    
    print("\nExample completed successfully!")
    
    # Uncomment the following lines to generate plots
    print("\nGenerating plots...")
    comparator.plot_vectors().show()
    # breakpoint()
    # plt.savefig('vector_comparison.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    comparator.plot_distance_heatmap().show()
    # plt.savefig('distance_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # breakpoint()

    comparator.plot_optimization_metrics().show()
    # plt.savefig('optimization_metrics.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # breakpoint()

    # print("Plots saved as PNG files")

if __name__ == "__main__":
    main()
