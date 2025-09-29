"""
Test script for VectorComparator 2.0 module.

This script tests various functionality of the VectorComparator class
to ensure it works correctly with different scenarios.
"""

import numpy as np
import pytest
from vector_run_comparator_2 import VectorComparator, quick_compare

def test_basic_functionality():
    """Test basic vector comparison functionality."""
    # Create test vectors
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([0.9, 1.9, 2.9])
    ]
    
    labels = ["Vector_1", "Vector_2", "Vector_3"]
    
    # Test basic initialization
    comparator = VectorComparator(vectors, labels)
    
    # Test statistical summary
    stats = comparator.statistical_summary()
    assert len(stats) == 3
    assert all(col in stats.columns for col in ['Mean', 'Std', 'Euclidean_Norm'])
    
    # Test magnitude comparison
    mag_comp = comparator.magnitude_comparison()
    assert len(mag_comp['norms']) == 3
    assert len(mag_comp['rankings']) == 3
    
    # Test distance matrix
    distances = comparator.distance_matrix('euclidean')
    assert distances.shape == (3, 3)
    assert np.allclose(distances.values, distances.values.T)  # Symmetric
    
    # Test similarity matrix
    similarities = comparator.similarity_matrix('cosine')
    assert similarities.shape == (3, 3)
    assert np.allclose(np.diag(similarities), 1.0)  # Diagonal should be 1
    
    print("+ Basic functionality test passed")

def test_optimization_metrics():
    """Test functionality with optimization metrics."""
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([0.9, 1.9, 2.9])
    ]
    
    labels = ["Run_1", "Run_2", "Run_3"]
    objective_values = [0.123, 0.145, 0.134]
    gradient_norms = [1e-6, 2e-5, 5e-7]
    hessian_min_eigenvalues = [-0.1, 0.05, -0.05]
    hessian_max_eigenvalues = [10.0, 9.5, 11.0]
    
    comparator = VectorComparator(
        vectors, labels,
        objective_values=objective_values,
        gradient_norms=gradient_norms,
        hessian_min_eigenvalues=hessian_min_eigenvalues,
        hessian_max_eigenvalues=hessian_max_eigenvalues
    )
    
    # Test statistical summary includes optimization metrics
    stats = comparator.statistical_summary()
    assert 'Objective_Value' in stats.columns
    assert 'Gradient_Norm' in stats.columns
    assert 'Hessian_Min_Eigenvalue' in stats.columns
    assert 'Hessian_Max_Eigenvalue' in stats.columns
    
    # Test optimization analysis
    opt_analysis = comparator._analyze_optimization_metrics()
    assert 'objective_analysis' in opt_analysis
    assert 'gradient_analysis' in opt_analysis
    assert 'hessian_analysis' in opt_analysis
    
    print("+ Optimization metrics test passed")

def test_normalization():
    """Test vector normalization functionality."""
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        np.array([0.5, 1.0, 1.5])
    ]
    
    comparator = VectorComparator(vectors, ["V1", "V2", "V3"])
    
    # Test L2 normalization
    l2_normalized = comparator.normalize_vectors('l2')
    l2_stats = l2_normalized.statistical_summary()
    assert np.allclose(l2_stats['Euclidean_Norm'], 1.0)
    
    # Test standardization
    standardized = comparator.normalize_vectors('standardize')
    std_stats = standardized.statistical_summary()
    assert np.allclose(std_stats['Mean'], 0.0, atol=1e-10)
    assert np.allclose(std_stats['Std'], 1.0, atol=1e-10)
    
    # Test min-max scaling
    minmax_scaled = comparator.normalize_vectors('minmax')
    minmax_stats = minmax_scaled.statistical_summary()
    assert np.allclose(minmax_stats['Min'], 0.0, atol=1e-10)
    assert np.allclose(minmax_stats['Max'], 1.0, atol=1e-10)
    
    print("+ Normalization test passed")

def test_component_analysis():
    """Test component-wise analysis functionality."""
    vectors = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.1, 2.1, 3.1, 4.1]),
        np.array([0.8, 1.8, 2.8, 3.8])
    ]
    
    comparator = VectorComparator(vectors, ["Ref", "Close", "Far"])
    
    # Test component analysis
    comp_analysis = comparator.component_wise_analysis(reference_idx=0)
    
    assert 'reference' in comp_analysis
    assert 'Close' in comp_analysis
    assert 'Far' in comp_analysis
    
    # Check that differences are calculated correctly
    close_diff = comp_analysis['Close']['differences']
    expected_diff = np.array([0.1, 0.1, 0.1, 0.1])
    assert np.allclose(close_diff, expected_diff)
    
    print("+ Component analysis test passed")

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test empty vectors
    try:
        VectorComparator([])
        assert False, "Should have raised ValueError for empty vectors"
    except ValueError:
        pass
    
    # Test mismatched lengths
    try:
        VectorComparator([np.array([1, 2]), np.array([1, 2, 3])])
        assert False, "Should have raised ValueError for mismatched lengths"
    except ValueError:
        pass
    
    # Test invalid dimensions
    try:
        VectorComparator([np.array([[1, 2], [3, 4]])])
        assert False, "Should have raised ValueError for 2D vector"
    except ValueError:
        pass
    
    # Test mismatched optimization metrics
    try:
        VectorComparator(
            [np.array([1, 2]), np.array([3, 4])],
            objective_values=[1.0]  # Wrong length
        )
        assert False, "Should have raised ValueError for mismatched metrics"
    except ValueError:
        pass
    
    print("+ Error handling test passed")

def test_quick_compare():
    """Test quick comparison functionality."""
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1])
    ]
    
    quick_comp = quick_compare(vectors, ["A", "B"])
    stats = quick_comp.statistical_summary()
    
    assert len(stats) == 2
    assert list(stats['Vector']) == ["A", "B"]
    
    print("+ Quick compare test passed")

def test_report_generation():
    """Test report generation functionality."""
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1])
    ]
    
    comparator = VectorComparator(vectors, ["A", "B"])
    
    # Test report generation
    report = comparator.generate_report(include_plots=False)
    
    assert 'summary' in report
    assert 'statistical_summary' in report
    assert 'magnitude_comparison' in report
    assert 'distance_analysis' in report
    assert 'similarity_analysis' in report
    assert 'component_analysis' in report
    
    print("+ Report generation test passed")

def run_all_tests():
    """Run all tests."""
    print("Running VectorComparator 2.0 tests...")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_optimization_metrics()
        test_normalization()
        test_component_analysis()
        test_error_handling()
        test_quick_compare()
        test_report_generation()
        
        print("=" * 50)
        print("+ All tests passed successfully!")
        
    except Exception as e:
        print(f"- Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()
