# Vector Comparator 2.0

A comprehensive Python module for comparing one-dimensional vectors resulting from different optimization algorithm runs, with extensive analysis capabilities and optimization metrics support.

## Features

### Core Functionality
- **Magnitude Comparison**: Euclidean norms, rankings, and norm ratios
- **Component-wise Analysis**: Element-wise differences, ratios, and significant position identification
- **Statistical Summaries**: Comprehensive statistics including optimization metrics
- **Distance Metrics**: Euclidean, Manhattan, and Chebyshev distances
- **Similarity Metrics**: Cosine similarity and Pearson correlation
- **Normalization Options**: L2 normalization, standardization, and min-max scaling
- **Visualization**: Multiple plot types for comprehensive analysis
- **Report Generation**: Detailed comparison reports with export capabilities

### Optimization Metrics Support
- Objective function values
- Projected gradient norms
- Hessian minimum and maximum eigenvalues
- Convergence analysis
- Condition number analysis

## Installation

The module requires the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Quick Start

```python
from vector_run_comparator_2 import VectorComparator
import numpy as np

# Create sample vectors
vectors = [
    np.array([1.0, 2.5, 3.2, 4.1, 5.0]),
    np.array([1.1, 2.4, 3.3, 4.0, 5.1]),
    np.array([0.8, 2.8, 2.9, 4.3, 4.8])
]

# Optional optimization metrics
objective_values = [0.123, 0.145, 0.134]
gradient_norms = [1.2e-6, 2.3e-5, 8.7e-7]
hessian_min_eigenvalues = [-0.1, 0.05, -0.05]
hessian_max_eigenvalues = [10.2, 9.8, 11.3]

# Create comparator
comparator = VectorComparator(
    vectors,
    labels=["Run 1", "Run 2", "Run 3"],
    objective_values=objective_values,
    gradient_norms=gradient_norms,
    hessian_min_eigenvalues=hessian_min_eigenvalues,
    hessian_max_eigenvalues=hessian_max_eigenvalues
)

# Get statistical summary
stats = comparator.statistical_summary()
print(stats)

# Generate comprehensive report
report = comparator.generate_report()
```

## API Reference

### VectorComparator Class

#### Constructor
```python
VectorComparator(
    vectors: List[Union[np.ndarray, List[float]]],
    labels: Optional[List[str]] = None,
    objective_values: Optional[List[float]] = None,
    gradient_norms: Optional[List[float]] = None,
    hessian_min_eigenvalues: Optional[List[float]] = None,
    hessian_max_eigenvalues: Optional[List[float]] = None,
    tolerance: float = 1e-6
)
```

#### Core Methods

##### Statistical Analysis
- `statistical_summary()`: Comprehensive statistical summary including optimization metrics
- `magnitude_comparison()`: Euclidean norms, rankings, and norm ratios
- `component_wise_analysis(reference_idx=0)`: Element-wise analysis against reference vector

##### Distance and Similarity
- `distance_matrix(metric='euclidean')`: Distance matrix between all vector pairs
- `similarity_matrix(metric='cosine')`: Similarity matrix between all vector pairs

##### Normalization
- `normalize_vectors(method='l2')`: Create new comparator with normalized vectors
  - Methods: 'l2', 'standardize', 'minmax'

##### Visualization
- `plot_vectors(figsize=(12, 8))`: Line plot comparing all vectors
- `plot_distance_heatmap(metric='euclidean')`: Heatmap of distance matrix
- `plot_optimization_metrics(figsize=(15, 10))`: Bar charts of optimization metrics
- `plot_component_analysis(reference_idx=0)`: Component-wise analysis plots

##### Reporting
- `generate_report(include_plots=False, output_file=None)`: Comprehensive comparison report
- `save_results(filename, format='json')`: Save results to file (JSON or CSV)

### Convenience Functions

- `quick_compare(vectors, labels=None, **kwargs)`: Quick comparison with minimal setup
- `load_comparison_results(filename)`: Load previously saved results

## Examples

### Basic Vector Comparison
```python
from vector_run_comparator_2 import VectorComparator
import numpy as np

vectors = [np.random.randn(10) for _ in range(5)]
comparator = VectorComparator(vectors, labels=[f"Run_{i}" for i in range(5)])

# Statistical analysis
stats = comparator.statistical_summary()
print(stats)

# Distance analysis
distances = comparator.distance_matrix('euclidean')
print(distances)
```

### With Optimization Metrics
```python
# Include optimization metrics
comparator = VectorComparator(
    vectors,
    labels=["Run_1", "Run_2", "Run_3"],
    objective_values=[0.123, 0.145, 0.134],
    gradient_norms=[1.2e-6, 2.3e-5, 8.7e-7],
    hessian_min_eigenvalues=[-0.1, 0.05, -0.05],
    hessian_max_eigenvalues=[10.2, 9.8, 11.3]
)

# Analyze optimization performance
opt_analysis = comparator._analyze_optimization_metrics()
print(f"Best objective: {opt_analysis['objective_analysis']['best_objective']}")
print(f"Converged runs: {opt_analysis['gradient_analysis']['converged_runs']}")
```

### Visualization
```python
# Generate plots
comparator.plot_vectors()
comparator.plot_distance_heatmap()
comparator.plot_optimization_metrics()
comparator.plot_component_analysis()
```

### Normalization
```python
# L2 normalization
normalized = comparator.normalize_vectors('l2')

# Standardization
standardized = comparator.normalize_vectors('standardize')

# Min-max scaling
minmax_scaled = comparator.normalize_vectors('minmax')
```

### Report Generation
```python
# Generate comprehensive report
report = comparator.generate_report(include_plots=False)

# Save results
comparator.save_results('results.json', format='json')
comparator.save_results('stats.csv', format='csv')
```

## Error Handling

The module includes comprehensive error handling for:
- Invalid vector dimensions (must be 1D)
- Mismatched vector lengths
- Invalid optimization metrics dimensions
- Unsupported distance/similarity metrics
- File I/O errors

## Performance Considerations

- Uses NumPy for efficient numerical computations
- Leverages SciPy for distance calculations
- Optimized for typical optimization result vectors (length < 1000)
- Memory efficient for reasonable numbers of vectors (< 100)

## Dependencies

- NumPy >= 1.19.0
- Pandas >= 1.3.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
- SciPy >= 1.7.0

## License

This module is part of the merger retrospective studies project.
