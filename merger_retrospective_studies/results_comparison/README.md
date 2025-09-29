# Vector Run Comparator

A comprehensive Python module for comparing one-dimensional vectors resulting from different optimization algorithm runs.

## Overview

The `VectorComparator` class provides extensive tools for analyzing, comparing, and visualizing multiple 1D vectors from optimization experiments. It's particularly useful for comparing results from different optimization runs, algorithm variants, or parameter settings.

## Features

### Core Functionality
- **Magnitude Comparison**: Calculate and compare Euclidean norms (L2 norms) of all vectors
- **Component-wise Analysis**: Element-wise differences and ratios between vector pairs
- **Statistical Summaries**: Comprehensive statistics including mean, median, std, skewness, kurtosis
- **Distance Metrics**: Euclidean and Manhattan distance matrices
- **Similarity Metrics**: Cosine similarity and Pearson correlation matrices
- **Normalization Options**: L2 normalization, standardization, and min-max scaling
- **Visualization**: Line plots, bar charts, heatmaps, and component analysis plots
- **Report Generation**: Comprehensive comparison reports with export capabilities

## Installation

The module requires the following dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- scipy

Install with pip:
```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Quick Start

```python
from vector_run_comparator import VectorComparator
import numpy as np

# Create sample vectors
vectors = [
    np.array([1.0, 2.5, 3.2, 4.1, 5.0]),
    np.array([1.1, 2.4, 3.3, 4.0, 5.1]),
    np.array([0.8, 2.8, 2.9, 4.3, 4.8])
]
labels = ["Run 1", "Run 2", "Run 3"]

# Create comparator
comparator = VectorComparator(vectors, labels)

# Get statistical summary
stats = comparator.statistical_summary()
print(stats)

# Calculate distances
distances = comparator.distance_matrix('euclidean')
print(distances)

# Generate visualizations
comparator.plot_vectors()
comparator.plot_distance_heatmap()

# Generate comprehensive report
report = comparator.generate_report()
```

## API Reference

### VectorComparator Class

#### Constructor
```python
VectorComparator(vectors, labels=None, normalize=False, normalization_method='l2')
```

**Parameters:**
- `vectors`: List of 1D numpy arrays or lists to compare
- `labels`: Optional list of string labels for each vector
- `normalize`: Whether to normalize vectors upon initialization
- `normalization_method`: Method for normalization ('l2', 'standardize', 'minmax')

#### Core Methods

##### Statistical Analysis
- `statistical_summary()`: Returns pandas DataFrame with comprehensive statistics
- `magnitude_comparison()`: Returns dictionary with norms, rankings, and ratios
- `component_wise_analysis(reference_idx=0)`: Element-wise analysis vs reference vector

##### Distance and Similarity
- `distance_matrix(metric='euclidean')`: Calculate distance matrix
- `similarity_matrix(metric='cosine')`: Calculate similarity matrix

##### Normalization
- `normalize_vectors(method='l2')`: Create new comparator with normalized vectors

##### Visualization
- `plot_vectors(style='line')`: Plot all vectors (line or bar chart)
- `plot_distance_heatmap(metric='euclidean')`: Heatmap of distance/similarity matrix
- `plot_component_analysis(reference_idx=0)`: Detailed component-wise analysis plots

##### Reporting
- `generate_report(include_plots=True, save_path=None)`: Generate comprehensive report

## Examples

### Basic Comparison
```python
# Compare optimization results
vectors = [result1, result2, result3]
comparator = VectorComparator(vectors, labels=["Method A", "Method B", "Method C"])

# Get basic statistics
stats = comparator.statistical_summary()
print(stats[['Vector', 'Mean', 'Std', 'Min', 'Max']])
```

### Distance Analysis
```python
# Calculate distances between all pairs
distances = comparator.distance_matrix('euclidean')
similarities = comparator.similarity_matrix('cosine')

# Visualize distance matrix
comparator.plot_distance_heatmap('euclidean')
```

### Component-wise Analysis
```python
# Compare all vectors against the first one
analysis = comparator.component_wise_analysis(reference_idx=0)

# Visualize detailed analysis
comparator.plot_component_analysis(reference_idx=0)
```

### Normalization
```python
# Create normalized version
normalized = comparator.normalize_vectors('l2')

# Compare original vs normalized
print("Original norms:", comparator.magnitude_comparison()['norms'])
print("Normalized norms:", normalized.magnitude_comparison()['norms'])
```

### Comprehensive Report
```python
# Generate full report
report = comparator.generate_report(save_path='comparison_report.json')

# Access specific sections
print("Summary:", report['summary'])
print("Distances:", report['distance_matrix'])
```

## Advanced Usage

### Custom Analysis
```python
# Get specific metrics
mag_comp = comparator.magnitude_comparison()
print(f"Most similar vectors: {mag_comp['rankings']}")

# Component analysis with custom reference
analysis = comparator.component_wise_analysis(reference_idx=1)
for label, data in analysis.items():
    if label != 'reference':
        print(f"{label}: max diff = {data['max_difference']:.3f}")
```

### Batch Processing
```python
# Process multiple sets of vectors
results = []
for i, vector_set in enumerate(vector_sets):
    comp = VectorComparator(vector_set, labels=[f"Set_{i}_{j}" for j in range(len(vector_set))])
    report = comp.generate_report(include_plots=False)
    results.append(report)
```

## Error Handling

The module includes comprehensive error handling for common issues:
- Empty vector lists
- Non-1D arrays
- Inconsistent vector lengths (with warnings)
- Invalid normalization methods
- Invalid distance/similarity metrics

## Performance Notes

- Distance calculations use scipy's optimized pdist function
- Large vector sets may require significant memory for distance matrices
- Visualization methods may be slow for very large vectors (>1000 elements)

## Contributing

This module is part of the merger retrospective studies project. For issues or improvements, please refer to the main project repository.

## License

See the main project LICENSE file for details.
