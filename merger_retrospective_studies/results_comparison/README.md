# Optimization Results Visualization Module

A comprehensive Python module for visualizing optimization results, including convergence analysis, parameter distributions, and solution similarities. This module is designed for merger retrospective studies and provides publication-quality visualizations for optimization analysis.

## Features

- **Objective Function Analysis**: Bar charts showing objective values across solutions
- **Convergence Analysis**: Gradient norm visualization with convergence thresholds
- **Hessian Analysis**: Scatter plots of min vs max eigenvalues to identify local minima
- **Parameter Similarity**: Pairwise distance analysis using Euclidean, Manhattan, and cosine metrics
- **Heatmap Visualizations**: Comprehensive similarity/distance matrices
- **Dashboard Creation**: Combined visualization dashboard for comprehensive analysis
- **Summary Statistics**: Automated analysis and reporting of key metrics

## Installation

### Prerequisites

- Python 3.7+
- Required packages (see requirements.txt)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Quick Setup

```python
# Clone or download the module files
# Ensure optimization_visualizer.py is in your Python path
from optimization_visualizer import OptimizationVisualizer
```

## Data Format

The module expects optimization results in the following format:

```python
# NumPy array or pandas DataFrame with shape (n_solutions, n_features)
# Columns:
# 0: row_index          # Solution identifier
# 1: objective          # Objective function value
# 2: projected_gradient_norm  # Convergence metric
# 3: min_reduced_hessian      # Minimum eigenvalue
# 4: max_reduced_hessian      # Maximum eigenvalue
# 5-end: parameter_values     # Optimization parameters (sigma_*, pi_prices_*, pi_tar_*)
```

### Example Data Structure

```python
import numpy as np

# Example data with 3 solutions and 5 parameters
data = np.array([
    [0, 2.965584, 5.87551e-04, 1.171039e-01, 3.515181e+03, 7.188430, 0.0, 1.2, 0.5, 2.1],
    [1, 0.033116, 4.374441e-05, -8.592206e-15, 7.149121e+01, 6.826741, 0.0, 1.1, 0.4, 2.0],
    [2, 1.234567, 1.234567e-03, 2.345678e-01, 4.567890e+02, 5.678901, 0.1, 1.3, 0.6, 2.2]
])
```

## Quick Start

### Basic Usage

```python
import numpy as np
from optimization_visualizer import OptimizationVisualizer

# Load your optimization data
data = np.load('optimization_results.npy')  # or pd.read_csv('results.csv')

# Create visualizer
viz = OptimizationVisualizer(data, parameter_start_col=5)

# Print summary statistics
viz.print_summary()

# Generate individual plots
viz.plot_objective_function(save_path='objective.png')
viz.plot_gradient_norm(save_path='gradient.png')
viz.plot_hessian_eigenvalues(save_path='hessian.png')

# Create comprehensive dashboard
viz.create_dashboard(save_path='dashboard.png')
```

### Generate All Plots

```python
# Generate all plots to a directory
viz.generate_all_plots(output_dir='./optimization_plots')
```

## Detailed Usage

### 1. Individual Plot Types

#### Objective Function Plot
```python
fig = viz.plot_objective_function(figsize=(10, 6), save_path='objective.png')
```
- Bar chart showing objective values
- Highlights best and worst solutions
- Color-coded by solution index

#### Gradient Norm Plot
```python
fig = viz.plot_gradient_norm(figsize=(10, 6), save_path='gradient.png')
```
- Log-scale bar chart of gradient norms
- Shows convergence threshold (1e-6)
- Identifies best converged solutions

#### Hessian Eigenvalues Plot
```python
fig = viz.plot_hessian_eigenvalues(figsize=(10, 6), save_path='hessian.png')
```
- Scatter plot of min vs max eigenvalues
- Identifies local minima (positive min eigenvalues)
- Labels saddle points (negative min eigenvalues)

#### Distance Analysis
```python
# Pairwise distances bar chart
fig = viz.plot_pairwise_distances(metric='euclidean', save_path='distances.png')

# Distance heatmap
fig = viz.plot_distance_heatmap(metric='euclidean', save_path='heatmap.png')
```

Available metrics:
- `'euclidean'`: Euclidean distance
- `'manhattan'`: Manhattan distance  
- `'cosine'`: Cosine similarity

### 2. Comprehensive Dashboard

```python
fig = viz.create_dashboard(figsize=(20, 16), save_path='dashboard.png')
```

The dashboard includes:
- Row 1: Objective function, Gradient norm
- Row 2: Hessian scatter, Euclidean distances
- Row 3: Euclidean heatmap, Manhattan heatmap
- Row 4: Cosine similarity heatmap

### 3. Summary Statistics

```python
# Get detailed statistics
stats = viz.get_summary_statistics()
print(f"Best objective: {stats['best_objective']}")
print(f"Local minima: {stats['local_minima']}")
print(f"Convergence rate: {stats['converged_solutions']}/{stats['n_solutions']}")

# Print formatted summary
viz.print_summary()
```

## Advanced Usage

### Custom Figure Sizes

```python
# Customize figure sizes
viz.plot_objective_function(figsize=(15, 10))
viz.create_dashboard(figsize=(24, 18))
```

### Distance Metric Analysis

```python
# Compare different distance metrics
for metric in ['euclidean', 'manhattan', 'cosine']:
    viz.plot_distance_heatmap(metric=metric, save_path=f'heatmap_{metric}.png')
```

### Batch Processing

```python
# Process multiple datasets
datasets = ['results1.npy', 'results2.npy', 'results3.npy']

for i, dataset in enumerate(datasets):
    data = np.load(dataset)
    viz = OptimizationVisualizer(data)
    viz.generate_all_plots(output_dir=f'./results_{i+1}')
```

## API Reference

### OptimizationVisualizer Class

#### Constructor
```python
OptimizationVisualizer(data, parameter_start_col=5)
```

**Parameters:**
- `data`: NumPy array or pandas DataFrame with optimization results
- `parameter_start_col`: Column index where parameters start (default: 5)

#### Methods

##### Plotting Methods
- `plot_objective_function(figsize=(10, 6), save_path=None)`: Objective function bar chart
- `plot_gradient_norm(figsize=(10, 6), save_path=None)`: Gradient norm log-scale plot
- `plot_hessian_eigenvalues(figsize=(10, 6), save_path=None)`: Hessian eigenvalues scatter plot
- `plot_pairwise_distances(metric='euclidean', figsize=(10, 6), save_path=None)`: Distance bar chart
- `plot_distance_heatmap(metric='euclidean', figsize=(10, 8), save_path=None)`: Distance heatmap
- `create_dashboard(figsize=(20, 16), save_path=None)`: Comprehensive dashboard

##### Utility Methods
- `generate_all_plots(output_dir='./optimization_plots')`: Generate all plots
- `get_summary_statistics()`: Get detailed statistics dictionary
- `print_summary()`: Print formatted summary to console

##### Internal Methods
- `_calculate_pairwise_distances(metric)`: Calculate pairwise distances
- `_get_color_palette(n)`: Get color palette for n solutions

## Examples

### Example 1: Basic Analysis

```python
import numpy as np
from optimization_visualizer import OptimizationVisualizer

# Create sample data
data = np.array([
    [0, 2.96, 5.88e-04, 0.117, 3515.18, 7.19, 0.0, 1.2, 0.5],
    [1, 0.033, 4.37e-05, -8.59e-15, 71.49, 6.83, 0.0, 1.1, 0.4],
    [2, 1.23, 1.23e-03, 0.235, 456.79, 5.68, 0.1, 1.3, 0.6]
])

# Analyze
viz = OptimizationVisualizer(data, parameter_start_col=5)
viz.print_summary()
viz.create_dashboard(save_path='analysis.png')
```

### Example 2: Convergence Analysis

```python
# Focus on convergence
viz.plot_gradient_norm(save_path='convergence.png')

# Check for local minima
stats = viz.get_summary_statistics()
print(f"Local minima found: {stats['local_minima']}")
print(f"Convergence rate: {stats['converged_solutions']}/{stats['n_solutions']}")
```

### Example 3: Parameter Similarity

```python
# Analyze parameter similarities
viz.plot_distance_heatmap(metric='euclidean', save_path='euclidean_similarity.png')
viz.plot_distance_heatmap(metric='cosine', save_path='cosine_similarity.png')

# Get most similar/dissimilar pairs
stats = viz.get_summary_statistics()
print(f"Most similar: {stats['most_similar_pair']}")
print(f"Most dissimilar: {stats['most_dissimilar_pair']}")
```

## Testing

Run the comprehensive test suite:

```bash
python test_optimization_visualizer.py
```

Or run individual test categories:

```python
import unittest
from test_optimization_visualizer import TestOptimizationVisualizer

# Run specific tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestOptimizationVisualizer)
unittest.TextTestRunner(verbosity=2).run(suite)
```

## Error Handling

The module includes comprehensive error handling:

- **Data Validation**: Checks for proper data format and dimensions
- **NaN/Inf Handling**: Warns about problematic values but continues processing
- **Edge Cases**: Handles single solutions, identical solutions, etc.
- **Invalid Metrics**: Raises clear error messages for unsupported distance metrics

## Performance Considerations

- **Distance Caching**: Pairwise distances are calculated once and cached
- **Memory Efficient**: Uses NumPy arrays for efficient computation
- **Scalable**: Handles datasets with hundreds of solutions
- **Optional Dependencies**: Can use Numba for JIT compilation (see requirements.txt)

## Customization

### Color Schemes
The module uses matplotlib's color palettes:
- `tab10` for ≤10 solutions
- `tab20` for ≤20 solutions  
- `gist_rainbow` for >20 solutions

### Plot Styling
- Seaborn whitegrid style
- 300 DPI for publication quality
- Consistent font sizes and styling
- Colorblind-friendly heatmaps

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Format Error**: Check that your data has the correct column structure
   ```python
   print(f"Data shape: {data.shape}")
   print(f"Expected: (n_solutions, 5 + n_parameters)")
   ```

3. **Memory Issues**: For large datasets, consider processing in batches
   ```python
   # Process subsets
   for i in range(0, len(data), 50):
       subset = data[i:i+50]
       viz = OptimizationVisualizer(subset)
       viz.generate_all_plots(output_dir=f'./batch_{i//50}')
   ```

4. **Plot Not Saving**: Check file permissions and directory existence
   ```python
   import os
   os.makedirs('output_directory', exist_ok=True)
   ```

## Contributing

To contribute to this module:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This module is part of the merger retrospective studies project. Please refer to the main project license.

## Citation

If you use this module in your research, please cite:

```bibtex
@software{optimization_visualizer,
  title={Optimization Results Visualization Module},
  author={Merger Retrospective Studies Team},
  year={2024},
  url={https://github.com/your-repo/merger-retrospective-studies}
}
```

## Support

For questions, issues, or feature requests, please:

1. Check the troubleshooting section above
2. Review the test cases for usage examples
3. Open an issue on the project repository
4. Contact the development team

---

**Note**: This module is designed specifically for optimization analysis in merger retrospective studies but can be adapted for other optimization visualization needs.
