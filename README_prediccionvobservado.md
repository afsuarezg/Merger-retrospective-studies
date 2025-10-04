# Prediction vs Observation Comparison Module

This module provides comprehensive statistical and visual analysis tools for comparing predictions against multiple observed values. It's particularly useful for evaluating model performance, forecast accuracy, and understanding how predictions relate to actual outcomes.

## Features

### Statistical Analysis
- **Central Tendency Comparison**: Mean, median, and percentage differences
- **Distribution Analysis**: Range, standard deviation, z-scores, percentile ranks
- **Error Metrics**: MAE, RMSE, MAPE, and bias calculations
- **Statistical Testing**: One-sample t-tests with confidence intervals

### Visualizations
- **Scatter Plots**: Observations vs predictions with trend lines
- **Box Plots**: Distribution comparison between observations and predictions
- **Histograms**: Distribution of observations with prediction markers
- **Error Bar Plots**: Predictions with observation uncertainty

### Data Handling
- **Missing Data**: Automatic handling of NaN values with reporting
- **Multiple Predictions**: Support for comparing multiple predictions simultaneously
- **Flexible Input**: Accepts various data types (lists, arrays, pandas Series)

## Installation

The module requires the following Python packages:
```bash
pip install numpy pandas matplotlib seaborn scipy
```

## Quick Start

### Basic Usage

```python
from prediction_observation_comparison import compare_prediction_observations

# Single prediction vs multiple observations
prediction = 105.5
observations = [98.2, 102.1, 99.8, 104.3, 101.7, 97.9, 103.2, 100.5]

# Run complete analysis
results = compare_prediction_observations(
    prediction_data=prediction,
    observation_data=observations,
    prediction_name="Model Prediction",
    observation_name="Actual Values",
    units="USD"
)

# Print comprehensive report
print(results['report'])
```

### Advanced Usage

```python
from prediction_observation_comparison import PredictionObservationComparison

# Create comparison object
comparison = PredictionObservationComparison(
    prediction_data=[95.0, 105.5, 110.0],  # Multiple predictions
    observation_data=observations,
    prediction_name="Scenario Forecasts",
    observation_name="Historical Data",
    units="USD"
)

# Run individual analyses
central_tendency = comparison.calculate_central_tendency()
distribution = comparison.calculate_distribution_analysis()
error_metrics = comparison.calculate_error_metrics()
statistical_test = comparison.perform_statistical_test()

# Generate visualizations
figures = comparison.create_visualizations()

# Generate comprehensive report
report = comparison.generate_report()
print(report)
```

## API Reference

### PredictionObservationComparison Class

#### Constructor
```python
PredictionObservationComparison(
    prediction_data: Union[float, List[float], pd.Series],
    observation_data: Union[List[float], pd.Series, np.ndarray],
    prediction_name: str = "Prediction",
    observation_name: str = "Observations",
    units: str = ""
)
```

#### Methods

- `calculate_central_tendency()`: Calculate mean, median, and differences
- `calculate_distribution_analysis()`: Analyze distribution properties
- `calculate_error_metrics()`: Compute MAE, RMSE, MAPE, and bias
- `perform_statistical_test(alpha=0.05)`: Run one-sample t-tests
- `create_visualizations(figsize=(15, 10))`: Generate all plots
- `generate_report(alpha=0.05)`: Create comprehensive text report
- `run_full_analysis(alpha=0.05, create_plots=True, figsize=(15, 10))`: Run complete analysis

### Convenience Function

```python
compare_prediction_observations(
    prediction_data: Union[float, List[float], pd.Series],
    observation_data: Union[List[float], pd.Series, np.ndarray],
    prediction_name: str = "Prediction",
    observation_name: str = "Observations",
    units: str = "",
    alpha: float = 0.05,
    create_plots: bool = True,
    figsize: Tuple[int, int] = (15, 10)
) -> Dict[str, Any]
```

## Output Format

The module generates a structured report with the following sections:

```
PREDICTION vs OBSERVATIONS COMPARISON REPORT
=============================================

Prediction Value: [value]
Number of Observations: [n]

1. CENTRAL TENDENCY
   - Observed Mean: [value]
   - Observed Median: [value]
   - Difference from Mean: [value] ([percentage]%)
   - Difference from Median: [value] ([percentage]%)

2. DISTRIBUTION
   - Observed Range: [min] to [max]
   - Prediction within range: [Yes/No]
   - Standard Deviation: [value]
   - Z-score (std deviations from mean): [value]
   - Percentile rank: [value]th percentile

3. ERROR METRICS
   - MAE: [value]
   - RMSE: [value]
   - MAPE: [value]%
   - Bias: [value] ([Over/Under]-estimation)

4. STATISTICAL TEST
   - One-sample t-test p-value: [value]
   - 95% CI for observations: [lower, upper]
   - Conclusion: Prediction is [significantly different/not significantly different] from observations

5. INTERPRETATION
   [Provide a brief narrative summary of the findings]
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Single prediction vs multiple observations
- Multiple predictions comparison
- Economic forecast evaluation
- Custom analysis with detailed results
- Handling missing data

## Use Cases

This module is particularly useful for:

- **Model Validation**: Comparing model predictions against test data
- **Forecast Evaluation**: Assessing the accuracy of economic or business forecasts
- **Merger Analysis**: Evaluating predicted merger effects against actual outcomes
- **Research Studies**: Comparing theoretical predictions with empirical observations
- **Performance Monitoring**: Tracking prediction accuracy over time

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

## License

This module is part of the merger retrospective studies project.
