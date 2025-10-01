"""
Module for comprehensive comparison between predictions and multiple observed values.

This module provides statistical and visual analysis tools to compare single or multiple
predictions against corresponding observed values, including central tendency analysis,
distribution analysis, error metrics, and statistical testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Union, List, Dict, Any, Optional, Tuple
import warnings

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PredictionObservationComparison:
    """
    A comprehensive class for comparing predictions against multiple observed values.
    
    This class provides methods to perform statistical analysis, error metrics calculation,
    and visualization of prediction vs observation comparisons.
    """
    
    def __init__(self, prediction_data: Union[float, List[float], pd.Series], 
                 observation_data: Union[List[float], pd.Series, np.ndarray],
                 prediction_name: str = "Prediction",
                 observation_name: str = "Observations",
                 units: str = ""):
        """
        Initialize the comparison object.
        
        Parameters:
        -----------
        prediction_data : float, list, or pd.Series
            The predicted value(s) to compare
        observation_data : list, pd.Series, or np.ndarray
            The observed values to compare against
        prediction_name : str, optional
            Name for the prediction data (default: "Prediction")
        observation_name : str, optional
            Name for the observation data (default: "Observations")
        units : str, optional
            Units of measurement for the data
        """
        self.prediction_name = prediction_name
        self.observation_name = observation_name
        self.units = units
        
        # Convert prediction data to numpy array
        if isinstance(prediction_data, (int, float)):
            self.predictions = np.array([prediction_data])
        else:
            self.predictions = np.array(prediction_data)
        
        # Convert observation data to numpy array and handle missing values
        self.observations = np.array(observation_data)
        self.original_obs_count = len(self.observations)
        
        # Remove NaN values
        valid_mask = ~np.isnan(self.observations)
        self.observations = self.observations[valid_mask]
        self.missing_count = self.original_obs_count - len(self.observations)
        
        if len(self.observations) == 0:
            raise ValueError("No valid observations found after removing NaN values")
    
    def calculate_central_tendency(self) -> Dict[str, Any]:
        """Calculate central tendency measures for observations."""
        obs_mean = np.mean(self.observations)
        obs_median = np.median(self.observations)
        
        results = {
            'observed_mean': obs_mean,
            'observed_median': obs_median,
            'prediction_vs_mean': {},
            'prediction_vs_median': {}
        }
        
        for i, pred in enumerate(self.predictions):
            diff_mean = pred - obs_mean
            diff_median = pred - obs_median
            pct_diff_mean = (diff_mean / obs_mean) * 100 if obs_mean != 0 else np.inf
            pct_diff_median = (diff_median / obs_median) * 100 if obs_median != 0 else np.inf
            
            results['prediction_vs_mean'][f'pred_{i}'] = {
                'difference': diff_mean,
                'percentage': pct_diff_mean,
                'overestimates': diff_mean > 0
            }
            
            results['prediction_vs_median'][f'pred_{i}'] = {
                'difference': diff_median,
                'percentage': pct_diff_median,
                'overestimates': diff_median > 0
            }
        
        return results
    
    def calculate_distribution_analysis(self) -> Dict[str, Any]:
        """Analyze how predictions relate to the distribution of observations."""
        obs_min = np.min(self.observations)
        obs_max = np.max(self.observations)
        obs_std = np.std(self.observations, ddof=1)  # Sample standard deviation
        obs_mean = np.mean(self.observations)
        
        results = {
            'observed_range': (obs_min, obs_max),
            'standard_deviation': obs_std,
            'prediction_analysis': {}
        }
        
        for i, pred in enumerate(self.predictions):
            within_range = obs_min <= pred <= obs_max
            z_score = (pred - obs_mean) / obs_std if obs_std != 0 else 0
            
            # Calculate percentile rank
            percentile_rank = (np.sum(self.observations <= pred) / len(self.observations)) * 100
            
            results['prediction_analysis'][f'pred_{i}'] = {
                'within_range': within_range,
                'z_score': z_score,
                'percentile_rank': percentile_rank
            }
        
        return results
    
    def calculate_error_metrics(self) -> Dict[str, Any]:
        """Calculate various error metrics for each prediction."""
        results = {}
        
        for i, pred in enumerate(self.predictions):
            # Calculate errors
            errors = pred - self.observations
            abs_errors = np.abs(errors)
            squared_errors = errors ** 2
            
            # Error metrics
            mae = np.mean(abs_errors)
            rmse = np.sqrt(np.mean(squared_errors))
            
            # MAPE - handle division by zero
            mape_mask = self.observations != 0
            if np.any(mape_mask):
                mape = np.mean(np.abs(errors[mape_mask] / self.observations[mape_mask])) * 100
            else:
                mape = np.inf
            
            bias = np.mean(errors)
            
            results[f'pred_{i}'] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'bias': bias,
                'overestimation': bias > 0
            }
        
        return results
    
    def perform_statistical_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform one-sample t-test for each prediction."""
        results = {}
        
        for i, pred in enumerate(self.predictions):
            # One-sample t-test: test if observations are significantly different from prediction
            t_stat, p_value = stats.ttest_1samp(self.observations, pred)
            
            # Calculate confidence interval for observations
            n = len(self.observations)
            mean_obs = np.mean(self.observations)
            std_obs = np.std(self.observations, ddof=1)
            se = std_obs / np.sqrt(n)
            
            # 95% confidence interval
            ci_lower = mean_obs - stats.t.ppf(1 - alpha/2, n-1) * se
            ci_upper = mean_obs + stats.t.ppf(1 - alpha/2, n-1) * se
            
            results[f'pred_{i}'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_difference': p_value < alpha,
                'confidence_interval': (ci_lower, ci_upper),
                'alpha': alpha
            }
        
        return results
    
    def create_visualizations(self, figsize: Tuple[int, int] = (15, 10)) -> Dict[str, plt.Figure]:
        """Create comprehensive visualizations comparing predictions and observations."""
        figures = {}
        
        # 1. Scatter plot with prediction lines
        fig1, axes1 = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Observations vs Index with prediction lines
        axes1[0].scatter(range(len(self.observations)), self.observations, 
                        alpha=0.6, label=self.observation_name)
        for i, pred in enumerate(self.predictions):
            axes1[0].axhline(y=pred, color=f'C{i+1}', linestyle='--', 
                           label=f'{self.prediction_name} {i+1}' if len(self.predictions) > 1 else self.prediction_name)
        axes1[0].set_xlabel('Observation Index')
        axes1[0].set_ylabel(f'Value {self.units}')
        axes1[0].set_title('Observations vs Predictions')
        axes1[0].legend()
        axes1[0].grid(True, alpha=0.3)
        
        # Right: Prediction vs Mean comparison
        obs_mean = np.mean(self.observations)
        axes1[1].scatter([obs_mean] * len(self.predictions), self.predictions, 
                        s=100, alpha=0.7, label='Predictions')
        axes1[1].axhline(y=obs_mean, color='red', linestyle='-', 
                        label=f'Observed Mean ({obs_mean:.2f})')
        axes1[1].set_xlabel('Observed Mean')
        axes1[1].set_ylabel(f'Prediction Value {self.units}')
        axes1[1].set_title('Predictions vs Observed Mean')
        axes1[1].legend()
        axes1[1].grid(True, alpha=0.3)
        
        figures['scatter_plots'] = fig1
        
        # 2. Box plot with predictions
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        box_data = [self.observations] + [[pred] for pred in self.predictions]
        labels = [self.observation_name] + [f'{self.prediction_name} {i+1}' for i in range(len(self.predictions))]
        
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        colors = ['lightblue'] + [f'C{i}' for i in range(len(self.predictions))]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel(f'Value {self.units}')
        ax2.set_title('Distribution of Observations vs Predictions')
        ax2.grid(True, alpha=0.3)
        
        figures['box_plot'] = fig2
        
        # 3. Histogram with prediction lines
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.hist(self.observations, bins=min(30, len(self.observations)//2), 
                alpha=0.7, label=self.observation_name, density=True)
        
        for i, pred in enumerate(self.predictions):
            ax3.axvline(x=pred, color=f'C{i+1}', linestyle='--', linewidth=2,
                       label=f'{self.prediction_name} {i+1}' if len(self.predictions) > 1 else self.prediction_name)
        
        ax3.set_xlabel(f'Value {self.units}')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Observations with Predictions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        figures['histogram'] = fig3
        
        # 4. Error bars plot
        if len(self.predictions) > 0:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            
            obs_mean = np.mean(self.observations)
            obs_std = np.std(self.observations, ddof=1)
            n = len(self.predictions)
            
            x_pos = np.arange(n)
            ax4.errorbar(x_pos, self.predictions, yerr=obs_std, 
                        fmt='o', capsize=5, capthick=2, 
                        label=f'Predictions ± 1 std of observations')
            ax4.axhline(y=obs_mean, color='red', linestyle='-', 
                       label=f'Observed Mean ({obs_mean:.2f})')
            
            ax4.set_xlabel('Prediction Index')
            ax4.set_ylabel(f'Value {self.units}')
            ax4.set_title('Predictions with Observation Uncertainty')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'Pred {i+1}' for i in range(n)])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            figures['error_bars'] = fig4
        
        return figures
    
    def generate_report(self, alpha: float = 0.05) -> str:
        """Generate a comprehensive comparison report."""
        # Calculate all analyses
        central_tendency = self.calculate_central_tendency()
        distribution = self.calculate_distribution_analysis()
        error_metrics = self.calculate_error_metrics()
        statistical_test = self.perform_statistical_test(alpha)
        
        # Start building report
        report = []
        report.append("PREDICTION vs OBSERVATIONS COMPARISON REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Basic info
        report.append(f"Prediction Value(s): {self.predictions}")
        report.append(f"Number of Observations: {len(self.observations)}")
        if self.missing_count > 0:
            report.append(f"Missing Values Removed: {self.missing_count}")
        if self.units:
            report.append(f"Units: {self.units}")
        report.append("")
        
        # Central tendency
        report.append("1. CENTRAL TENDENCY")
        report.append(f"   - Observed Mean: {central_tendency['observed_mean']:.4f}")
        report.append(f"   - Observed Median: {central_tendency['observed_median']:.4f}")
        report.append("")
        
        for i, pred in enumerate(self.predictions):
            pred_key = f'pred_{i}'
            report.append(f"   Prediction {i+1} ({pred:.4f}):")
            mean_diff = central_tendency['prediction_vs_mean'][pred_key]
            median_diff = central_tendency['prediction_vs_median'][pred_key]
            
            report.append(f"   - Difference from Mean: {mean_diff['difference']:.4f} ({mean_diff['percentage']:.2f}%)")
            report.append(f"   - Difference from Median: {median_diff['difference']:.4f} ({median_diff['percentage']:.2f}%)")
            report.append(f"   - {'Overestimates' if mean_diff['overestimates'] else 'Underestimates'} typical observation")
            report.append("")
        
        # Distribution analysis
        report.append("2. DISTRIBUTION")
        report.append(f"   - Observed Range: {distribution['observed_range'][0]:.4f} to {distribution['observed_range'][1]:.4f}")
        report.append(f"   - Standard Deviation: {distribution['standard_deviation']:.4f}")
        report.append("")
        
        for i, pred in enumerate(self.predictions):
            pred_key = f'pred_{i}'
            dist_analysis = distribution['prediction_analysis'][pred_key]
            report.append(f"   Prediction {i+1}:")
            report.append(f"   - Within range: {'Yes' if dist_analysis['within_range'] else 'No'}")
            report.append(f"   - Z-score: {dist_analysis['z_score']:.4f}")
            report.append(f"   - Percentile rank: {dist_analysis['percentile_rank']:.1f}th percentile")
            report.append("")
        
        # Error metrics
        report.append("3. ERROR METRICS")
        for i, pred in enumerate(self.predictions):
            pred_key = f'pred_{i}'
            metrics = error_metrics[pred_key]
            report.append(f"   Prediction {i+1}:")
            report.append(f"   - MAE: {metrics['mae']:.4f}")
            report.append(f"   - RMSE: {metrics['rmse']:.4f}")
            report.append(f"   - MAPE: {metrics['mape']:.2f}%")
            report.append(f"   - Bias: {metrics['bias']:.4f} ({'Over' if metrics['overestimation'] else 'Under'}-estimation)")
            report.append("")
        
        # Statistical test
        report.append("4. STATISTICAL TEST")
        for i, pred in enumerate(self.predictions):
            pred_key = f'pred_{i}'
            test_results = statistical_test[pred_key]
            report.append(f"   Prediction {i+1}:")
            report.append(f"   - One-sample t-test p-value: {test_results['p_value']:.6f}")
            report.append(f"   - 95% CI for observations: [{test_results['confidence_interval'][0]:.4f}, {test_results['confidence_interval'][1]:.4f}]")
            significance = "significantly different" if test_results['significant_difference'] else "not significantly different"
            report.append(f"   - Conclusion: Prediction is {significance} from observations")
            report.append("")
        
        # Interpretation
        report.append("5. INTERPRETATION")
        
        # Overall assessment
        obs_mean = central_tendency['observed_mean']
        obs_std = distribution['standard_deviation']
        
        if len(self.predictions) == 1:
            pred = self.predictions[0]
            mean_diff = central_tendency['prediction_vs_mean']['pred_0']
            z_score = distribution['prediction_analysis']['pred_0']['z_score']
            
            if abs(z_score) < 1:
                accuracy = "very close to"
            elif abs(z_score) < 2:
                accuracy = "reasonably close to"
            else:
                accuracy = "quite different from"
            
            report.append(f"   The prediction ({pred:.4f}) is {accuracy} the observed mean ({obs_mean:.4f}).")
            
            if mean_diff['overestimates']:
                report.append(f"   It overestimates the typical observation by {abs(mean_diff['percentage']):.1f}%.")
            else:
                report.append(f"   It underestimates the typical observation by {abs(mean_diff['percentage']):.1f}%.")
            
            if distribution['prediction_analysis']['pred_0']['within_range']:
                report.append("   The prediction falls within the observed range, suggesting it's plausible.")
            else:
                report.append("   The prediction falls outside the observed range, which may indicate an outlier.")
        else:
            report.append(f"   Multiple predictions were compared against {len(self.observations)} observations.")
            report.append(f"   The observed values have a mean of {obs_mean:.4f} and standard deviation of {obs_std:.4f}.")
            report.append("   Individual prediction assessments are provided above.")
        
        return "\n".join(report)
    
    def run_full_analysis(self, alpha: float = 0.05, create_plots: bool = True, 
                         figsize: Tuple[int, int] = (15, 10)) -> Dict[str, Any]:
        """
        Run the complete analysis and return all results.
        
        Parameters:
        -----------
        alpha : float
            Significance level for statistical tests (default: 0.05)
        create_plots : bool
            Whether to create visualizations (default: True)
        figsize : tuple
            Figure size for plots (default: (15, 10))
        
        Returns:
        --------
        dict : Complete analysis results including all calculations and plots
        """
        results = {
            'central_tendency': self.calculate_central_tendency(),
            'distribution_analysis': self.calculate_distribution_analysis(),
            'error_metrics': self.calculate_error_metrics(),
            'statistical_test': self.perform_statistical_test(alpha),
            'report': self.generate_report(alpha)
        }
        
        if create_plots:
            results['visualizations'] = self.create_visualizations(figsize)
        
        return results


def compare_prediction_observations(prediction_data: Union[float, List[float], pd.Series], 
                                  observation_data: Union[List[float], pd.Series, np.ndarray],
                                  prediction_name: str = "Prediction",
                                  observation_name: str = "Observations",
                                  units: str = "",
                                  alpha: float = 0.05,
                                  create_plots: bool = True,
                                  figsize: Tuple[int, int] = (15, 10)) -> Dict[str, Any]:
    """
    Convenience function to perform complete prediction vs observation comparison.
    
    Parameters:
    -----------
    prediction_data : float, list, or pd.Series
        The predicted value(s) to compare
    observation_data : list, pd.Series, or np.ndarray
        The observed values to compare against
    prediction_name : str, optional
        Name for the prediction data (default: "Prediction")
    observation_name : str, optional
        Name for the observation data (default: "Observations")
    units : str, optional
        Units of measurement for the data
    alpha : float
        Significance level for statistical tests (default: 0.05)
    create_plots : bool
        Whether to create visualizations (default: True)
    figsize : tuple
        Figure size for plots (default: (15, 10))
    
    Returns:
    --------
    dict : Complete analysis results
    """
    comparison = PredictionObservationComparison(
        prediction_data=prediction_data,
        observation_data=observation_data,
        prediction_name=prediction_name,
        observation_name=observation_name,
        units=units
    )
    
    return comparison.run_full_analysis(alpha=alpha, create_plots=create_plots, figsize=figsize)


if __name__ == "__main__":
    """
    Example implementation demonstrating the prediction vs observation comparison.
    """
    print("Running Prediction vs Observation Comparison Example")
    print("=" * 60)
    
    # Example 1: Single prediction vs multiple observations
    print("\n1. Single Prediction Example:")
    print("-" * 30)
    
    # Simulate some observed values (e.g., actual stock prices)
    np.random.seed(42)
    observed_prices = np.random.normal(100, 15, 50)  # Mean=100, Std=15, 50 observations
    predicted_price = 105.5  # Single prediction
    
    # Run comparison
    results1 = compare_prediction_observations(
        prediction_data=predicted_price,
        observation_data=observed_prices,
        prediction_name="Predicted Price",
        observation_name="Actual Prices",
        units="USD"
    )
    
    print(results1['report'])
    breakpoint()
    # Show plots
    if 'visualizations' in results1:
        pass
        # plt.show()
    
    # Example 2: Multiple predictions vs observations
    print("\n\n2. Multiple Predictions Example:")
    print("-" * 35)
    
    # Simulate multiple predictions for different scenarios
    predicted_prices = [95.0, 105.5, 110.0]  # Three different predictions
    observed_prices_2 = np.random.normal(102, 12, 30)  # Different set of observations
    
    # Run comparison
    results2 = compare_prediction_observations(
        prediction_data=predicted_prices,
        observation_data=observed_prices_2,
        prediction_name="Scenario Prediction",
        observation_name="Market Prices",
        units="USD"
    )
    
    print(results2['report'])
    
    # Show plots
    if 'visualizations' in results2:
        pass
        # plt.show()
    
    # Example 3: Real-world example with economic data
    print("\n\n3. Economic Data Example:")
    print("-" * 30)
    
    # Simulate economic indicators
    gdp_growth_observed = np.random.normal(2.5, 0.8, 20)  # GDP growth rates
    gdp_growth_predicted = 2.8  # Predicted growth rate
    
    results3 = compare_prediction_observations(
        prediction_data=gdp_growth_predicted,
        observation_data=gdp_growth_observed,
        prediction_name="IMF Forecast",
        observation_name="Actual GDP Growth",
        units="%"
    )
    
    print(results3['report'])
    
    # Show plots
    if 'visualizations' in results3:
        pass
        #plt.show()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("The module provides comprehensive statistical and visual analysis")
    print("for comparing predictions against multiple observed values.")
