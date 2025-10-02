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
from scipy.stats import ks_2samp, mannwhitneyu, chi2_contingency
from typing import Union, List, Dict, Any, Optional, Tuple
import warnings
import plotly.graph_objects as go
import plotly.express as px

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

    # =============================
    # Single-prediction visualizations
    # =============================
    def _validate_inputs_single(self, observations: Union[List[float], np.ndarray], prediction: float) -> np.ndarray:
        """Validate inputs and handle edge cases for single prediction visuals."""
        observations = np.array(observations)
        if len(observations) < 3:
            warnings.warn("Very few observations (< 3). Box plot may not be meaningful.")
        if np.any(np.isnan(observations)) or np.any(np.isinf(observations)):
            raise ValueError("Observations contain NaN or infinite values")
        if np.isnan(prediction) or np.isinf(prediction):
            raise ValueError("Prediction is NaN or infinite")
        if np.allclose(observations, prediction):
            warnings.warn("All observations match prediction exactly (zero residuals)")
        return observations

    def _optimize_for_large_data(self, observations: np.ndarray, max_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample observations for visualization if needed."""
        if len(observations) > max_points:
            warnings.warn(f"Downsampling from {len(observations)} to {max_points} points for visualization")
            indices = np.linspace(0, len(observations) - 1, max_points, dtype=int)
            return observations[indices], indices
        return observations, np.arange(len(observations))

    # --- Scatter Plot with Reference Line ---
    def plot_scatter_with_reference(self, observations: Union[List[float], np.ndarray], prediction: float, observation_ids: Optional[List[str]] = None):
        """
        Create scatter plot with prediction reference line (Matplotlib).
        """
        observations = self._validate_inputs_single(observations, prediction)
        observations, _ = self._optimize_for_large_data(observations)

        fig, ax = plt.subplots(figsize=(10, 6))
        if observation_ids is None:
            x_values = np.arange(1, len(observations) + 1)
            x_label = 'Observation Index'
        else:
            x_values = np.arange(len(observations))
            x_label = 'Observation ID'

        ax.scatter(x_values, observations, s=100, alpha=0.7, color='#3b82f6', label='Observed Values', zorder=3)
        ax.axhline(y=prediction, color='#ef4444', linestyle='--', linewidth=2, label=f'{self.prediction_name}: {prediction:.2f}', zorder=2)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f'Value {self.units}'.strip(), fontsize=12)
        ax.set_title('Scatter Plot: Observations vs Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, zorder=1)
        if observation_ids is not None:
            ax.set_xticks(x_values)
            ax.set_xticklabels(observation_ids, rotation=45, ha='right')
        plt.tight_layout()
        return fig, ax

    def plot_scatter_with_reference_plotly(self, observations: Union[List[float], np.ndarray], prediction: float, observation_ids: Optional[List[str]] = None):
        """Create interactive scatter plot with Plotly."""
        observations = self._validate_inputs_single(observations, prediction)
        observations, _ = self._optimize_for_large_data(observations)
        if observation_ids is None:
            observation_ids = [f'Obs {i+1}' for i in range(len(observations))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(observations))),
            y=observations,
            mode='markers',
            name=self.observation_name,
            marker=dict(size=10, color='#3b82f6'),
            text=observation_ids,
            hovertemplate='<b>%{text}</b><br>Value: %{y:.2f}<extra></extra>'
        ))

        fig.add_hline(
            y=prediction,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text=f"{self.prediction_name}: {prediction:.2f}",
            annotation_position="right"
        )

        fig.update_layout(
            title='Scatter Plot: Observations vs Prediction',
            xaxis_title='Observation Index',
            yaxis_title=f'Value {self.units}'.strip(),
            hovermode='closest',
            showlegend=True
        )
        return fig

    def plot_scatter_with_reference_seaborn(self, observations: Union[List[float], np.ndarray], prediction: float, observation_ids: Optional[List[str]] = None):
        """Create scatter plot using Seaborn."""
        import pandas as pd
        observations = self._validate_inputs_single(observations, prediction)
        observations, _ = self._optimize_for_large_data(observations)
        df = pd.DataFrame({
            'index': range(len(observations)),
            'value': observations,
            'id': observation_ids if observation_ids else [f'Obs {i+1}' for i in range(len(observations))]
        })
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='index', y='value', s=100, color='#3b82f6', ax=ax)
        ax.axhline(y=prediction, color='#ef4444', linestyle='--', linewidth=2, label=f'{self.prediction_name}: {prediction:.2f}')
        ax.set_xlabel('Observation Index')
        ax.set_ylabel(f'Value {self.units}'.strip())
        ax.set_title('Scatter Plot: Observations vs Prediction')
        ax.legend()
        plt.tight_layout()
        return fig, ax

    # --- Box Plot with Prediction Marker ---
    def calculate_box_stats(self, observations: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Calculate box plot statistics for observations."""
        observations = np.array(observations)
        stats_dict = {
            'q1': np.percentile(observations, 25),
            'median': np.median(observations),
            'q3': np.percentile(observations, 75),
            'min': np.min(observations),
            'max': np.max(observations),
            'mean': np.mean(observations),
            'std': np.std(observations)
        }
        iqr = stats_dict['q3'] - stats_dict['q1']
        lower_bound = stats_dict['q1'] - 1.5 * iqr
        upper_bound = stats_dict['q3'] + 1.5 * iqr
        stats_dict['iqr'] = iqr
        stats_dict['lower_bound'] = lower_bound
        stats_dict['upper_bound'] = upper_bound
        stats_dict['outliers'] = observations[(observations < lower_bound) | (observations > upper_bound)]
        return stats_dict

    def plot_boxplot_with_prediction(self, observations: Union[List[float], np.ndarray], prediction: float):
        """Create box plot with prediction marker (Matplotlib)."""
        observations = self._validate_inputs_single(observations, prediction)
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.boxplot([observations], vert=True, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor='#3b82f6', alpha=0.5),
                    medianprops=dict(color='#1e40af', linewidth=2),
                    whiskerprops=dict(color='#3b82f6', linewidth=1.5),
                    capprops=dict(color='#3b82f6', linewidth=1.5))
        ax.axhline(y=prediction, color='#ef4444', linestyle='--', linewidth=2.5, label=f'{self.prediction_name}: {prediction:.2f}', zorder=3)
        stats_box = self.calculate_box_stats(observations)
        ax.text(1.15, stats_box['q1'], f"Q1: {stats_box['q1']:.2f}", verticalalignment='center', fontsize=9)
        ax.text(1.15, stats_box['median'], f"Median: {stats_box['median']:.2f}", verticalalignment='center', fontsize=9, fontweight='bold')
        ax.text(1.15, stats_box['q3'], f"Q3: {stats_box['q3']:.2f}", verticalalignment='center', fontsize=9)
        ax.set_ylabel(f'Value {self.units}'.strip(), fontsize=12)
        ax.set_title('Box Plot: Distribution with Prediction', fontsize=14, fontweight='bold')
        ax.set_xticks([1])
        ax.set_xticklabels([self.observation_name])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig, ax

    def plot_boxplot_with_prediction_plotly(self, observations: Union[List[float], np.ndarray], prediction: float):
        """Create interactive box plot with Plotly."""
        observations = self._validate_inputs_single(observations, prediction)
        fig = go.Figure()
        fig.add_trace(go.Box(y=observations, name=self.observation_name, marker_color='#3b82f6', boxmean='sd'))
        fig.add_hline(y=prediction, line_dash="dash", line_color="#ef4444", line_width=2.5,
                      annotation_text=f"{self.prediction_name}: {prediction:.2f}", annotation_position="right")
        fig.update_layout(title='Box Plot: Distribution with Prediction', yaxis_title=f'Value {self.units}'.strip(), showlegend=True, height=600)
        return fig

    def plot_boxplot_with_prediction_seaborn(self, observations: Union[List[float], np.ndarray], prediction: float):
        """Create box plot using Seaborn."""
        observations = self._validate_inputs_single(observations, prediction)
        fig, ax = plt.subplots(figsize=(8, 10))
        sns.boxplot(y=observations, color='#3b82f6', ax=ax, width=0.3)
        ax.axhline(y=prediction, color='#ef4444', linestyle='--', linewidth=2.5, label=f'{self.prediction_name}: {prediction:.2f}')
        ax.set_ylabel(f'Value {self.units}'.strip(), fontsize=12)
        ax.set_title('Box Plot: Distribution with Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig, ax

    # --- Residual Plot ---
    def calculate_residuals(self, observations: Union[List[float], np.ndarray], prediction: float, observation_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate residuals and prepare data for plotting."""
        import pandas as pd
        observations = np.array(observations)
        residuals = observations - prediction
        if observation_ids is None:
            observation_ids = [f'Obs {i+1}' for i in range(len(observations))]
        df = pd.DataFrame({
            'observation_id': observation_ids,
            'observed': observations,
            'predicted': prediction,
            'residual': residuals,
            'abs_residual': np.abs(residuals)
        })
        df['mean_residual'] = residuals.mean()
        df['std_residual'] = residuals.std()
        return df

    def plot_residuals(self, observations: Union[List[float], np.ndarray], prediction: float, observation_ids: Optional[List[str]] = None):
        """Create residual plot (Matplotlib)."""
        residuals_df = self.calculate_residuals(observations, prediction, observation_ids)
        fig, ax = plt.subplots(figsize=(12, 6))
        x_values = np.arange(len(residuals_df))
        colors = ['#3b82f6' if r >= 0 else '#ef4444' for r in residuals_df['residual']]
        ax.bar(x_values, residuals_df['residual'], color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='#374151', linestyle='-', linewidth=2, zorder=1)
        mean_residual = residuals_df['residual'].mean()
        if abs(mean_residual) > 0.01:
            ax.axhline(y=mean_residual, color='#f97316', linestyle='--', linewidth=1.5, label=f'Mean Residual: {mean_residual:.2f}', zorder=2)
        ax.set_xlabel('Observation', fontsize=12)
        ax.set_ylabel('Residual (Observed - Predicted)', fontsize=12)
        ax.set_title('Residual Plot: Prediction Errors', fontsize=14, fontweight='bold')
        ax.set_xticks(x_values)
        ax.set_xticklabels(residuals_df['observation_id'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3b82f6', alpha=0.7, label='Positive Residual (Over-prediction)'),
            Patch(facecolor='#ef4444', alpha=0.7, label='Negative Residual (Under-prediction)')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        plt.tight_layout()
        return fig, ax

    def plot_residuals_plotly(self, observations: Union[List[float], np.ndarray], prediction: float, observation_ids: Optional[List[str]] = None):
        """Create interactive residual plot with Plotly."""
        residuals_df = self.calculate_residuals(observations, prediction, observation_ids)
        colors = ['#3b82f6' if r >= 0 else '#ef4444' for r in residuals_df['residual']]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=residuals_df['observation_id'],
            y=residuals_df['residual'],
            marker_color=colors,
            name='Residual',
            hovertemplate='<b>%{x}</b><br>' +
                          'Observed: %{customdata[0]:.2f}<br>' +
                          'Predicted: %{customdata[1]:.2f}<br>' +
                          'Residual: %{y:.2f}<extra></extra>',
            customdata=residuals_df[['observed', 'predicted']].values
        ))
        fig.add_hline(y=0, line_color='#374151', line_width=2)
        mean_residual = residuals_df['residual'].mean()
        if abs(mean_residual) > 0.01:
            fig.add_hline(y=mean_residual, line_dash="dash", line_color="#f97316", line_width=1.5,
                          annotation_text=f"Mean: {mean_residual:.2f}", annotation_position="right")
        fig.update_layout(title='Residual Plot: Prediction Errors', xaxis_title='Observation', yaxis_title='Residual (Observed - Predicted)', showlegend=False, hovermode='closest')
        return fig

    def visualize_single_prediction(self, observation_ids: Optional[List[str]] = None, plot_type: str = 'all', backend: str = 'matplotlib') -> Dict[str, Any]:
        """
        Generate visualizations for single prediction vs multiple observations.
        Only enabled when exactly one prediction and more than one observation exist.
        """
        if not (len(self.predictions) == 1 and len(self.observations) > 1):
            return {}
        prediction = float(self.predictions[0])
        observations = np.array(self.observations)
        figures: Dict[str, Any] = {}
        if backend == 'matplotlib':
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_scatter_with_reference(observations, prediction, observation_ids)
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_boxplot_with_prediction(observations, prediction)
            if plot_type in ['residual', 'all']:
                figures['residual'] = self.plot_residuals(observations, prediction, observation_ids)
        elif backend == 'plotly':
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_scatter_with_reference_plotly(observations, prediction, observation_ids)
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_boxplot_with_prediction_plotly(observations, prediction)
            if plot_type in ['residual', 'all']:
                figures['residual'] = self.plot_residuals_plotly(observations, prediction, observation_ids)
        elif backend == 'seaborn':
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_scatter_with_reference_seaborn(observations, prediction, observation_ids)
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_boxplot_with_prediction_seaborn(observations, prediction)
            if plot_type in ['residual', 'all']:
                figures['residual'] = self.plot_residuals(observations, prediction, observation_ids)
        return figures

    def save_plots(self, figures: Dict[str, Any], output_dir: str = './plots', format: str = 'png', dpi: int = 300) -> None:
        """Save all generated plots to disk."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, fig_obj in figures.items():
            filepath = os.path.join(output_dir, f'{name}_plot.{format}')
            if isinstance(fig_obj, tuple):
                fig, ax = fig_obj
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            else:
                if format == 'html':
                    fig_obj.write_html(filepath)
                else:
                    fig_obj.write_image(filepath, width=1200, height=800)

    def print_summary_statistics(self, observations: Union[List[float], np.ndarray], prediction: float) -> None:
        """Print summary statistics for observations and prediction."""
        observations = np.array(observations)
        residuals = observations - prediction
        print("=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"{self.prediction_name} Value: {prediction:.2f}")
        print(f"\n{self.observation_name} (n={len(observations)}):")
        print(f"  Mean:   {observations.mean():.2f}")
        print(f"  Median: {np.median(observations):.2f}")
        print(f"  Std:    {observations.std():.2f}")
        print(f"  Min:    {observations.min():.2f}")
        print(f"  Max:    {observations.max():.2f}")
        print(f"  Range:  {observations.max() - observations.min():.2f}")
        print(f"\nResiduals:")
        print(f"  Mean:   {residuals.mean():.2f} (bias)")
        print(f"  Std:    {residuals.std():.2f}")
        print(f"  MAE:    {np.abs(residuals).mean():.2f}")
        print(f"  RMSE:   {np.sqrt((residuals**2).mean()):.2f}")
    
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
    
    def perform_ks_test(self) -> Dict[str, Any]:
        """
        Perform Kolmogorov-Smirnov test to compare distributions.
        
        The KS test compares the empirical distribution functions of two samples
        to determine if they come from the same distribution.
        
        Returns:
        --------
        dict : KS test results for each prediction
        """
        results = {}
        
        for i, pred in enumerate(self.predictions):
            # Create a sample by repeating the prediction to match observation count
            # This allows us to compare the prediction "distribution" with observations
            prediction_sample = np.full(len(self.observations), pred)
            
            # Perform two-sample KS test
            ks_statistic, p_value = ks_2samp(self.observations, prediction_sample)
            
            # Calculate effect size (Cohen's d approximation)
            pooled_std = np.sqrt((np.var(self.observations) + np.var(prediction_sample)) / 2)
            cohens_d = (np.mean(self.observations) - pred) / pooled_std if pooled_std != 0 else 0
            
            results[f'pred_{i}'] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'significant_difference': p_value < 0.05,
                'cohens_d': cohens_d,
                'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
            }
        
        return results
    
    def perform_mann_whitney_test(self) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (Wilcoxon rank-sum test).
        
        This non-parametric test compares two independent samples to determine
        if one tends to have larger values than the other.
        
        Returns:
        --------
        dict : Mann-Whitney U test results for each prediction
        """
        results = {}
        
        for i, pred in enumerate(self.predictions):
            # Create a sample by repeating the prediction
            prediction_sample = np.full(len(self.observations), pred)
            
            # Perform Mann-Whitney U test
            try:
                u_statistic, p_value = mannwhitneyu(
                    self.observations, 
                    prediction_sample, 
                    alternative='two-sided'
                )
                
                # Calculate effect size (r = Z / sqrt(N))
                n1, n2 = len(self.observations), len(prediction_sample)
                z_score = (u_statistic - (n1 * n2) / 2) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
                effect_size_r = abs(z_score) / np.sqrt(n1 + n2)
                
                # Determine which group tends to have larger values
                median_obs = np.median(self.observations)
                median_pred = pred
                obs_tends_larger = median_obs > median_pred
                
            except ValueError as e:
                # Handle edge cases (e.g., identical values)
                u_statistic, p_value = np.nan, 1.0
                effect_size_r = 0.0
                obs_tends_larger = False
            
            results[f'pred_{i}'] = {
                'u_statistic': u_statistic,
                'p_value': p_value,
                'significant_difference': p_value < 0.05,
                'effect_size_r': effect_size_r,
                'effect_size': 'small' if effect_size_r < 0.3 else 'medium' if effect_size_r < 0.5 else 'large',
                'observations_tend_larger': obs_tends_larger
            }
        
        return results
    
    def perform_chi_square_test(self, bins: int = 10) -> Dict[str, Any]:
        """
        Perform Chi-square test of independence.
        
        This test compares the frequency distributions of two samples by
        creating bins and testing if the distributions are independent.
        
        Parameters:
        -----------
        bins : int
            Number of bins to use for creating frequency tables (default: 10)
        
        Returns:
        --------
        dict : Chi-square test results for each prediction
        """
        results = {}
        
        # Create bins based on the combined range of observations and predictions
        all_values = np.concatenate([self.observations, self.predictions])
        bin_edges = np.linspace(np.min(all_values), np.max(all_values), bins + 1)
        
        for i, pred in enumerate(self.predictions):
            # Create frequency tables
            obs_freq, _ = np.histogram(self.observations, bins=bin_edges)
            pred_freq, _ = np.histogram([pred], bins=bin_edges)
            
            # Ensure we have at least 2 bins with non-zero frequencies
            if np.sum(obs_freq) == 0 or np.sum(pred_freq) == 0:
                results[f'pred_{i}'] = {
                    'chi2_statistic': np.nan,
                    'p_value': 1.0,
                    'significant_difference': False,
                    'cramers_v': 0.0,
                    'effect_size': 'none',
                    'warning': 'Insufficient data for chi-square test'
                }
                continue
            
            # Create contingency table
            contingency_table = np.array([obs_freq, pred_freq])
            
            # Remove bins with zero frequencies in both groups
            valid_bins = np.any(contingency_table > 0, axis=0)
            if np.sum(valid_bins) < 2:
                results[f'pred_{i}'] = {
                    'chi2_statistic': np.nan,
                    'p_value': 1.0,
                    'significant_difference': False,
                    'cramers_v': 0.0,
                    'effect_size': 'none',
                    'warning': 'Insufficient variation for chi-square test'
                }
                continue
            
            contingency_table = contingency_table[:, valid_bins]
            
            try:
                # Perform chi-square test
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Calculate Cramer's V (effect size)
                n = np.sum(contingency_table)
                min_dim = min(contingency_table.shape)
                cramers_v = np.sqrt(chi2_stat / (n * (min_dim - 1))) if n > 0 and min_dim > 1 else 0
                
                # Determine effect size
                if cramers_v < 0.1:
                    effect_size = 'negligible'
                elif cramers_v < 0.3:
                    effect_size = 'small'
                elif cramers_v < 0.5:
                    effect_size = 'medium'
                else:
                    effect_size = 'large'
                
            except Exception as e:
                chi2_stat, p_value, cramers_v = np.nan, 1.0, 0.0
                effect_size = 'none'
            
            results[f'pred_{i}'] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05 if not np.isnan(p_value) else False,
                'cramers_v': cramers_v,
                'effect_size': effect_size,
                'degrees_of_freedom': dof if 'dof' in locals() else np.nan
            }
        
        return results
    
    def perform_all_statistical_tests(self, alpha: float = 0.05, chi_square_bins: int = 10) -> Dict[str, Any]:
        """
        Perform all statistical tests for comprehensive comparison.
        
        Parameters:
        -----------
        alpha : float
            Significance level for tests (default: 0.05)
        chi_square_bins : int
            Number of bins for chi-square test (default: 10)
        
        Returns:
        --------
        dict : Results from all statistical tests
        """
        results = {
            'one_sample_t_test': self.perform_statistical_test(alpha),
            'kolmogorov_smirnov': self.perform_ks_test(),
            'mann_whitney_u': self.perform_mann_whitney_test(),
            'chi_square': self.perform_chi_square_test(chi_square_bins),
            'alpha': alpha
        }
        
        # Add summary statistics
        results['summary'] = self._generate_test_summary(results)
        
        return results
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        summary = {
            'total_predictions': len(self.predictions),
            'total_observations': len(self.observations),
            'tests_performed': ['one_sample_t_test', 'kolmogorov_smirnov', 'mann_whitney_u', 'chi_square'],
            'significant_differences': {},
            'effect_sizes': {},
            'recommendations': []
        }
        
        # Analyze significance across tests
        for pred_key in [f'pred_{i}' for i in range(len(self.predictions))]:
            significant_tests = []
            effect_sizes = {}
            
            # Check each test
            for test_name in summary['tests_performed']:
                if pred_key in test_results[test_name]:
                    test_result = test_results[test_name][pred_key]
                    if test_result.get('significant_difference', False):
                        significant_tests.append(test_name)
                    
                    # Collect effect sizes
                    if 'cohens_d' in test_result:
                        effect_sizes[test_name] = test_result['cohens_d']
                    elif 'effect_size_r' in test_result:
                        effect_sizes[test_name] = test_result['effect_size_r']
                    elif 'cramers_v' in test_result:
                        effect_sizes[test_name] = test_result['cramers_v']
            
            summary['significant_differences'][pred_key] = significant_tests
            summary['effect_sizes'][pred_key] = effect_sizes
            
            # Generate recommendations
            if len(significant_tests) >= 3:
                summary['recommendations'].append(f"{pred_key}: Strong evidence of difference from observations")
            elif len(significant_tests) >= 2:
                summary['recommendations'].append(f"{pred_key}: Moderate evidence of difference from observations")
            elif len(significant_tests) == 1:
                summary['recommendations'].append(f"{pred_key}: Weak evidence of difference from observations")
            else:
                summary['recommendations'].append(f"{pred_key}: No significant difference from observations")
        
        return summary
    
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
    
    def generate_report(self, alpha: float = 0.05, include_advanced_tests: bool = False) -> str:
        """Generate a comprehensive comparison report."""
        # Calculate all analyses
        central_tendency = self.calculate_central_tendency()
        distribution = self.calculate_distribution_analysis()
        error_metrics = self.calculate_error_metrics()
        statistical_test = self.perform_statistical_test(alpha)
        
        # Calculate advanced statistical tests if requested
        advanced_tests = None
        if include_advanced_tests:
            advanced_tests = self.perform_all_statistical_tests(alpha)
        
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
        
        # Advanced statistical tests
        if include_advanced_tests and advanced_tests:
            report.append("5. ADVANCED STATISTICAL TESTS")
            report.append("   " + "-" * 40)
            
            for i, pred in enumerate(self.predictions):
                pred_key = f'pred_{i}'
                report.append(f"   Prediction {i+1} ({pred:.4f}):")
                
                # Kolmogorov-Smirnov test
                ks_result = advanced_tests['kolmogorov_smirnov'][pred_key]
                report.append(f"   - Kolmogorov-Smirnov test:")
                report.append(f"     * KS statistic: {ks_result['ks_statistic']:.6f}")
                report.append(f"     * P-value: {ks_result['p_value']:.6f}")
                report.append(f"     * Effect size (Cohen's d): {ks_result['cohens_d']:.4f} ({ks_result['effect_size']})")
                
                # Mann-Whitney U test
                mw_result = advanced_tests['mann_whitney_u'][pred_key]
                report.append(f"   - Mann-Whitney U test:")
                report.append(f"     * U statistic: {mw_result['u_statistic']:.2f}")
                report.append(f"     * P-value: {mw_result['p_value']:.6f}")
                report.append(f"     * Effect size (r): {mw_result['effect_size_r']:.4f} ({mw_result['effect_size']})")
                report.append(f"     * Observations tend larger: {mw_result['observations_tend_larger']}")
                
                # Chi-square test
                chi2_result = advanced_tests['chi_square'][pred_key]
                report.append(f"   - Chi-square test:")
                if 'warning' in chi2_result:
                    report.append(f"     * Warning: {chi2_result['warning']}")
                else:
                    report.append(f"     * Chi-square statistic: {chi2_result['chi2_statistic']:.4f}")
                    report.append(f"     * P-value: {chi2_result['p_value']:.6f}")
                    report.append(f"     * Cramer's V: {chi2_result['cramers_v']:.4f} ({chi2_result['effect_size']})")
                    report.append(f"     * Degrees of freedom: {chi2_result['degrees_of_freedom']}")
                
                report.append("")
            
            # Summary of all tests
            summary = advanced_tests['summary']
            report.append("   SUMMARY OF ALL TESTS:")
            report.append(f"   - Total predictions analyzed: {summary['total_predictions']}")
            report.append(f"   - Total observations: {summary['total_observations']}")
            report.append(f"   - Tests performed: {', '.join(summary['tests_performed'])}")
            report.append("")
            
            for i, pred in enumerate(self.predictions):
                pred_key = f'pred_{i}'
                significant_tests = summary['significant_differences'][pred_key]
                report.append(f"   Prediction {i+1}:")
                report.append(f"   - Significant differences found in: {significant_tests if significant_tests else 'None'}")
                report.append(f"   - Recommendation: {summary['recommendations'][i]}")
                report.append("")
            
            report.append("6. INTERPRETATION")
        else:
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
                         figsize: Tuple[int, int] = (15, 10), include_advanced_tests: bool = False) -> Dict[str, Any]:
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
        include_advanced_tests : bool
            Whether to include advanced statistical tests (default: False)
        
        Returns:
        --------
        dict : Complete analysis results including all calculations and plots
        """
        results = {
            'central_tendency': self.calculate_central_tendency(),
            'distribution_analysis': self.calculate_distribution_analysis(),
            'error_metrics': self.calculate_error_metrics(),
            'statistical_test': self.perform_statistical_test(alpha),
            'report': self.generate_report(alpha, include_advanced_tests)
        }
        
        if include_advanced_tests:
            results['advanced_statistical_tests'] = self.perform_all_statistical_tests(alpha)
        
        if create_plots:
            results['visualizations'] = self.create_visualizations(figsize)
            # Add single-prediction specific visualizations when applicable
            if len(self.predictions) == 1 and len(self.observations) > 1:
                results['single_prediction_visualizations'] = self.visualize_single_prediction(plot_type='all', backend='matplotlib')
        
        return results


def compare_prediction_observations(prediction_data: Union[float, List[float], pd.Series], 
                                  observation_data: Union[List[float], pd.Series, np.ndarray],
                                  prediction_name: str = "Prediction",
                                  observation_name: str = "Observations",
                                  units: str = "",
                                  alpha: float = 0.05,
                                  create_plots: bool = True,
                                  figsize: Tuple[int, int] = (15, 10),
                                  include_advanced_tests: bool = False) -> Dict[str, Any]:
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
    include_advanced_tests : bool
        Whether to include advanced statistical tests (default: False)
    
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
    
    return comparison.run_full_analysis(alpha=alpha, create_plots=create_plots, figsize=figsize, include_advanced_tests=include_advanced_tests)


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
    # results1 = compare_prediction_observations(
    #     prediction_data=predicted_price,
    #     observation_data=observed_prices,
    #     prediction_name="Predicted Price",
    #     observation_name="Actual Prices",
    #     units="USD", 
    #     include_advanced_tests=True,
    # )
    # print(results1['report'])
    # results1=PredictionObservationComparison(
    #     prediction_data=predicted_price,
    #     observation_data=observed_prices,
    #     prediction_name="Predicted Price",
    #     observation_name="Actual Prices",
    #     units="USD", 
    # )
    # breakpoint()

    # breakpoint()
    # Show plots
    # if 'visualizations' in results1:
    #     pass  
        # plt.show()

    # results1.visualize_single_prediction(plot_type='all', backend='matplotlib')
    # plt.show()
    # breakpoint()
    # results1.plot

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
        units="USD",
        include_advanced_tests=True,
    )
    
    print(results2['report'])
    
    # Show plots
    if 'visualizations' in results2:
        plt.show()

     
    
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
        units="%",
        include_advanced_tests=True,
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
