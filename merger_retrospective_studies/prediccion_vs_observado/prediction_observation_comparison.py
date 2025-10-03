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

    # --- Histogram with Prediction Marker ---
    def plot_histogram_with_prediction(self, observations: Union[List[float], np.ndarray], prediction: float, 
                                     bins: Union[int, str] = 'auto', density: bool = False, 
                                     include_stats: bool = True, figsize: Tuple[int, int] = (10, 6)):
        """
        Create histogram comparing one prediction with multiple observed values.
        
        This function creates a histogram of observed values and marks the prediction
        with a vertical line, helping visualize where the prediction falls relative
        to the distribution of actual observations.
        
        Parameters:
        -----------
        observations : array-like
            The observed values to create histogram from
        prediction : float
            The single prediction value to mark on the histogram
        bins : int or str, optional
            Number of bins or binning strategy (default: 'auto')
        density : bool, optional
            If True, plot density instead of count (default: False)
        include_stats : bool, optional
            If True, add statistical markers (mean, median) (default: True)
        figsize : tuple, optional
            Figure size (width, height) (default: (10, 6))
        
        Returns:
        --------
        tuple : (fig, ax) matplotlib figure and axis objects
        """
        observations = self._validate_inputs_single(observations, prediction)
        
        # Determine optimal number of bins if 'auto' is specified
        if bins == 'auto':
            # Use Freedman-Diaconis rule for optimal bin width
            q75, q25 = np.percentile(observations, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr * (len(observations) ** (-1/3))
            if bin_width > 0:
                bins = int((np.max(observations) - np.min(observations)) / bin_width)
                bins = max(10, min(30, bins))  # Keep between 10 and 30 bins
            else:
                bins = 20
        elif isinstance(bins, str):
            bins = 20  # fallback for other string values
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create histogram
        n, bin_edges, patches = ax.hist(observations, bins=bins, alpha=0.7, color='#3b82f6', 
                                       edgecolor='black', linewidth=0.5, density=density,
                                       label=f'{self.observation_name} (n={len(observations)})')
        
        # Mark the prediction with a vertical line
        ax.axvline(prediction, color='#ef4444', linestyle='--', linewidth=3, 
                  label=f'{self.prediction_name}: {prediction:.2f}', zorder=3)
        
        # Add statistical markers if requested
        if include_stats:
            obs_mean = np.mean(observations)
            obs_median = np.median(observations)
            
            # Mark mean and median
            ax.axvline(obs_mean, color='#10b981', linestyle='-', linewidth=2, alpha=0.8,
                      label=f'Observed Mean: {obs_mean:.2f}', zorder=2)
            ax.axvline(obs_median, color='#f59e0b', linestyle=':', linewidth=2, alpha=0.8,
                      label=f'Observed Median: {obs_median:.2f}', zorder=2)
        
        # Determine if prediction is within the observed range
        obs_min, obs_max = np.min(observations), np.max(observations)
        within_range = obs_min <= prediction <= obs_max
        
        # Add range information
        ax.axvline(obs_min, color='#6b7280', linestyle=':', linewidth=1, alpha=0.6,
                  label=f'Observed Range: [{obs_min:.2f}, {obs_max:.2f}]', zorder=1)
        ax.axvline(obs_max, color='#6b7280', linestyle=':', linewidth=1, alpha=0.6, zorder=1)
        
        # Calculate and display percentile rank
        percentile_rank = (np.sum(observations <= prediction) / len(observations)) * 100
        
        # Set labels and title
        y_label = 'Density' if density else 'Frequency'
        ax.set_xlabel(f'Value {self.units}'.strip(), fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # Create informative title
        range_status = "within" if within_range else "outside"
        title = f'Histogram: Observed Data vs Prediction\n' \
                f'Prediction is {range_status} observed range ({percentile_rank:.1f}th percentile)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        
        # Add text box with key statistics
        stats_text = f'Prediction: {prediction:.2f}\n'
        stats_text += f'Observed Mean: {obs_mean:.2f}\n'
        stats_text += f'Observed Std: {np.std(observations):.2f}\n'
        stats_text += f'Percentile Rank: {percentile_rank:.1f}th'
        
        # Position text box in upper right
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax

    def plot_histogram_with_prediction_plotly(self, observations: Union[List[float], np.ndarray], prediction: float,
                                            bins: Union[int, str] = 'auto', density: bool = False,
                                            include_stats: bool = True):
        """
        Create interactive histogram with Plotly.
        
        Parameters:
        -----------
        observations : array-like
            The observed values to create histogram from
        prediction : float
            The single prediction value to mark on the histogram
        bins : int or str, optional
            Number of bins or binning strategy (default: 'auto')
        density : bool, optional
            If True, plot density instead of count (default: False)
        include_stats : bool, optional
            If True, add statistical markers (mean, median) (default: True)
        
        Returns:
        --------
        plotly.graph_objects.Figure : Interactive histogram figure
        """
        observations = self._validate_inputs_single(observations, prediction)
        
        # Determine optimal number of bins
        if bins == 'auto':
            q75, q25 = np.percentile(observations, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr * (len(observations) ** (-1/3))
            if bin_width > 0:
                bins = int((np.max(observations) - np.min(observations)) / bin_width)
                bins = max(10, min(30, bins))
            else:
                bins = 20
        elif isinstance(bins, str):
            bins = 20
        
        fig = go.Figure()
        
        # Create histogram
        y_label = 'Density' if density else 'Frequency'
        fig.add_trace(go.Histogram(
            x=observations,
            nbinsx=bins,
            name=f'{self.observation_name} (n={len(observations)})',
            marker_color='#3b82f6',
            opacity=0.7,
            histnorm='density' if density else 'count',
            hovertemplate='<b>Value Range</b>: %{x}<br>' +
                         f'<b>{y_label}</b>: %{{y}}<br>' +
                         '<extra></extra>'
        ))
        
        # Mark the prediction
        fig.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="#ef4444",
            line_width=3,
            annotation_text=f"{self.prediction_name}: {prediction:.2f}",
            annotation_position="top"
        )
        
        # Add statistical markers if requested
        if include_stats:
            obs_mean = np.mean(observations)
            obs_median = np.median(observations)
            
            # Mark mean
            fig.add_vline(
                x=obs_mean,
                line_dash="solid",
                line_color="#10b981",
                line_width=2,
                annotation_text=f"Mean: {obs_mean:.2f}",
                annotation_position="bottom"
            )
            
            # Mark median
            fig.add_vline(
                x=obs_median,
                line_dash="dot",
                line_color="#f59e0b",
                line_width=2,
                annotation_text=f"Median: {obs_median:.2f}",
                annotation_position="bottom"
            )
        
        # Calculate percentile rank
        percentile_rank = (np.sum(observations <= prediction) / len(observations)) * 100
        within_range = np.min(observations) <= prediction <= np.max(observations)
        range_status = "within" if within_range else "outside"
        
        # Update layout
        y_label = 'Density' if density else 'Frequency'
        fig.update_layout(
            title=f'Histogram: Observed Data vs Prediction<br>'
                  f'<sub>Prediction is {range_status} observed range ({percentile_rank:.1f}th percentile)</sub>',
            xaxis_title=f'Value {self.units}'.strip(),
            yaxis_title=y_label,
            showlegend=True,
            hovermode='closest',
            height=600
        )
        
        return fig

    def plot_histogram_with_prediction_seaborn(self, observations: Union[List[float], np.ndarray], prediction: float,
                                             bins: Union[int, str] = 'auto', density: bool = False,
                                             include_stats: bool = True, figsize: Tuple[int, int] = (10, 6)):
        """
        Create histogram using Seaborn.
        
        Parameters:
        -----------
        observations : array-like
            The observed values to create histogram from
        prediction : float
            The single prediction value to mark on the histogram
        bins : int or str, optional
            Number of bins or binning strategy (default: 'auto')
        density : bool, optional
            If True, plot density instead of count (default: False)
        include_stats : bool, optional
            If True, add statistical markers (mean, median) (default: True)
        figsize : tuple, optional
            Figure size (width, height) (default: (10, 6))
        
        Returns:
        --------
        tuple : (fig, ax) matplotlib figure and axis objects
        """
        observations = self._validate_inputs_single(observations, prediction)
        
        # Determine optimal number of bins
        if bins == 'auto':
            q75, q25 = np.percentile(observations, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr * (len(observations) ** (-1/3))
            if bin_width > 0:
                bins = int((np.max(observations) - np.min(observations)) / bin_width)
                bins = max(10, min(30, bins))
            else:
                bins = 20
        elif isinstance(bins, str):
            bins = 20
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create histogram using seaborn
        sns.histplot(data=observations, bins=bins, kde=False, stat='density' if density else 'count',
                    alpha=0.7, color='#3b82f6', edgecolor='black', linewidth=0.5, ax=ax)
        
        # Mark the prediction
        ax.axvline(prediction, color='#ef4444', linestyle='--', linewidth=3,
                  label=f'{self.prediction_name}: {prediction:.2f}', zorder=3)
        
        # Add statistical markers if requested
        if include_stats:
            obs_mean = np.mean(observations)
            obs_median = np.median(observations)
            
            ax.axvline(obs_mean, color='#10b981', linestyle='-', linewidth=2, alpha=0.8,
                      label=f'Observed Mean: {obs_mean:.2f}', zorder=2)
            ax.axvline(obs_median, color='#f59e0b', linestyle=':', linewidth=2, alpha=0.8,
                      label=f'Observed Median: {obs_median:.2f}', zorder=2)
        
        # Calculate percentile rank and range status
        percentile_rank = (np.sum(observations <= prediction) / len(observations)) * 100
        within_range = np.min(observations) <= prediction <= np.max(observations)
        range_status = "within" if within_range else "outside"
        
        # Set labels and title
        y_label = 'Density' if density else 'Frequency'
        ax.set_xlabel(f'Value {self.units}'.strip(), fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'Histogram: Observed Data vs Prediction\n'
                    f'Prediction is {range_status} observed range ({percentile_rank:.1f}th percentile)',
                    fontsize=14, fontweight='bold')
        
        # Add legend and grid
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        
        plt.tight_layout()
        return fig, ax

    # --- Distribution Comparison for Multiple Predictions vs Observations ---
    def plot_distribution_comparison(self, observations: Union[List[float], np.ndarray], 
                                   predictions: Union[List[float], np.ndarray], 
                                   figsize: Tuple[int, int] = (20, 12), 
                                   save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create comprehensive comparison of two distributions using 6 different visualization methods.
        
        This function generates six subplots comparing observations and predictions:
        1. Overlapping Histograms
        2. Density Plots
        3. Box Plots Side-by-Side
        4. Violin Plots
        5. Q-Q Plot
        6. Empirical CDF (ECDF)
        
        Parameters:
        -----------
        observations : array-like
            Array of observed values (must have length > 1)
        predictions : array-like
            Array of predicted values (must have length > 1)
        figsize : tuple, optional
            Figure size (width, height) (default: (20, 12))
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object, or None if inputs invalid
        """
        # Convert to numpy arrays and validate
        observations = np.array(observations)
        predictions = np.array(predictions)
        
        # Validation
        if len(observations) <= 1 or len(predictions) <= 1:
            raise ValueError("Both observations and predictions must have length > 1")
        
        # Remove NaN values
        obs_valid = ~np.isnan(observations)
        pred_valid = ~np.isnan(predictions)
        
        if not np.any(obs_valid) or not np.any(pred_valid):
            raise ValueError("No valid values found after removing NaN values")
        
        observations = observations[obs_valid]
        predictions = predictions[pred_valid]
        
        # Create figure with 6 subplots (2 rows × 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Distribution Comparison: Observations vs Predictions", fontsize=16, y=1.02)
        
        # Color scheme
        obs_color = '#3b82f6'  # Blue
        pred_color = '#ef4444'  # Red
        
        # 1. Overlapping Histograms (subplot 1)
        ax1 = axes[0, 0]
        ax1.hist(observations, bins=30, alpha=0.6, color=obs_color, 
                label=f'{self.observation_name} (n={len(observations)})', density=True)
        ax1.hist(predictions, bins=30, alpha=0.6, color=pred_color, 
                label=f'{self.prediction_name} (n={len(predictions)})', density=True)
        ax1.set_xlabel(f'Value {self.units}'.strip())
        ax1.set_ylabel('Density')
        ax1.set_title('Overlapping Histograms')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Density Plots (subplot 2)
        ax2 = axes[0, 1]
        from scipy.stats import gaussian_kde
        
        # Calculate density for observations
        if len(observations) > 1:
            obs_density = gaussian_kde(observations)
            obs_x = np.linspace(observations.min(), observations.max(), 200)
            obs_y = obs_density(obs_x)
            ax2.plot(obs_x, obs_y, color=obs_color, linewidth=2, 
                    label=f'{self.observation_name} (n={len(observations)})')
        
        # Calculate density for predictions
        if len(predictions) > 1:
            pred_density = gaussian_kde(predictions)
            pred_x = np.linspace(predictions.min(), predictions.max(), 200)
            pred_y = pred_density(pred_x)
            ax2.plot(pred_x, pred_y, color=pred_color, linewidth=2, 
                    label=f'{self.prediction_name} (n={len(predictions)})')
        
        ax2.set_xlabel(f'Value {self.units}'.strip())
        ax2.set_ylabel('Density')
        ax2.set_title('Density Plots')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box Plots Side-by-Side (subplot 3)
        ax3 = axes[0, 2]
        box_data = [observations, predictions]
        box_labels = [f'{self.observation_name}\n(n={len(observations)})', 
                     f'{self.prediction_name}\n(n={len(predictions)})']
        
        bp = ax3.boxplot(box_data, positions=[1, 2], labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor(obs_color)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(pred_color)
        bp['boxes'][1].set_alpha(0.7)
        
        ax3.set_ylabel(f'Value {self.units}'.strip())
        ax3.set_title('Box Plots')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Violin Plots (subplot 4)
        ax4 = axes[1, 0]
        
        # Create data for violin plot
        violin_data = []
        violin_labels = []
        violin_colors = []
        
        for i, (data, label, color) in enumerate(zip([observations, predictions], 
                                                    [f'{self.observation_name}', f'{self.prediction_name}'],
                                                    [obs_color, pred_color])):
            violin_data.extend(data)
            violin_labels.extend([label] * len(data))
            violin_colors.extend([color] * len(data))
        
        # Create DataFrame for seaborn violin plot
        import pandas as pd
        df_violin = pd.DataFrame({
            'value': violin_data,
            'type': violin_labels
        })
        
        sns.violinplot(data=df_violin, x='type', y='value', ax=ax4, 
                      palette=[obs_color, pred_color])
        ax4.set_xlabel('')
        ax4.set_ylabel(f'Value {self.units}'.strip())
        ax4.set_title('Violin Plots')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Q-Q Plot (subplot 5)
        ax5 = axes[1, 1]
        
        # Sort both arrays
        sorted_obs = np.sort(observations)
        sorted_pred = np.sort(predictions)
        
        # Create quantile pairs by matching indices
        min_len = min(len(sorted_obs), len(sorted_pred))
        obs_quantiles = sorted_obs[:min_len]
        pred_quantiles = sorted_pred[:min_len]
        
        ax5.scatter(obs_quantiles, pred_quantiles, alpha=0.6, s=20, color='#6b7280')
        
        # Add diagonal reference line
        min_val = min(obs_quantiles.min(), pred_quantiles.min())
        max_val = max(obs_quantiles.max(), pred_quantiles.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)
        
        ax5.set_xlabel(f'{self.observation_name} Quantiles')
        ax5.set_ylabel(f'{self.prediction_name} Quantiles')
        ax5.set_title('Q-Q Plot')
        ax5.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(obs_quantiles) > 1:
            corr = np.corrcoef(obs_quantiles, pred_quantiles)[0, 1]
            ax5.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax5.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. Empirical CDF (subplot 6)
        ax6 = axes[1, 2]
        
        # Calculate ECDF for observations
        obs_sorted = np.sort(observations)
        obs_ecdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
        ax6.step(obs_sorted, obs_ecdf, where='post', color=obs_color, 
                linewidth=2, label=f'{self.observation_name} (n={len(observations)})')
        
        # Calculate ECDF for predictions
        pred_sorted = np.sort(predictions)
        pred_ecdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
        ax6.step(pred_sorted, pred_ecdf, where='post', color=pred_color, 
                linewidth=2, label=f'{self.prediction_name} (n={len(predictions)})')
        
        ax6.set_xlabel(f'Value {self.units}'.strip())
        ax6.set_ylabel('Cumulative Probability')
        ax6.set_title('Empirical CDF (ECDF)')
        ax6.set_ylim([0, 1])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add statistical information text box
        obs_mean, obs_std = np.mean(observations), np.std(observations)
        pred_mean, pred_std = np.mean(predictions), np.std(predictions)
        
        stats_text = f'{self.observation_name}:\n'
        stats_text += f'  n = {len(observations)}\n'
        stats_text += f'  μ = {obs_mean:.2f}\n'
        stats_text += f'  σ = {obs_std:.2f}\n\n'
        stats_text += f'{self.prediction_name}:\n'
        stats_text += f'  n = {len(predictions)}\n'
        stats_text += f'  μ = {pred_mean:.2f}\n'
        stats_text += f'  σ = {pred_std:.2f}'
        
        # Add KS test statistic
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_p = ks_2samp(observations, predictions)
            stats_text += f'\n\nKS test:\n'
            stats_text += f'  D = {ks_stat:.3f}\n'
            stats_text += f'  p = {ks_p:.3f}'
        except:
            pass
        
        ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
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
            if plot_type in ['histogram', 'all']:
                figures['histogram'] = self.plot_histogram_with_prediction(observations, prediction)
        elif backend == 'plotly':
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_scatter_with_reference_plotly(observations, prediction, observation_ids)
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_boxplot_with_prediction_plotly(observations, prediction)
            if plot_type in ['residual', 'all']:
                figures['residual'] = self.plot_residuals_plotly(observations, prediction, observation_ids)
            if plot_type in ['histogram', 'all']:
                figures['histogram'] = self.plot_histogram_with_prediction_plotly(observations, prediction)
        elif backend == 'seaborn':
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_scatter_with_reference_seaborn(observations, prediction, observation_ids)
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_boxplot_with_prediction_seaborn(observations, prediction)
            if plot_type in ['residual', 'all']:
                figures['residual'] = self.plot_residuals(observations, prediction, observation_ids)
            if plot_type in ['histogram', 'all']:
                figures['histogram'] = self.plot_histogram_with_prediction_seaborn(observations, prediction)
        return figures

    def visualize_multiple_prediction(self, plot_type: str = 'all', backend: str = 'matplotlib') -> Dict[str, Any]:
        """
        Generate visualizations for multiple predictions vs multiple observations.
        Only enabled when both predictions and observations have more than one value.
        
        Parameters:
        -----------
        plot_type : str, optional
            Type of plots to generate ('all', 'distribution', 'scatter', 'boxplot', 'histogram', 'qq', 'ecdf')
        backend : str, optional
            Backend to use ('matplotlib', 'plotly', 'seaborn')
        
        Returns:
        --------
        dict : Dictionary containing generated figures
        """
        if not (len(self.predictions) > 1 and len(self.observations) > 1):
            return {}
        
        figures: Dict[str, Any] = {}
        
        if backend == 'matplotlib':
            if plot_type in ['distribution', 'all']:
                figures['distribution_comparison'] = self.plot_distribution_comparison(
                    self.observations, self.predictions
                )
            
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_multiple_scatter()
            
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_multiple_boxplot()
            
            if plot_type in ['histogram', 'all']:
                figures['histogram'] = self.plot_multiple_histogram()
            
            if plot_type in ['qq', 'all']:
                figures['qq_plot'] = self.plot_qq_comparison()
            
            if plot_type in ['ecdf', 'all']:
                figures['ecdf'] = self.plot_ecdf_comparison()
        
        elif backend == 'plotly':
            if plot_type in ['distribution', 'all']:
                figures['distribution_comparison'] = self.plot_distribution_comparison_plotly()
            
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_multiple_scatter_plotly()
            
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_multiple_boxplot_plotly()
        
        elif backend == 'seaborn':
            if plot_type in ['distribution', 'all']:
                figures['distribution_comparison'] = self.plot_distribution_comparison(
                    self.observations, self.predictions
                )
            
            if plot_type in ['scatter', 'all']:
                figures['scatter'] = self.plot_multiple_scatter_seaborn()
            
            if plot_type in ['boxplot', 'all']:
                figures['boxplot'] = self.plot_multiple_boxplot_seaborn()
        
        return figures

    def plot_multiple_scatter(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create scatter plot for multiple predictions vs observations."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot observations
        ax.scatter(range(len(self.observations)), self.observations, 
                  alpha=0.6, s=50, color='#3b82f6', label=f'{self.observation_name} (n={len(self.observations)})')
        
        # Plot predictions
        ax.scatter(range(len(self.predictions)), self.predictions, 
                  alpha=0.8, s=80, color='#ef4444', marker='s', 
                  label=f'{self.prediction_name} (n={len(self.predictions)})')
        
        ax.set_xlabel('Index')
        ax.set_ylabel(f'Value {self.units}'.strip())
        ax.set_title('Scatter Plot: Observations vs Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def plot_multiple_boxplot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create box plot for multiple predictions vs observations."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        box_data = [self.observations, self.predictions]
        box_labels = [f'{self.observation_name}\n(n={len(self.observations)})', 
                     f'{self.prediction_name}\n(n={len(self.predictions)})']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3b82f6')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#ef4444')
        bp['boxes'][1].set_alpha(0.7)
        
        ax.set_ylabel(f'Value {self.units}'.strip())
        ax.set_title('Box Plot: Observations vs Predictions')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax

    def plot_multiple_histogram(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create overlapping histograms for multiple predictions vs observations."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.hist(self.observations, bins=30, alpha=0.6, color='#3b82f6', 
                label=f'{self.observation_name} (n={len(self.observations)})', density=True)
        ax.hist(self.predictions, bins=30, alpha=0.6, color='#ef4444', 
                label=f'{self.prediction_name} (n={len(self.predictions)})', density=True)
        
        ax.set_xlabel(f'Value {self.units}'.strip())
        ax.set_ylabel('Density')
        ax.set_title('Overlapping Histograms: Observations vs Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def plot_qq_comparison(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create Q-Q plot for multiple predictions vs observations."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort both arrays
        sorted_obs = np.sort(self.observations)
        sorted_pred = np.sort(self.predictions)
        
        # Create quantile pairs by matching indices
        min_len = min(len(sorted_obs), len(sorted_pred))
        obs_quantiles = sorted_obs[:min_len]
        pred_quantiles = sorted_pred[:min_len]
        
        ax.scatter(obs_quantiles, pred_quantiles, alpha=0.6, s=20, color='#6b7280')
        
        # Add diagonal reference line
        min_val = min(obs_quantiles.min(), pred_quantiles.min())
        max_val = max(obs_quantiles.max(), pred_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)
        
        ax.set_xlabel(f'{self.observation_name} Quantiles')
        ax.set_ylabel(f'{self.prediction_name} Quantiles')
        ax.set_title('Q-Q Plot: Observations vs Predictions')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(obs_quantiles) > 1:
            corr = np.corrcoef(obs_quantiles, pred_quantiles)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax

    def plot_ecdf_comparison(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create ECDF plot for multiple predictions vs observations."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate ECDF for observations
        obs_sorted = np.sort(self.observations)
        obs_ecdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
        ax.step(obs_sorted, obs_ecdf, where='post', color='#3b82f6', 
                linewidth=2, label=f'{self.observation_name} (n={len(self.observations)})')
        
        # Calculate ECDF for predictions
        pred_sorted = np.sort(self.predictions)
        pred_ecdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
        ax.step(pred_sorted, pred_ecdf, where='post', color='#ef4444', 
                linewidth=2, label=f'{self.prediction_name} (n={len(self.predictions)})')
        
        ax.set_xlabel(f'Value {self.units}'.strip())
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Empirical CDF: Observations vs Predictions')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def plot_multiple_scatter_plotly(self):
        """Create interactive scatter plot with Plotly."""
        fig = go.Figure()
        
        # Add observations
        fig.add_trace(go.Scatter(
            x=list(range(len(self.observations))),
            y=self.observations,
            mode='markers',
            name=f'{self.observation_name} (n={len(self.observations)})',
            marker=dict(size=8, color='#3b82f6'),
            hovertemplate='<b>Observation</b><br>Index: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=list(range(len(self.predictions))),
            y=self.predictions,
            mode='markers',
            name=f'{self.prediction_name} (n={len(self.predictions)})',
            marker=dict(size=10, color='#ef4444', symbol='square'),
            hovertemplate='<b>Prediction</b><br>Index: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Scatter Plot: Observations vs Predictions',
            xaxis_title='Index',
            yaxis_title=f'Value {self.units}'.strip(),
            hovermode='closest',
            showlegend=True,
            height=600
        )
        
        return fig

    def plot_multiple_boxplot_plotly(self):
        """Create interactive box plot with Plotly."""
        fig = go.Figure()
        
        # Add observations box plot
        fig.add_trace(go.Box(
            y=self.observations,
            name=f'{self.observation_name} (n={len(self.observations)})',
            marker_color='#3b82f6',
            boxmean='sd'
        ))
        
        # Add predictions box plot
        fig.add_trace(go.Box(
            y=self.predictions,
            name=f'{self.prediction_name} (n={len(self.predictions)})',
            marker_color='#ef4444',
            boxmean='sd'
        ))
        
        fig.update_layout(
            title='Box Plot: Observations vs Predictions',
            yaxis_title=f'Value {self.units}'.strip(),
            showlegend=True,
            height=600
        )
        
        return fig

    def plot_distribution_comparison_plotly(self):
        """Create interactive distribution comparison with Plotly."""
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Overlapping Histograms', 'Density Plots', 'Box Plots',
                          'Violin Plots', 'Q-Q Plot', 'Empirical CDF'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Histograms
        fig.add_trace(go.Histogram(x=self.observations, name=f'{self.observation_name}', 
                                 opacity=0.6, marker_color='#3b82f6'), row=1, col=1)
        fig.add_trace(go.Histogram(x=self.predictions, name=f'{self.prediction_name}', 
                                 opacity=0.6, marker_color='#ef4444'), row=1, col=1)
        
        # 2. Density plots
        from scipy.stats import gaussian_kde
        obs_density = gaussian_kde(self.observations)
        pred_density = gaussian_kde(self.predictions)
        
        x_range = np.linspace(min(self.observations.min(), self.predictions.min()),
                             max(self.observations.max(), self.predictions.max()), 200)
        
        fig.add_trace(go.Scatter(x=x_range, y=obs_density(x_range), 
                               name=f'{self.observation_name} Density', 
                               line=dict(color='#3b82f6', width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_range, y=pred_density(x_range), 
                               name=f'{self.prediction_name} Density', 
                               line=dict(color='#ef4444', width=2)), row=1, col=2)
        
        # 3. Box plots
        fig.add_trace(go.Box(y=self.observations, name=f'{self.observation_name}', 
                            marker_color='#3b82f6'), row=1, col=3)
        fig.add_trace(go.Box(y=self.predictions, name=f'{self.prediction_name}', 
                            marker_color='#ef4444'), row=1, col=3)
        
        # 4. Violin plots
        fig.add_trace(go.Violin(y=self.observations, name=f'{self.observation_name}', 
                               marker_color='#3b82f6'), row=2, col=1)
        fig.add_trace(go.Violin(y=self.predictions, name=f'{self.prediction_name}', 
                               marker_color='#ef4444'), row=2, col=1)
        
        # 5. Q-Q plot
        sorted_obs = np.sort(self.observations)
        sorted_pred = np.sort(self.predictions)
        min_len = min(len(sorted_obs), len(sorted_pred))
        obs_quantiles = sorted_obs[:min_len]
        pred_quantiles = sorted_pred[:min_len]
        
        fig.add_trace(go.Scatter(x=obs_quantiles, y=pred_quantiles, 
                               mode='markers', name='Q-Q Points',
                               marker=dict(color='#6b7280', size=4)), row=2, col=2)
        
        # Add diagonal line
        min_val = min(obs_quantiles.min(), pred_quantiles.min())
        max_val = max(obs_quantiles.max(), pred_quantiles.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                               mode='lines', name='Perfect Agreement',
                               line=dict(dash='dash', color='black', width=1)), row=2, col=2)
        
        # 6. ECDF
        obs_sorted = np.sort(self.observations)
        obs_ecdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
        pred_sorted = np.sort(self.predictions)
        pred_ecdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
        
        fig.add_trace(go.Scatter(x=obs_sorted, y=obs_ecdf, mode='lines', 
                               name=f'{self.observation_name} ECDF',
                               line=dict(color='#3b82f6', width=2)), row=2, col=3)
        fig.add_trace(go.Scatter(x=pred_sorted, y=pred_ecdf, mode='lines', 
                               name=f'{self.prediction_name} ECDF',
                               line=dict(color='#ef4444', width=2)), row=2, col=3)
        
        fig.update_layout(
            title_text="Distribution Comparison: Observations vs Predictions",
            showlegend=True,
            height=800
        )
        
        return fig

    def plot_multiple_scatter_seaborn(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create scatter plot using Seaborn."""
        import pandas as pd
        
        # Create DataFrame
        data = []
        for i, val in enumerate(self.observations):
            data.append({'index': i, 'value': val, 'type': self.observation_name})
        for i, val in enumerate(self.predictions):
            data.append({'index': i, 'value': val, 'type': self.prediction_name})
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x='index', y='value', hue='type', s=50, ax=ax)
        
        ax.set_xlabel('Index')
        ax.set_ylabel(f'Value {self.units}'.strip())
        ax.set_title('Scatter Plot: Observations vs Predictions')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def plot_multiple_boxplot_seaborn(self) -> Tuple[plt.Figure, plt.Axes]:
        """Create box plot using Seaborn."""
        import pandas as pd
        
        # Create DataFrame
        data = []
        for val in self.observations:
            data.append({'value': val, 'type': self.observation_name})
        for val in self.predictions:
            data.append({'value': val, 'type': self.prediction_name})
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df, x='type', y='value', ax=ax, palette=['#3b82f6', '#ef4444'])
        
        ax.set_xlabel('')
        ax.set_ylabel(f'Value {self.units}'.strip())
        ax.set_title('Box Plot: Observations vs Predictions')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax

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
            # Use appropriate visualization method based on data structure
            if len(self.predictions) == 1 and len(self.observations) > 1:
                # Single prediction vs multiple observations
                results['single_prediction_visualizations'] = self.visualize_single_prediction(plot_type='all', backend='matplotlib')
            elif len(self.predictions) > 1 and len(self.observations) > 1:
                # Multiple predictions vs multiple observations
                results['multiple_prediction_visualizations'] = self.visualize_multiple_prediction(plot_type='all', backend='matplotlib')
            else:
                # Fallback for other cases - create basic visualizations
                results['basic_visualizations'] = self._create_basic_visualizations(figsize)
        
        return results

    def _create_basic_visualizations(self, figsize: Tuple[int, int] = (15, 10)) -> Dict[str, Any]:
        """Create basic visualizations for edge cases."""
        figures = {}
        
        # Simple scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(self.observations)), self.observations, 
                  alpha=0.6, label=f'{self.observation_name} (n={len(self.observations)})')
        
        if len(self.predictions) > 0:
            ax.scatter(range(len(self.predictions)), self.predictions, 
                      alpha=0.8, s=80, color='red', marker='s', 
                      label=f'{self.prediction_name} (n={len(self.predictions)})')
        
        ax.set_xlabel('Index')
        ax.set_ylabel(f'Value {self.units}'.strip())
        ax.set_title('Basic Comparison: Observations vs Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['basic_scatter'] = (fig, ax)
        
        return figures


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
    # # print(results1['report'])
    # results1=PredictionObservationComparison(
    #     prediction_data=predicted_price,
    #     observation_data=observed_prices,
    #     prediction_name="Predicted Price",
    #     observation_name="Actual Prices",
    #     units="USD", 
    # )
    # # breakpoint()

    # # breakpoint()
    # # Show plots
    # # if 'visualizations' in results1:
    # #     pass  
    #     # plt.show()

    # results1.visualize_single_prediction(plot_type='all', backend='matplotlib')
    # plt.show()
    # # breakpoint()
    # # results1.plot

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
