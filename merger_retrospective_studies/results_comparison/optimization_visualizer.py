"""
Optimization Results Visualization Module

This module provides comprehensive visualization tools for optimization results,
including individual plots and a combined dashboard for analyzing convergence,
parameter distributions, and solution similarities.

Author: Generated for merger retrospective studies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
import os
from typing import Union, Tuple, List, Dict, Any, Optional
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


class OptimizationVisualizer:
    """
    A comprehensive visualization tool for optimization results analysis.
    
    This class provides methods to visualize objective function values,
    convergence metrics, parameter distributions, and solution similarities
    through various plot types including bar charts, scatter plots, and heatmaps.
    """
    
    def __init__(self, data: Union[np.ndarray, pd.DataFrame], parameter_start_col: int = 5):
        """
        Initialize visualizer with optimization results.
        
        Parameters:
        -----------
        data : numpy.ndarray or pandas.DataFrame
            Optimization results with columns as described:
            - Column 0: row_index
            - Column 1: objective
            - Column 2: projected_gradient_norm
            - Column 3: min_reduced_hessian
            - Column 4: max_reduced_hessian
            - Columns 5+: parameter values (sigma_*, pi_prices_*, pi_tar_*)
        parameter_start_col : int
            Column index where parameter values start (default: 5)
        """
        # Store original data and column names
        self.original_data = data
        self.parameter_start_col = parameter_start_col
        
        # Convert to numpy array if needed and store column names
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.column_names = data.columns.tolist()
            self.parameter_names = data.columns[parameter_start_col:-1].tolist()
        else:
            self.data = data.copy()
            self.column_names = None
            self.parameter_names = None
        
        self.n_solutions, self.n_features = self.data.shape
        self.n_parameters = self.n_features - parameter_start_col - 1  # -1 for the last column
        
        # Extract key columns
        self.row_indices = self.data[:, 0].astype(int)
        self.objectives = self.data[:, 1]
        self.gradient_norms = self.data[:, 2]
        self.min_hessian = self.data[:, 3]
        self.max_hessian = self.data[:, 4]
        self.parameters = self.data[:, parameter_start_col:-1]
        
        # Validate data
        self._validate_data()
        
        # Calculate pairwise distances (lazy loading)
        self._distance_cache = {}
        
    def _validate_data(self):
        """Validate input data for common issues."""
        if self.n_solutions < 1:
            raise ValueError("Data must contain at least one solution")
        
        if self.n_parameters < 1:
            raise ValueError("No parameter columns found")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            warnings.warn("Data contains NaN or Inf values")
        
        # Check for negative gradient norms
        if np.any(self.gradient_norms < 0):
            warnings.warn("Some gradient norms are negative")
    
    def _get_color_palette(self, n: int) -> np.ndarray:
        """Return list of n distinct colors."""
        if n <= 10:
            return plt.cm.tab10(np.linspace(0, 1, n))
        elif n <= 20:
            return plt.cm.tab20(np.linspace(0, 1, n))
        else:
            return plt.cm.gist_rainbow(np.linspace(0, 1, n))
    
    def plot_objective_function(self, figsize: Tuple[int, int] = (10, 6), 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot objective function values as bar chart.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar chart
        colors = self._get_color_palette(self.n_solutions)
        bars = ax.bar(range(self.n_solutions), self.objectives, color=colors)
        
        # Find best and worst solutions
        best_idx = np.argmin(self.objectives)
        worst_idx = np.argmax(self.objectives)
        
        # Highlight best and worst
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
        bars[worst_idx].set_edgecolor('red')
        bars[worst_idx].set_linewidth(3)
        
        # Add annotations
        ax.annotate(f'Best: {self.objectives[best_idx]:.6f}', 
                   xy=(best_idx, self.objectives[best_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Worst: {self.objectives[worst_idx]:.6f}', 
                   xy=(worst_idx, self.objectives[worst_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Solution Index')
        ax.set_ylabel('Objective Value')
        ax.set_title('Objective Function Values', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(self.n_solutions))
        ax.set_xticklabels([f'Row {i}' for i in self.row_indices])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_gradient_norm(self, figsize: Tuple[int, int] = (10, 6), 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot projected gradient norm on log scale.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar chart on log scale
        colors = self._get_color_palette(self.n_solutions)
        bars = ax.bar(range(self.n_solutions), self.gradient_norms, color=colors)
        
        # Set log scale
        ax.set_yscale('log')
        
        # Add convergence threshold line
        threshold = 1e-6
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Convergence threshold ({threshold})')
        
        # Find best converged solution
        converged_mask = self.gradient_norms <= threshold
        if np.any(converged_mask):
            best_converged_idx = np.argmin(self.gradient_norms[converged_mask])
            converged_indices = np.where(converged_mask)[0]
            actual_best_idx = converged_indices[best_converged_idx]
            
            # Highlight best converged solution
            bars[actual_best_idx].set_edgecolor('green')
            bars[actual_best_idx].set_linewidth(3)
            
            ax.annotate(f'Best converged: {self.gradient_norms[actual_best_idx]:.2e}', 
                       xy=(actual_best_idx, self.gradient_norms[actual_best_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Solution Index')
        ax.set_ylabel('Gradient Norm (Log Scale)')
        ax.set_title('Projected Gradient Norm (Log Scale)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(self.n_solutions))
        ax.set_xticklabels([f'Row {i}' for i in self.row_indices])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_log_scatter(self, 
                        figsize: Tuple[int, int] = (10, 6), 
                        show_annotations: bool = True, 
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a log-scale scatter plot of objective vs gradient norm.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        show_annotations : bool
            Whether to annotate best and worst points
        show_stats : bool
            Whether to print summary statistics
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with log scale on y-axis
        scatter = ax.scatter(self.objectives, 
                           self.gradient_norms,
                           c=range(len(self.objectives)),  # Color by iteration order
                           cmap='viridis',
                           s=50,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Labels and title
        ax.set_xlabel('Objective Value', fontsize=12)
        ax.set_ylabel('Projected Gradient Norm (log scale)', fontsize=12)
        ax.set_title('Objective vs Gradient Norm (Log Scale)', fontsize=14, fontweight='bold')
        
        # Add colorbar to show iteration progression
        # cbar = plt.colorbar(scatter, ax=ax)
        # cbar.set_label('Solution Index', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Annotate best and worst points
        if show_annotations:
            best_idx = np.argmin(self.objectives)
            worst_idx = np.argmax(self.objectives)
            
            ax.annotate(f'Best: {self.objectives[best_idx]:.4f}',
                        xy=(self.objectives[best_idx], self.gradient_norms[best_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.annotate(f'Worst: {self.objectives[worst_idx]:.2f}',
                        xy=(self.objectives[worst_idx], self.gradient_norms[worst_idx]),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
               
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def _plot_log_scatter_ax(self, ax):
        """Helper method to plot log scatter on given axis."""
        # Create scatter plot with log scale on y-axis
        scatter = ax.scatter(self.objectives, 
                           self.gradient_norms,
                           c=range(len(self.objectives)),  # Color by iteration order
                           cmap='viridis',
                           s=50,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Labels and title
        ax.set_xlabel('Objective Value', fontsize=12)
        ax.set_ylabel('Projected Gradient Norm (log scale)', fontsize=12)
        ax.set_title('Objective vs Gradient Norm (Log Scale)', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Annotate best and worst points
        best_idx = np.argmin(self.objectives)
        worst_idx = np.argmax(self.objectives)
        
        ax.annotate(f'Best: {self.objectives[best_idx]:.4f}',
                    xy=(self.objectives[best_idx], self.gradient_norms[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Worst: {self.objectives[worst_idx]:.2f}',
                    xy=(self.objectives[worst_idx], self.gradient_norms[worst_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    def plot_hessian_eigenvalues(self, figsize: Tuple[int, int] = (10, 6), 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot min vs max Hessian eigenvalues as scatter plot.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Separate positive and negative min eigenvalues
        positive_mask = self.min_hessian > 0
        negative_mask = self.min_hessian <= 0
        
        # Plot points
        if np.any(positive_mask):
            ax.scatter(self.min_hessian[positive_mask], self.max_hessian[positive_mask], 
                      c='green', s=100, alpha=0.7, label='Local minima', zorder=3)
        
        if np.any(negative_mask):
            ax.scatter(self.min_hessian[negative_mask], self.max_hessian[negative_mask], 
                      c='red', s=100, alpha=0.7, label='Saddle points', zorder=3)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
        
        # Label each point with row index
        for i, (min_val, max_val) in enumerate(zip(self.min_hessian, self.max_hessian)):
            ax.annotate(f'{self.row_indices[i]}', (min_val, max_val), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Set log scale for y-axis
        ax.set_yscale('log')
        
        ax.set_xlabel('Min Eigenvalue')
        ax.set_ylabel('Max Eigenvalue (Log Scale)')
        ax.set_title('Reduced Hessian Eigenvalues', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _calculate_pairwise_distances(self, metric: str = 'euclidean') -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        """
        Calculate pairwise distances between all parameter vectors.
        
        Parameters:
        -----------
        metric : str
            Distance metric ('euclidean', 'manhattan', 'cosine')
            
        Returns:
        --------
        distance_matrix : numpy.ndarray
            N×N symmetric matrix of distances
        pairs : list of tuples
            List of (i, j, distance) for all pairs i < j
        """
        if metric in self._distance_cache:
            return self._distance_cache[metric]
        
        if metric == 'euclidean':
            distance_matrix = self._euclidean_distance_matrix(self.parameters)
        elif metric == 'manhattan':
            distance_matrix = self._manhattan_distance_matrix(self.parameters)
        elif metric == 'cosine':
            distance_matrix = self._cosine_similarity_matrix(self.parameters)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Create pairs list
        pairs = []
        for i in range(self.n_solutions):
            for j in range(i + 1, self.n_solutions):
                pairs.append((i, j, distance_matrix[i, j]))
        
        # Cache result
        self._distance_cache[metric] = (distance_matrix, pairs)
        
        return distance_matrix, pairs
    
    def _euclidean_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance matrix."""
        return euclidean_distances(vectors)
    
    def _manhattan_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Calculate Manhattan distance matrix."""
        return manhattan_distances(vectors)
    
    def _cosine_similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix."""
        return cosine_similarity(vectors)
    
    def plot_pairwise_distances(self, metric: str = 'euclidean', figsize: Tuple[int, int] = (10, 6), 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pairwise distances between parameter vectors as bar chart.
        
        Parameters:
        -----------
        metric : str
            Distance metric ('euclidean', 'manhattan', 'cosine')
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        _, pairs = self._calculate_pairwise_distances(metric)
        
        # Sort pairs by distance/similarity
        pairs.sort(key=lambda x: x[2], reverse=(metric == 'cosine'))
        
        # Take top 30 pairs
        top_pairs = pairs[:30]
        
        # Create labels and values
        labels = [f"{self.row_indices[i]}-{self.row_indices[j]}" for i, j, _ in top_pairs]
        values = [dist for _, _, dist in top_pairs]
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_pairs)))
        bars = ax.bar(range(len(top_pairs)), values, color=colors)
        
        # Add annotations for most/least similar pairs
        if metric == 'cosine':
            ax.annotate(f'Most similar: {values[0]:.3f}', 
                       xy=(0, values[0]), xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
            ax.annotate(f'Least similar: {values[-1]:.3f}', 
                       xy=(len(values)-1, values[-1]), xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral'))
        else:
            ax.annotate(f'Most similar: {values[0]:.3f}', 
                       xy=(0, values[0]), xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
            ax.annotate(f'Most different: {values[-1]:.3f}', 
                       xy=(len(values)-1, values[-1]), xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral'))
        
        ax.set_xlabel('Solution Pairs')
        ylabel = f'{metric.title()} Similarity' if metric == 'cosine' else f'{metric.title()} Distance'
        ax.set_ylabel(ylabel)
        ax.set_title(f'Pairwise {metric.title()} {"Similarity" if metric == "cosine" else "Distances"} (Top 30)', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distance_heatmap(self, metric: str = 'euclidean', figsize: Tuple[int, int] = (10, 8), 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pairwise distance heatmap.
        
        Parameters:
        -----------
        metric : str
            Distance metric ('euclidean', 'manhattan', 'cosine')
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        distance_matrix, _ = self._calculate_pairwise_distances(metric)
        
        # Set diagonal to NaN for better visualization
        np.fill_diagonal(distance_matrix, np.nan)
        
        # Choose colormap
        if metric == 'cosine':
            cmap = 'RdYlGn'  # Red-Yellow-Green for similarity
            vmin, vmax = -1, 1
        else:
            cmap = 'RdYlGn_r'  # Red-Yellow-Green reversed for distance
            vmin, vmax = None, None
        
        # Create heatmap
        im = ax.imshow(distance_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric.title()} {"Similarity" if metric == "cosine" else "Distance"}')
        
        # Add annotations
        for i in range(self.n_solutions):
            for j in range(self.n_solutions):
                if not np.isnan(distance_matrix[i, j]):
                    text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        # Set labels
        ax.set_xticks(range(self.n_solutions))
        ax.set_yticks(range(self.n_solutions))
        ax.set_xticklabels([str(i) for i in self.row_indices])
        ax.set_yticklabels([str(i) for i in self.row_indices])
        
        ax.set_xlabel('Solution Index')
        ax.set_ylabel('Solution Index')
        ax.set_title(f'{metric.title()} {"Similarity" if metric == "cosine" else "Distance"} Heatmap', 
                    fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(self, figsize: Tuple[int, int] = (20, 20), 
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive dashboard with all visualizations.
        
        Layout:
        - Row 1: Objective function, Gradient norm
        - Row 2: Log scatter, Euclidean distances (bar)
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create grid layout - 3 rows, 2 columns (last row for parameters)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Row 1: Objective function and Gradient norm
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_objective_function_ax(ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_gradient_norm_ax(ax2)
        
        # Row 2: Log scatter and Euclidean distances
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_log_scatter_ax(ax3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_distance_heatmap_ax(ax4, 'cosine')
        
        # Row 3: Parameters plot spanning both columns
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_parameters_ax(ax5)
        
        # Add overall title
        fig.suptitle('Optimization Results Analysis Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_objective_function_ax(self, ax):
        """Helper method to plot objective function on given axis."""
        # Create histogram with frequency counts
        n_bins = 10000   # Adaptive number of bins
        n_array, bins, patches = ax.hist(self.objectives, bins='auto', alpha=0.7, 
                                  edgecolor='black', linewidth=0.5, density=False)
        # breakpoint()
        # Define colors for each bin using viridis colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
        
        # Apply different color to each bin
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
        
        # Find best and worst solutions
        best_idx = np.argmin(self.objectives)
        worst_idx = np.argmax(self.objectives)
        best_value = self.objectives[best_idx]
        worst_value = self.objectives[worst_idx]
        
        # Restrict x-axis to 1st–99th percentile range
        p1, p95 = np.percentile(self.objectives, [1, 90])
        
        # Add vertical lines for best and worst
        ax.axvline(x=best_value, color='green', linestyle='--', linewidth=3, 
                  label=f'Best: {best_value:.6f}')
        ax.axvline(x=worst_value, color='red', linestyle='--', linewidth=3, 
                  label=f'Worst: {worst_value:.6f}')
        
        ax.set_xlabel('Objective Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Objective Function Values Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Set x-axis limits based on percentiles (1st–99th)
        ax.set_xlim(p1, p95)
    
    def _plot_gradient_norm_ax(self, ax):
        """Helper method to plot gradient norm on given axis."""
        # Filter out zero and negative values for log space calculation
        positive_data = self.gradient_norms[self.gradient_norms > 0]
        
        if len(positive_data) == 0:
            # If no positive values, use regular histogram
            n, bins, patches = ax.hist(self.gradient_norms, bins='auto', edgecolor='black', linewidth=0.5)
            colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
        else:
            # Create histogram with logarithmic bins (since data spans many orders of magnitude)
            # Use 25 bins with log spacing
            bins = np.logspace(np.log10(positive_data.min()), np.log10(self.gradient_norms.max()), 25)
            
            # Create histogram and get the patches (bars)
            n, bins, patches = ax.hist(self.gradient_norms, bins='auto', edgecolor='black', linewidth=0.5)
            
            # Define colors for each bin using viridis colormap
            colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
            
            # Apply different color to each bin
            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
        
        # Use log scale for x-axis due to wide range of values
        ax.set_xscale('log')
        
        # Add some statistics as text
        mean_val = np.mean(self.gradient_norms)
        median_val = np.median(self.gradient_norms)
        stats_text = f'Mean: {mean_val:.4e}\nMedian: {median_val:.4e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=10)
        
        # Customize the plot
        ax.set_xlabel('Projected Gradient Norm (log scale)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Projected Gradient Norms', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')



    def _plot_gradient_norm_ax__(self, ax):
        """Helper method to plot gradient norm on given axis."""
        # Create histogram with density=True for probability
        n_bins = 20  # Fixed number of bins
        n, bins, patches = ax.hist(self.gradient_norms, bins=n_bins, alpha=0.7, 
                                  edgecolor='black', density=True)
        
        # Color each bin differently
        colors = self._get_color_palette(n_bins)
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
        
        # Find best and worst solutions
        best_idx = np.argmin(self.gradient_norms)
        worst_idx = np.argmax(self.gradient_norms)
        best_value = self.gradient_norms[best_idx]
        worst_value = self.gradient_norms[worst_idx]
        
        # Calculate 90% range of observations
        sorted_gradients = np.sort(self.gradient_norms)
        n_90_percent = int(0.9 * len(sorted_gradients))
        start_idx = (len(sorted_gradients) - n_90_percent) // 2
        end_idx = start_idx + n_90_percent
        x_min_90 = sorted_gradients[start_idx]
        x_max_90 = sorted_gradients[end_idx - 1]
        
        # Add vertical lines for best and worst
        ax.axvline(x=best_value, color='green', linestyle='--', linewidth=3, 
                  label=f'Best: {best_value:.2e}')
        ax.axvline(x=worst_value, color='red', linestyle='--', linewidth=3, 
                  label=f'Worst: {worst_value:.2e}')
        
        # Add convergence threshold line
        threshold = 1e-6
        ax.axvline(x=threshold, color='orange', linestyle=':', linewidth=2, 
                  label=f'Threshold ({threshold})')
        
        ax.set_xlabel('Gradient Norm')
        ax.set_ylabel('Probability Density')
        ax.set_title('Projected Gradient Norm Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Set x-axis limits to show 90% of observations
        ax.set_xlim(left=x_min_90, right=x_max_90)
        # Set log scale for better visualization
        ax.set_xscale('log')
    
    def _plot_hessian_eigenvalues_ax(self, ax):
        """Helper method to plot Hessian eigenvalues on given axis."""
        positive_mask = self.min_hessian > 0
        negative_mask = self.min_hessian <= 0
        
        if np.any(positive_mask):
            ax.scatter(self.min_hessian[positive_mask], self.max_hessian[positive_mask], 
                      c='green', s=100, alpha=0.7, label='Local minima', zorder=3)
        
        if np.any(negative_mask):
            ax.scatter(self.min_hessian[negative_mask], self.max_hessian[negative_mask], 
                      c='red', s=100, alpha=0.7, label='Saddle points', zorder=3)
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=1)
        ax.set_yscale('log')
        ax.set_xlabel('Min Eigenvalue')
        ax.set_ylabel('Max Eigenvalue (Log Scale)')
        ax.set_title('Reduced Hessian Eigenvalues', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pairwise_distances_ax(self, ax, metric):
        """Helper method to plot pairwise distances on given axis."""
        _, pairs = self._calculate_pairwise_distances(metric)
        pairs.sort(key=lambda x: x[2], reverse=(metric == 'cosine'))
        top_pairs = pairs[:15]  # Fewer for dashboard
        
        labels = [f"{self.row_indices[i]}-{self.row_indices[j]}" for i, j, _ in top_pairs]
        values = [dist for _, _, dist in top_pairs]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_pairs)))
        ax.bar(range(len(top_pairs)), values, color=colors)
        
        ax.set_xlabel('Solution Pairs')
        ylabel = f'{metric.title()} Similarity' if metric == 'cosine' else f'{metric.title()} Distance'
        ax.set_ylabel(ylabel)
        ax.set_title(f'Pairwise {metric.title()} {"Similarity" if metric == "cosine" else "Distances"}', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    def _plot_distance_heatmap_ax(self, ax, metric):
        """Helper method to plot distance heatmap on given axis."""
        distance_matrix, _ = self._calculate_pairwise_distances(metric)

        # np.fill_diagonal(distance_matrix, np.nan)

        if metric == 'cosine':
            cmap = 'RdYlBu'
            vmin, vmax = -1, 1
        else:
            cmap = 'RdYlGn_r'
            vmin, vmax = None, None
        
        im = ax.imshow(distance_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric.title()} {"Similarity" if metric == "cosine" else "Distance"}')
        
        # Add annotations for smaller matrices
        if self.n_solutions <= 10:
            for i in range(self.n_solutions):
                for j in range(self.n_solutions):
                    if not np.isnan(distance_matrix[i, j]):
                        ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xticks(range(self.n_solutions))
        ax.set_yticks(range(self.n_solutions))
        # ax.set_xticklabels([str(i) for i in self.row_indices])
        # ax.set_yticklabels([str(i) for i in self.row_indices])
        ax.set_xlabel('Solution Index')
        ax.set_ylabel('Solution Index')
        ax.set_title(f'{metric.title()} {"Similarity" if metric == "cosine" else "Distance"} Heatmap', 
                    fontweight='bold')
    
    def generate_all_plots(self, output_dir: str = './optimization_plots') -> None:
        """
        Generate all individual plots and save to directory.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate individual plots
        self.plot_objective_function(save_path=os.path.join(output_dir, 'objective_function.png'))
        self.plot_gradient_norm(save_path=os.path.join(output_dir, 'gradient_norm.png'))
        self.plot_log_scatter(save_path=os.path.join(output_dir, 'log_scatter.png'))
        self.plot_hessian_eigenvalues(save_path=os.path.join(output_dir, 'hessian_eigenvalues.png'))
        self.plot_parameters(save_path=os.path.join(output_dir, 'parameters.png'))
        
        # Distance plots
        for metric in ['euclidean', 'manhattan', 'cosine']:
            self.plot_pairwise_distances(metric=metric, 
                                       save_path=os.path.join(output_dir, f'pairwise_{metric}.png'))
            self.plot_distance_heatmap(metric=metric, 
                                     save_path=os.path.join(output_dir, f'heatmap_{metric}.png'))
        
        # Dashboard
        self.create_dashboard(save_path=os.path.join(output_dir, 'optimization_dashboard.png'))
        
        print(f"All plots saved to {output_dir}")
    
    def get_parameter_names(self) -> List[str]:
        """
        Return list of parameter names.
        
        Returns:
        --------
        list
            List of parameter names, or None if not available
        """
        return self.parameter_names
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Return information about parameters.
        
        Returns:
        --------
        dict
            Dictionary containing parameter information
        """
        return {
            'parameter_names': self.parameter_names,
            'n_parameters': self.n_parameters,
            'parameter_start_col': self.parameter_start_col
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Return dictionary with key statistics.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - best_objective: (row, value)
            - best_convergence: (row, gradient norm)
            - local_minima: list of rows with positive min Hessian
            - most_similar_pair: (rows, distance/similarity)
            - most_dissimilar_pair: (rows, distance/similarity)
        """
        # Best objective
        best_obj_idx = np.argmin(self.objectives)
        best_objective = (self.row_indices[best_obj_idx], self.objectives[best_obj_idx])
        
        # Best convergence
        best_conv_idx = np.argmin(self.gradient_norms)
        best_convergence = (self.row_indices[best_conv_idx], self.gradient_norms[best_conv_idx])
        
        # Local minima
        local_minima = self.row_indices[self.min_hessian > 0].tolist()
        
        # Most similar/dissimilar pairs
        _, pairs = self._calculate_pairwise_distances('euclidean')
        pairs.sort(key=lambda x: x[2])
        most_similar = (self.row_indices[pairs[0][0]], self.row_indices[pairs[0][1]], pairs[0][2])
        most_dissimilar = (self.row_indices[pairs[-1][0]], self.row_indices[pairs[-1][1]], pairs[-1][2])
        
        return {
            'best_objective': best_objective,
            'best_convergence': best_convergence,
            'local_minima': local_minima,
            'most_similar_pair': most_similar,
            'most_dissimilar_pair': most_dissimilar,
            'n_solutions': self.n_solutions,
            'n_parameters': self.n_parameters,
            'converged_solutions': np.sum(self.gradient_norms <= 1e-6),
            'parameter_names': self.parameter_names
        }
    
    def plot_vectors(self, vectors, labels=None, title="Vector Plot", xlabel="Index", ylabel="Value"):
        """
        Plot n vectors where each vector has t values.
        
        Parameters:
        -----------
        vectors : list or array-like
            List of n vectors, where each vector contains t values.
            Can be a list of lists, list of arrays, or 2D numpy array.
        labels : list, optional
            List of labels for each vector. If None, vectors are labeled as "Vector 1", "Vector 2", etc.
        title : str, optional
            Title of the plot (default: "Vector Plot")
        xlabel : str, optional
            Label for x-axis (default: "Index")
        ylabel : str, optional
            Label for y-axis (default: "Value")
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes objects
        
        Example:
        --------
        >>> vec1 = [1, 2, 3, 4, 5]
        >>> vec2 = [2, 4, 3, 5, 6]
        >>> vec3 = [1, 3, 2, 4, 5]
        >>> plot_vectors([vec1, vec2, vec3], labels=['A', 'B', 'C'])
        """
        # Convert to numpy array for easier handling
        vectors = np.array(vectors)
        
        # Get dimensions
        n = len(vectors)  # number of vectors
        t = len(vectors[0])  # number of values per vector
        
        # Create x-axis values
        x = np.arange(t)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each vector
        for i, vector in enumerate(vectors):
            if labels is not None and i < len(labels):
                label = labels[i]
            else:
                label = f"Vector {i+1}"
            
            ax.plot(x, vector, marker='o', label=label, linewidth=2)
        
        # Customize the plot
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax

    def plot_parameters(self, figsize: Tuple[int, int] = (10, 6), 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot parameter vectors for all solutions.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Create labels for each solution
        labels = [f'Row {i}' for i in self.row_indices]
        
        # Create the plot manually to use parameter names
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get dimensions
        n = len(self.parameters)  # number of vectors
        t = len(self.parameters[0])  # number of values per vector
        
        # Create x-axis values - use parameter names if available
        if self.parameter_names is not None:
            x = np.arange(t)
            x_labels = self.parameter_names
        else:
            x = np.arange(t)
            x_labels = [f'Param {i+1}' for i in range(t)]
        
        # Plot each vector
        for i, vector in enumerate(self.parameters):
            if i < len(labels):
                label = labels[i]
            else:
                label = f"Vector {i+1}"
            
            ax.plot(x, vector, marker='o', label=label, linewidth=2, markersize=4)
        
        # Customize the plot
        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title('Parameter Vectors Across Solutions', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels to parameter names
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def _plot_parameters_ax(self, ax):
        """Helper method to plot parameters on given axis."""
        # Select a subset with the smallest objective values
        sorted_indices = np.argsort(self.objectives)
        n_best = max(1, min(15, int(0.25 * self.n_solutions)))
        best_indices = sorted_indices[:n_best]
        labels = [f'Row {self.row_indices[i]}' for i in best_indices]
        
        # Get dimensions
        n = len(self.parameters)  # number of vectors
        t = len(self.parameters[0])  # number of values per vector
        
        # Create x-axis values - use parameter names if available
        if self.parameter_names is not None:
            x = np.arange(t)
            x_labels = self.parameter_names
        else:
            x = np.arange(t)
            x_labels = [f'Param {i+1}' for i in range(t)]
        
        # Plot only the selected vectors (lowest objectives)
        for idx, label in zip(best_indices, labels):
            vector = self.parameters[idx]
            ax.plot(x, vector, marker='o', label=label, linewidth=2, markersize=3)
        
        # Customize the plot
        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title('Parameter Vectors Across Solutions', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels to parameter names
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

    def print_summary(self) -> None:
        """Print formatted summary statistics to console."""
        stats = self.get_summary_statistics()
        
        print("=" * 60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Number of solutions: {stats['n_solutions']}")
        print(f"Number of parameters: {stats['n_parameters']}")
        print(f"Converged solutions (gradient < 1e-6): {stats['converged_solutions']}")
        
        # Print parameter names if available
        if stats['parameter_names'] is not None:
            print(f"Parameter names: {stats['parameter_names']}")
        else:
            print("Parameter names: Not available (data from numpy array)")
        
        print()
        print(f"Best objective: Row {stats['best_objective'][0]} = {stats['best_objective'][1]:.6f}")
        print(f"Best convergence: Row {stats['best_convergence'][0]} = {stats['best_convergence'][1]:.2e}")
        print(f"Local minima (positive Hessian): {stats['local_minima']}")
        print()
        print(f"Most similar pair: Rows {stats['most_similar_pair'][0]}-{stats['most_similar_pair'][1]} "
              f"(distance: {stats['most_similar_pair'][2]:.6f})")
        print(f"Most dissimilar pair: Rows {stats['most_dissimilar_pair'][0]}-{stats['most_dissimilar_pair'][1]} "
              f"(distance: {stats['most_dissimilar_pair'][2]:.6f})")
        print("=" * 60)


if __name__ == "__main__":
    # Read the CSV file as a pandas DataFrame
    df = pd.read_csv(r"C:\Users\Andres.DESKTOP-D77KM25\Downloads\resultados_opt_compilados.csv")
    print("DataFrame loaded. Shape:", df.shape)
    print(df.loc[:, 'objective' :'beta_se_prices'].head())
    print(df.loc[:, 'sigma_1_1' :'beta_se_prices'].shape)
    analisis=OptimizationVisualizer(df.loc[:, 'objective' :'beta_se_prices'])
    analisis.create_dashboard(save_path='dashboard.png')
    # Save the 'projected_gradient_norm' column to a text file
    df['projected_gradient_norm'].to_csv('projected_gradient_norm.txt', index=False, header=True)
    df[['objective', 'projected_gradient_norm']].to_csv('objective_projected_gradient_norm.txt', index=False, header=True)
    analisis.plot_log_scatter(save_path='log_scatter.png')
    print(analisis.parameters[0])
    print(analisis.n_parameters)
    
    print(euclidean_distances(analisis.parameters, analisis.parameters))
    print(cosine_similarity(analisis.parameters, analisis.parameters))