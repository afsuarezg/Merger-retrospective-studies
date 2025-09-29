"""
Vector Comparison Module for Optimization Results

This module provides comprehensive tools for comparing one-dimensional vectors
resulting from different optimization algorithm runs, including optimization
metrics analysis and visualization capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
import json
from pathlib import Path


class VectorComparator:
    """
    A comprehensive class for comparing optimization result vectors.
    
    This class provides methods to analyze, compare, and visualize multiple
    one-dimensional vectors along with their associated optimization metrics.
    """
    
    def __init__(
        self,
        vectors: List[Union[np.ndarray, List[float]]],
        labels: Optional[List[str]] = None,
        objective_values: Optional[List[float]] = None,
        gradient_norms: Optional[List[float]] = None,
        hessian_min_eigenvalues: Optional[List[float]] = None,
        hessian_max_eigenvalues: Optional[List[float]] = None,
        tolerance: float = 1e-6
    ):
        """
        Initialize the VectorComparator with vectors and optimization metrics.
        
        Parameters
        ----------
        vectors : List[Union[np.ndarray, List[float]]]
            List of 1D vectors to compare. All vectors must have the same length.
        labels : Optional[List[str]], default=None
            Labels for each vector. If None, will use 'Vector_0', 'Vector_1', etc.
        objective_values : Optional[List[float]], default=None
            Objective function values for each vector.
        gradient_norms : Optional[List[float]], default=None
            Projected gradient norms for each vector.
        hessian_min_eigenvalues : Optional[List[float]], default=None
            Minimum eigenvalues of Hessian for each vector.
        hessian_max_eigenvalues : Optional[List[float]], default=None
            Maximum eigenvalues of Hessian for each vector.
        tolerance : float, default=1e-6
            Tolerance for numerical comparisons.
        
        Raises
        ------
        ValueError
            If vectors have different lengths, are empty, or have invalid dimensions.
        """
        # Validate inputs
        if not vectors:
            raise ValueError("At least one vector must be provided")
        
        # Convert to numpy arrays and validate
        self.vectors = [np.asarray(v, dtype=float) for v in vectors]
        
        # Check vector dimensions
        if any(v.ndim != 1 for v in self.vectors):
            raise ValueError("All vectors must be 1-dimensional")
        
        # Check vector lengths
        lengths = [len(v) for v in self.vectors]
        if len(set(lengths)) > 1:
            raise ValueError(f"All vectors must have the same length. Found lengths: {lengths}")
        
        self.n_vectors = len(self.vectors)
        self.vector_length = lengths[0]
        self.tolerance = tolerance
        
        # Set labels
        if labels is None:
            self.labels = [f"Vector_{i}" for i in range(self.n_vectors)]
        else:
            if len(labels) != self.n_vectors:
                raise ValueError(f"Number of labels ({len(labels)}) must match number of vectors ({self.n_vectors})")
            self.labels = labels
        
        # Store optimization metrics
        self.objective_values = objective_values
        self.gradient_norms = gradient_norms
        self.hessian_min_eigenvalues = hessian_min_eigenvalues
        self.hessian_max_eigenvalues = hessian_max_eigenvalues
        
        # Validate optimization metrics
        self._validate_optimization_metrics()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _validate_optimization_metrics(self) -> None:
        """Validate that optimization metrics have correct dimensions."""
        metrics = {
            'objective_values': self.objective_values,
            'gradient_norms': self.gradient_norms,
            'hessian_min_eigenvalues': self.hessian_min_eigenvalues,
            'hessian_max_eigenvalues': self.hessian_max_eigenvalues
        }
        
        for name, metric in metrics.items():
            if metric is not None and len(metric) != self.n_vectors:
                raise ValueError(f"{name} length ({len(metric)}) must match number of vectors ({self.n_vectors})")
    
    def magnitude_comparison(self) -> Dict[str, Any]:
        """
        Calculate and compare Euclidean norms (L2 norms) of all vectors.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'norms': List of Euclidean norms for each vector
            - 'rankings': List of vector indices sorted by norm (highest to lowest)
            - 'norm_ratio': Ratio of maximum to minimum norm
            - 'norm_differences': Pairwise differences between norms
        """
        norms = [np.linalg.norm(v) for v in self.vectors]
        rankings = np.argsort(norms)[::-1]  # Highest to lowest
        
        norm_ratio = max(norms) / min(norms) if min(norms) > 0 else np.inf
        
        # Calculate pairwise differences
        norm_differences = np.abs(np.array(norms)[:, np.newaxis] - np.array(norms))
        
        return {
            'norms': norms,
            'rankings': rankings.tolist(),
            'norm_ratio': norm_ratio,
            'norm_differences': norm_differences
        }
    
    def component_wise_analysis(
        self, 
        reference_idx: int = 0,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform element-wise analysis between vectors and a reference vector.
        
        Parameters
        ----------
        reference_idx : int, default=0
            Index of the reference vector for comparison.
        threshold : Optional[float], default=None
            Threshold for identifying significant differences. If None, uses 2*std.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing component-wise analysis results.
        """
        if reference_idx >= self.n_vectors:
            raise ValueError(f"reference_idx ({reference_idx}) must be < {self.n_vectors}")
        
        reference = self.vectors[reference_idx]
        results = {'reference': self.labels[reference_idx]}
        
        for i, (vector, label) in enumerate(zip(self.vectors, self.labels)):
            if i == reference_idx:
                continue
            
            # Calculate differences and ratios
            differences = vector - reference
            ratios = np.divide(vector, reference, out=np.zeros_like(vector), where=reference!=0)
            
            # Statistical measures
            mean_abs_diff = np.mean(np.abs(differences))
            max_diff = np.max(np.abs(differences))
            max_ratio = np.max(np.abs(ratios)) if np.any(ratios != 0) else 0
            
            # Find positions of significant differences
            if threshold is None:
                threshold = 2 * np.std(differences)
            
            significant_positions = np.where(np.abs(differences) > threshold)[0]
            
            results[label] = {
                'differences': differences,
                'ratios': ratios,
                'mean_absolute_difference': mean_abs_diff,
                'max_difference': max_diff,
                'max_ratio': max_ratio,
                'significant_positions': significant_positions.tolist(),
                'threshold': threshold
            }
        
        return results
    
    def statistical_summary(self) -> pd.DataFrame:
        """
        Calculate comprehensive statistical summaries for all vectors.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with statistical measures for each vector, including
            optimization metrics if available.
        """
        data = []
        
        for i, (vector, label) in enumerate(zip(self.vectors, self.labels)):
            row = {
                'Vector': label,
                'Length': len(vector),
                'Mean': np.mean(vector),
                'Median': np.median(vector),
                'Std': np.std(vector),
                'Min': np.min(vector),
                'Max': np.max(vector),
                'Range': np.ptp(vector),
                'Q1': np.percentile(vector, 25),
                'Q3': np.percentile(vector, 75),
                'IQR': np.percentile(vector, 75) - np.percentile(vector, 25),
                'Euclidean_Norm': np.linalg.norm(vector)
            }
            
            # Add optimization metrics if available
            if self.objective_values is not None:
                row['Objective_Value'] = self.objective_values[i]
            if self.gradient_norms is not None:
                row['Gradient_Norm'] = self.gradient_norms[i]
            if self.hessian_min_eigenvalues is not None:
                row['Hessian_Min_Eigenvalue'] = self.hessian_min_eigenvalues[i]
            if self.hessian_max_eigenvalues is not None:
                row['Hessian_Max_Eigenvalue'] = self.hessian_max_eigenvalues[i]
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def distance_matrix(self, metric: str = 'euclidean') -> pd.DataFrame:
        """
        Calculate distance matrix between all pairs of vectors.
        
        Parameters
        ----------
        metric : str, default='euclidean'
            Distance metric to use. Options: 'euclidean', 'manhattan', 'chebyshev'.
        
        Returns
        -------
        pd.DataFrame
            Distance matrix with vector labels as index and columns.
        """
        if metric == 'euclidean':
            distances = pdist(self.vectors, metric='euclidean')
        elif metric == 'manhattan':
            distances = pdist(self.vectors, metric='cityblock')
        elif metric == 'chebyshev':
            distances = pdist(self.vectors, metric='chebyshev')
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        distance_matrix = squareform(distances)
        return pd.DataFrame(distance_matrix, index=self.labels, columns=self.labels)
    
    def similarity_matrix(self, metric: str = 'cosine') -> pd.DataFrame:
        """
        Calculate similarity matrix between all pairs of vectors.
        
        Parameters
        ----------
        metric : str, default='cosine'
            Similarity metric to use. Options: 'cosine', 'pearson'.
        
        Returns
        -------
        pd.DataFrame
            Similarity matrix with vector labels as index and columns.
        """
        n = self.n_vectors
        similarity_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                if metric == 'cosine':
                    # Cosine similarity
                    dot_product = np.dot(self.vectors[i], self.vectors[j])
                    norms = np.linalg.norm(self.vectors[i]) * np.linalg.norm(self.vectors[j])
                    similarity = dot_product / norms if norms > 0 else 0
                elif metric == 'pearson':
                    # Pearson correlation
                    similarity, _ = pearsonr(self.vectors[i], self.vectors[j])
                    if np.isnan(similarity):
                        similarity = 0
                else:
                    raise ValueError(f"Unsupported similarity metric: {metric}")
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return pd.DataFrame(similarity_matrix, index=self.labels, columns=self.labels)
    
    def normalize_vectors(self, method: str = 'l2') -> 'VectorComparator':
        """
        Create a new VectorComparator with normalized vectors.
        
        Parameters
        ----------
        method : str, default='l2'
            Normalization method. Options: 'l2', 'standardize', 'minmax'.
        
        Returns
        -------
        VectorComparator
            New VectorComparator instance with normalized vectors.
        """
        normalized_vectors = []
        
        for vector in self.vectors:
            if method == 'l2':
                # L2 normalization (unit length)
                norm = np.linalg.norm(vector)
                normalized = vector / norm if norm > 0 else vector
            elif method == 'standardize':
                # Standardization (zero mean, unit variance)
                normalized = (vector - np.mean(vector)) / (np.std(vector) + 1e-8)
            elif method == 'minmax':
                # Min-max scaling to [0, 1]
                v_min, v_max = np.min(vector), np.max(vector)
                normalized = (vector - v_min) / (v_max - v_min + 1e-8)
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            normalized_vectors.append(normalized)
        
        # Create new labels to indicate normalization
        new_labels = [f"{label}_{method}" for label in self.labels]
        
        return VectorComparator(
            vectors=normalized_vectors,
            labels=new_labels,
            objective_values=self.objective_values,
            gradient_norms=self.gradient_norms,
            hessian_min_eigenvalues=self.hessian_min_eigenvalues,
            hessian_max_eigenvalues=self.hessian_max_eigenvalues,
            tolerance=self.tolerance
        )
    
    def plot_vectors(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create line plot comparing all vectors.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(12, 8)
            Figure size (width, height).
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (vector, label) in enumerate(zip(self.vectors, self.labels)):
            ax.plot(vector, label=label, marker='o', markersize=4, linewidth=2)
        
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Value')
        ax.set_title('Vector Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distance_heatmap(self, metric: str = 'euclidean', figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Create heatmap of distance matrix.
        
        Parameters
        ----------
        metric : str, default='euclidean'
            Distance metric to use.
        figsize : Tuple[int, int], default=(8, 6)
            Figure size (width, height).
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        distance_matrix = self.distance_matrix(metric)
        sns.heatmap(distance_matrix, annot=True, cmap='viridis', ax=ax, 
                   square=True, fmt='.3f')
        
        ax.set_title(f'Distance Matrix ({metric.title()})')
        plt.tight_layout()
        return fig
    
    def plot_optimization_metrics(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create bar charts comparing optimization metrics.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(15, 10)
            Figure size (width, height).
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        metrics_data = self.statistical_summary()
        
        # Determine number of subplots needed
        metric_cols = []
        if 'Objective_Value' in metrics_data.columns:
            metric_cols.append('Objective_Value')
        if 'Gradient_Norm' in metrics_data.columns:
            metric_cols.append('Gradient_Norm')
        if 'Hessian_Min_Eigenvalue' in metrics_data.columns:
            metric_cols.append('Hessian_Min_Eigenvalue')
        if 'Hessian_Max_Eigenvalue' in metrics_data.columns:
            metric_cols.append('Hessian_Max_Eigenvalue')
        
        if not metric_cols:
            raise ValueError("No optimization metrics available for plotting")
        
        n_metrics = len(metric_cols)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metric_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            bars = ax.bar(metrics_data['Vector'], metrics_data[metric])
            ax.set_title(f'{metric.replace("_", " ")}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2e}' if abs(value) < 0.01 or abs(value) > 1000 else f'{value:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_component_analysis(self, reference_idx: int = 0, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create component-wise analysis plots.
        
        Parameters
        ----------
        reference_idx : int, default=0
            Index of reference vector.
        figsize : Tuple[int, int], default=(15, 10)
            Figure size (width, height).
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        analysis = self.component_wise_analysis(reference_idx)
        reference_label = analysis['reference']
        
        n_comparisons = len(analysis) - 1  # Exclude reference
        n_cols = min(2, n_comparisons)
        n_rows = (n_comparisons + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_comparisons == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        for label, data in analysis.items():
            if label == 'reference':
                continue
            
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot differences
            differences = data['differences']
            ax.plot(differences, marker='o', markersize=3, linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title(f'{label} vs {reference_label} (Differences)')
            ax.set_xlabel('Component Index')
            ax.set_ylabel('Difference')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(n_comparisons, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, include_plots: bool = False, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report.
        
        Parameters
        ----------
        include_plots : bool, default=False
            Whether to include plots in the report.
        output_file : Optional[str], default=None
            File path to save the report as JSON.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive report dictionary.
        """
        report = {
            'summary': {
                'n_vectors': self.n_vectors,
                'vector_length': self.vector_length,
                'labels': self.labels,
                'has_optimization_metrics': any([
                    self.objective_values is not None,
                    self.gradient_norms is not None,
                    self.hessian_min_eigenvalues is not None,
                    self.hessian_max_eigenvalues is not None
                ])
            },
            'statistical_summary': self.statistical_summary().to_dict('records'),
            'magnitude_comparison': self.magnitude_comparison(),
            'distance_analysis': {
                'euclidean': self.distance_matrix('euclidean').to_dict(),
                'manhattan': self.distance_matrix('manhattan').to_dict()
            },
            'similarity_analysis': {
                'cosine': self.similarity_matrix('cosine').to_dict(),
                'pearson': self.similarity_matrix('pearson').to_dict()
            },
            'component_analysis': self.component_wise_analysis()
        }
        
        # Add optimization-specific analysis if metrics are available
        if report['summary']['has_optimization_metrics']:
            report['optimization_analysis'] = self._analyze_optimization_metrics()
        
        if include_plots:
            report['plots'] = {
                'note': 'Plots would be generated here in a full implementation'
            }
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _analyze_optimization_metrics(self) -> Dict[str, Any]:
        """Analyze optimization metrics and provide insights."""
        analysis = {}
        
        if self.objective_values is not None:
            obj_values = np.array(self.objective_values)
            analysis['objective_analysis'] = {
                'best_objective': np.min(obj_values),
                'worst_objective': np.max(obj_values),
                'best_vector_idx': int(np.argmin(obj_values)),
                'best_vector_label': self.labels[np.argmin(obj_values)],
                'objective_range': np.ptp(obj_values),
                'objective_std': np.std(obj_values)
            }
        
        if self.gradient_norms is not None:
            grad_norms = np.array(self.gradient_norms)
            converged_runs = np.sum(grad_norms < self.tolerance)
            analysis['gradient_analysis'] = {
                'converged_runs': int(converged_runs),
                'convergence_rate': converged_runs / len(grad_norms),
                'min_gradient_norm': np.min(grad_norms),
                'max_gradient_norm': np.max(grad_norms),
                'best_gradient_idx': int(np.argmin(grad_norms)),
                'best_gradient_label': self.labels[np.argmin(grad_norms)]
            }
        
        if self.hessian_min_eigenvalues is not None and self.hessian_max_eigenvalues is not None:
            min_eigs = np.array(self.hessian_min_eigenvalues)
            max_eigs = np.array(self.hessian_max_eigenvalues)
            condition_numbers = max_eigs / np.abs(min_eigs)
            
            analysis['hessian_analysis'] = {
                'min_eigenvalue_range': [np.min(min_eigs), np.max(min_eigs)],
                'max_eigenvalue_range': [np.min(max_eigs), np.max(max_eigs)],
                'condition_number_range': [np.min(condition_numbers), np.max(condition_numbers)],
                'ill_conditioned_runs': int(np.sum(condition_numbers > 1e12)),
                'best_conditioned_idx': int(np.argmin(condition_numbers)),
                'best_conditioned_label': self.labels[np.argmin(condition_numbers)]
            }
        
        return analysis
    
    def save_results(self, filename: str, format: str = 'json') -> None:
        """
        Save comparison results to file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        format : str, default='json'
            Output format. Options: 'json', 'csv' (for statistical summary only).
        """
        if format == 'json':
            report = self.generate_report(include_plots=False)
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'csv':
            stats = self.statistical_summary()
            stats.to_csv(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions for quick analysis
def quick_compare(vectors: List[Union[np.ndarray, List[float]]], 
                 labels: Optional[List[str]] = None,
                 **kwargs) -> VectorComparator:
    """
    Quick comparison of vectors with minimal setup.
    
    Parameters
    ----------
    vectors : List[Union[np.ndarray, List[float]]]
        List of vectors to compare.
    labels : Optional[List[str]], default=None
        Labels for vectors.
    **kwargs
        Additional arguments passed to VectorComparator.
    
    Returns
    -------
    VectorComparator
        Configured VectorComparator instance.
    """
    return VectorComparator(vectors, labels, **kwargs)


def load_comparison_results(filename: str) -> Dict[str, Any]:
    """
    Load previously saved comparison results.
    
    Parameters
    ----------
    filename : str
        Path to the saved results file.
    
    Returns
    -------
    Dict[str, Any]
        Loaded comparison results.
    """
    with open(filename, 'r') as f:
        return json.load(f)
