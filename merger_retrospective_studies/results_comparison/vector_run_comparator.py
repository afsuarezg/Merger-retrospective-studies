"""
Vector Run Comparator Module

This module provides specialized tools for comparing multiple runs of a single algorithm
where each run produces a vector output. It focuses on distance-based and similarity-based
metrics to analyze the consistency, convergence, and performance of vector optimization results.

Key Features:
- Euclidean distance analysis between runs
- Cosine similarity measurements
- Convergence pattern analysis
- Statistical significance testing
- Comprehensive visualization tools
- Performance ranking and clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings
from typing import List, Dict, Optional, Tuple, Union
import itertools

class VectorRunComparator:
    """
    A specialized tool for comparing multiple vector runs from a single algorithm.
    
    This class provides comprehensive analysis of vector optimization results,
    focusing on distance-based metrics, similarity measures, and convergence patterns.
    """
    
    def __init__(self, vector_runs: Union[List[np.ndarray], np.ndarray], 
                 run_labels: Optional[List[str]] = None):
        """
        Initialize the VectorRunComparator with multiple vector runs.
        
        Parameters:
        -----------
        vector_runs : Union[List[np.ndarray], np.ndarray]
            List of 1D arrays where each array represents one run's results.
            Each array should have shape (n_iterations,) for 1D vectors.
        run_labels : List[str], optional
            Labels for each run. If None, will use 'Run_1', 'Run_2', etc.
        """
        if isinstance(vector_runs, np.ndarray):
            if vector_runs.ndim == 2:
                # Shape: (n_runs, n_iterations) for 1D vectors
                self.vector_runs = [vector_runs[i] for i in range(vector_runs.shape[0])]
            else:
                raise ValueError("If providing numpy array, it must be 2D: (n_runs, n_iterations) for 1D vectors")
        else:
            self.vector_runs = [np.array(run) for run in vector_runs]
        
        self.n_runs = len(self.vector_runs)
        self.n_iterations = self.vector_runs[0].shape[0]
        self.n_dimensions = 1  # Always 1 for 1D vectors
        
        # Validate all runs have same length
        for i, run in enumerate(self.vector_runs):
            if run.shape[0] != self.n_iterations:
                raise ValueError(f"Run {i} has {run.shape[0]} iterations, expected {self.n_iterations}")
        
        # Set run labels
        if run_labels is None:
            self.run_labels = [f'Run_{i+1}' for i in range(self.n_runs)]
        else:
            if len(run_labels) != self.n_runs:
                raise ValueError(f"Number of labels ({len(run_labels)}) must match number of runs ({self.n_runs})")
            self.run_labels = run_labels
        
        # Calculate basic statistics
        self._calculate_basic_stats()
    
    def _calculate_basic_stats(self):
        """Calculate basic statistics for all runs."""
        self.run_stats = {}
        
        for i, (run, label) in enumerate(zip(self.vector_runs, self.run_labels)):
            # For 1D vectors, the values themselves are the 'norms'
            values = np.abs(run)  # Use absolute values as the magnitude
            
            # Calculate final value (last iteration)
            final_value = run[-1]
            
            # Calculate mean value across all iterations
            mean_value = np.mean(run)
            
            self.run_stats[label] = {
                'values': values,
                'final_value': final_value,
                'mean_value': mean_value,
                'best_value': np.min(values),
                'worst_value': np.max(values),
                'mean_abs_value': np.mean(values),
                'std_abs_value': np.std(values),
                'convergence_rate': self._calculate_convergence_rate(values)
            }
    
    def _calculate_convergence_rate(self, values: np.ndarray) -> float:
        """Calculate convergence rate based on value improvement."""
        if len(values) < 2:
            return 0.0
        
        improvements = np.diff(values)
        negative_improvements = improvements[improvements < 0]
        
        if len(negative_improvements) == 0:
            return 0.0
        
        return np.mean(negative_improvements)
    
    def euclidean_distance_analysis(self) -> pd.DataFrame:
        """
        Analyze Euclidean distances between all pairs of runs.
        
        Returns:
        --------
        pd.DataFrame : Distance analysis results
        """
        # Calculate pairwise distances between final values
        final_values = np.array([stats['final_value'] for stats in self.run_stats.values()])
        distances = np.abs(final_values[:, np.newaxis] - final_values[np.newaxis, :])
        
        # Calculate pairwise distances between mean values
        mean_values = np.array([stats['mean_value'] for stats in self.run_stats.values()])
        mean_distances = np.abs(mean_values[:, np.newaxis] - mean_values[np.newaxis, :])
        
        # Create results dataframe
        results = []
        for i in range(self.n_runs):
            for j in range(i+1, self.n_runs):
                results.append({
                    'Run_1': self.run_labels[i],
                    'Run_2': self.run_labels[j],
                    'Final_Value_Distance': distances[i, j],
                    'Mean_Value_Distance': mean_distances[i, j],
                    'Distance_Ratio': distances[i, j] / mean_distances[i, j] if mean_distances[i, j] > 0 else np.inf
                })
        
        return pd.DataFrame(results)
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Analyze correlations between all pairs of runs.
        
        Returns:
        --------
        pd.DataFrame : Correlation analysis results
        """
        # Calculate pairwise correlations between runs
        run_data = np.array(self.vector_runs)  # Shape: (n_runs, n_iterations)
        correlations = np.corrcoef(run_data)
        
        # Create results dataframe
        results = []
        for i in range(self.n_runs):
            for j in range(i+1, self.n_runs):
                results.append({
                    'Run_1': self.run_labels[i],
                    'Run_2': self.run_labels[j],
                    'Correlation': correlations[i, j],
                    'Abs_Correlation': abs(correlations[i, j]),
                    'Correlation_Strength': self._interpret_correlation(abs(correlations[i, j]))
                })
        
        return pd.DataFrame(results)
    
    def convergence_analysis(self) -> pd.DataFrame:
        """
        Analyze convergence patterns across runs.
        
        Returns:
        --------
        pd.DataFrame : Convergence analysis results
        """
        results = []
        
        for label, stats in self.run_stats.items():
            values = stats['values']
            
            # Calculate convergence metrics
            initial_value = float(values[0])
            final_value = float(values[-1])
            improvement = initial_value - final_value
            improvement_ratio = improvement / initial_value if abs(float(initial_value)) > 1e-10 else 0
            
            # Calculate stability (variance in last quarter)
            last_quarter = int(0.75 * len(values))
            stability = np.std(values[last_quarter:])
            
            # Calculate monotonicity (fraction of improving steps)
            improvements = np.diff(values)
            monotonicity = np.sum(improvements < 0) / len(improvements) if len(improvements) > 0 else 0
            
            results.append({
                'Run': label,
                'Initial_Value': initial_value,
                'Final_Value': final_value,
                'Improvement': improvement,
                'Improvement_Ratio': improvement_ratio,
                'Convergence_Rate': stats['convergence_rate'],
                'Stability': stability,
                'Monotonicity': monotonicity,
                'Best_Value': stats['best_value'],
                'Mean_Value': stats['mean_abs_value'],
                'Std_Value': stats['std_abs_value']
            })
        
        return pd.DataFrame(results)
    
    def statistical_significance_tests(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform statistical significance tests between runs.
        
        Parameters:
        -----------
        alpha : float
            Significance level for tests
            
        Returns:
        --------
        pd.DataFrame : Statistical test results
        """
        results = []
        
        for i in range(self.n_runs):
            for j in range(i+1, self.n_runs):
                run1_values = self.vector_runs[i]
                run2_values = self.vector_runs[j]
                
                # Mann-Whitney U test
                mw_stat, mw_p = stats.mannwhitneyu(run1_values, run2_values, alternative='two-sided')
                
                # Welch's t-test
                t_stat, t_p = stats.ttest_ind(run1_values, run2_values, equal_var=False)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.ks_2samp(run1_values, run2_values)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(run1_values)-1)*np.var(run1_values, ddof=1) + 
                                    (len(run2_values)-1)*np.var(run2_values, ddof=1)) / 
                                   (len(run1_values) + len(run2_values) - 2))
                cohens_d = (np.mean(run1_values) - np.mean(run2_values)) / pooled_std
                
                results.append({
                    'Run_1': self.run_labels[i],
                    'Run_2': self.run_labels[j],
                    'MannWhitney_Stat': mw_stat,
                    'MannWhitney_p': mw_p,
                    'MannWhitney_Significant': mw_p < alpha,
                    'TTest_Stat': t_stat,
                    'TTest_p': t_p,
                    'TTest_Significant': t_p < alpha,
                    'KS_Stat': ks_stat,
                    'KS_p': ks_p,
                    'KS_Significant': ks_p < alpha,
                    'Cohens_d': cohens_d,
                    'Effect_Size': self._interpret_effect_size(abs(cohens_d))
                })
        
        return pd.DataFrame(results)
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return 'Negligible'
        elif d < 0.5:
            return 'Small'
        elif d < 0.8:
            return 'Medium'
        else:
            return 'Large'
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength."""
        if r < 0.1:
            return 'Negligible'
        elif r < 0.3:
            return 'Small'
        elif r < 0.5:
            return 'Medium'
        elif r < 0.7:
            return 'Large'
        else:
            return 'Very Large'
    
    def clustering_analysis(self, method: str = 'ward') -> Dict:
        """
        Perform hierarchical clustering analysis of runs.
        
        Parameters:
        -----------
        method : str
            Linkage method for clustering ('ward', 'complete', 'average', 'single')
            
        Returns:
        --------
        Dict : Clustering analysis results
        """
        # Use final values for clustering
        final_values = np.array([stats['final_value'] for stats in self.run_stats.values()])
        
        # Calculate condensed distance matrix for linkage
        condensed_distances = pdist(final_values.reshape(-1, 1), metric='euclidean')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=method)
        
        # Calculate cophenetic correlation
        from scipy.cluster.hierarchy import cophenet
        cophenetic_distances = cophenet(linkage_matrix)
        cophenetic_corr = np.corrcoef(condensed_distances, cophenetic_distances)[0, 1]
        
        # Convert back to square form for visualization
        distance_matrix = squareform(condensed_distances)
        
        return {
            'linkage_matrix': linkage_matrix,
            'distance_matrix': distance_matrix,
            'cophenetic_correlation': cophenetic_corr,
            'run_labels': self.run_labels
        }
    
    def plot_comprehensive_analysis(self, figsize: tuple = (20, 15)) -> plt.Figure:
        """
        Create comprehensive visualization of vector run analysis.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig = plt.figure(figsize=figsize)
        
        # 1. Convergence plots
        ax1 = plt.subplot(3, 4, 1)
        for i, (label, stats) in enumerate(self.run_stats.items()):
            ax1.plot(stats['values'], alpha=0.7, label=label, linewidth=2)
        ax1.set_title('Convergence of All Runs')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Final values distribution
        ax2 = plt.subplot(3, 4, 2)
        final_values = [stats['final_value'] for stats in self.run_stats.values()]
        ax2.bar(self.run_labels, final_values, alpha=0.7)
        ax2.set_title('Final Values by Run')
        ax2.set_ylabel('Final Value')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Distance matrix heatmap
        ax3 = plt.subplot(3, 4, 3)
        euclidean_df = self.euclidean_distance_analysis()
        n_runs = len(self.run_labels)
        distance_matrix = np.zeros((n_runs, n_runs))
        
        for _, row in euclidean_df.iterrows():
            i = self.run_labels.index(row['Run_1'])
            j = self.run_labels.index(row['Run_2'])
            distance_matrix[i, j] = row['Final_Value_Distance']
            distance_matrix[j, i] = row['Final_Value_Distance']
        
        im = ax3.imshow(distance_matrix, cmap='viridis', aspect='auto')
        ax3.set_title('Value Distance Matrix')
        ax3.set_xticks(range(n_runs))
        ax3.set_yticks(range(n_runs))
        ax3.set_xticklabels(self.run_labels, rotation=45)
        ax3.set_yticklabels(self.run_labels)
        plt.colorbar(im, ax=ax3)
        
        # 4. Correlation heatmap
        ax4 = plt.subplot(3, 4, 4)
        correlation_df = self.correlation_analysis()
        correlation_matrix = np.eye(n_runs)  # Identity matrix for diagonal
        
        for _, row in correlation_df.iterrows():
            i = self.run_labels.index(row['Run_1'])
            j = self.run_labels.index(row['Run_2'])
            correlation_matrix[i, j] = row['Correlation']
            correlation_matrix[j, i] = row['Correlation']
        
        im = ax4.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_title('Correlation Matrix')
        ax4.set_xticks(range(n_runs))
        ax4.set_yticks(range(n_runs))
        ax4.set_xticklabels(self.run_labels, rotation=45)
        ax4.set_yticklabels(self.run_labels)
        plt.colorbar(im, ax=ax4)
        
        # 5. Final values scatter plot
        ax5 = plt.subplot(3, 4, 5)
        for i, (label, stats) in enumerate(self.run_stats.items()):
            final_val = stats['final_value']
            ax5.scatter(i, final_val, label=label, s=100, alpha=0.7)
        ax5.set_title('Final Values by Run')
        ax5.set_xlabel('Run Index')
        ax5.set_ylabel('Final Value')
        ax5.set_xticks(range(len(self.run_labels)))
        ax5.set_xticklabels(self.run_labels, rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Value distribution histogram
        ax6 = plt.subplot(3, 4, 6)
        all_values = np.concatenate([run for run in self.vector_runs])
        ax6.hist(all_values, bins=20, alpha=0.7, edgecolor='black')
        ax6.set_title('Distribution of All Values')
        ax6.set_xlabel('Value')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        # 7. Box plot of values
        ax7 = plt.subplot(3, 4, 7)
        value_data = [run for run in self.vector_runs]
        box_plot = ax7.boxplot(value_data, patch_artist=True)
        ax7.set_xticklabels(self.run_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        ax7.set_title('Distribution of Values by Run')
        ax7.set_ylabel('Value')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # 8. Improvement analysis
        ax8 = plt.subplot(3, 4, 8)
        convergence_df = self.convergence_analysis()
        ax8.bar(convergence_df['Run'], convergence_df['Improvement_Ratio'], alpha=0.7)
        ax8.set_title('Improvement Ratio by Run')
        ax8.set_ylabel('Improvement Ratio')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3)
        
        # 9. Stability analysis
        ax9 = plt.subplot(3, 4, 9)
        ax9.bar(convergence_df['Run'], convergence_df['Stability'], alpha=0.7)
        ax9.set_title('Stability by Run (Lower is Better)')
        ax9.set_ylabel('Stability (Std of Last Quarter)')
        ax9.tick_params(axis='x', rotation=45)
        ax9.grid(True, alpha=0.3)
        
        # 10. Monotonicity analysis
        ax10 = plt.subplot(3, 4, 10)
        ax10.bar(convergence_df['Run'], convergence_df['Monotonicity'], alpha=0.7)
        ax10.set_title('Monotonicity by Run (Higher is Better)')
        ax10.set_ylabel('Fraction of Improving Steps')
        ax10.tick_params(axis='x', rotation=45)
        ax10.grid(True, alpha=0.3)
        
        # 11. Dendrogram
        ax11 = plt.subplot(3, 4, 11)
        try:
            clustering_results = self.clustering_analysis()
            dendrogram(clustering_results['linkage_matrix'], 
                      labels=clustering_results['run_labels'],
                      ax=ax11)
            ax11.set_title(f'Clustering Dendrogram\n(Cophenetic Corr: {clustering_results["cophenetic_correlation"]:.3f})')
            ax11.tick_params(axis='x', rotation=45)
        except Exception as e:
            ax11.text(0.5, 0.5, f'Clustering Error: {str(e)}', ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('Clustering Dendrogram (Error)')
        
        # 12. Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Create summary text
        final_values = [stats['final_value'] for stats in self.run_stats.values()]
        summary_text = f"""
        Summary Statistics:
        
        Number of Runs: {self.n_runs}
        Iterations per Run: {self.n_iterations}
        Dimensions: {self.n_dimensions}
        
        Best Final Value: {min(final_values):.4f}
        Worst Final Value: {max(final_values):.4f}
        
        Mean Distance: {euclidean_df['Final_Value_Distance'].mean():.4f}
        Mean Correlation: {correlation_df['Correlation'].mean():.4f}
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive text report of the analysis.
        
        Returns:
        --------
        str : Formatted analysis report
        """
        report = []
        report.append("=" * 80)
        report.append("VECTOR RUN COMPARISON ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic information
        report.append("BASIC INFORMATION")
        report.append("-" * 40)
        report.append(f"Number of Runs: {self.n_runs}")
        report.append(f"Iterations per Run: {self.n_iterations}")
        report.append(f"Dimensions: {self.n_dimensions}")
        report.append("")
        
        # Convergence analysis
        report.append("CONVERGENCE ANALYSIS")
        report.append("-" * 40)
        convergence_df = self.convergence_analysis()
        report.append(convergence_df.to_string(index=False))
        report.append("")
        
        # Distance analysis
        report.append("EUCLIDEAN DISTANCE ANALYSIS")
        report.append("-" * 40)
        euclidean_df = self.euclidean_distance_analysis()
        report.append(euclidean_df.to_string(index=False))
        report.append("")
        
        # Correlation analysis
        report.append("CORRELATION ANALYSIS")
        report.append("-" * 40)
        correlation_df = self.correlation_analysis()
        report.append(correlation_df.to_string(index=False))
        report.append("")
        
        # Statistical tests
        report.append("STATISTICAL SIGNIFICANCE TESTS")
        report.append("-" * 40)
        stats_df = self.statistical_significance_tests()
        report.append(stats_df.to_string(index=False))
        report.append("")
        
        # Clustering analysis
        report.append("CLUSTERING ANALYSIS")
        report.append("-" * 40)
        clustering_results = self.clustering_analysis()
        report.append(f"Cophenetic Correlation: {clustering_results['cophenetic_correlation']:.4f}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find best and worst runs
        best_run = convergence_df.loc[convergence_df['Final_Value'].idxmin(), 'Run']
        worst_run = convergence_df.loc[convergence_df['Final_Value'].idxmax(), 'Run']
        
        report.append(f"• Best performing run: {best_run}")
        report.append(f"• Worst performing run: {worst_run}")
        
        # Consistency analysis
        mean_distance = euclidean_df['Final_Value_Distance'].mean()
        mean_correlation = correlation_df['Correlation'].mean()
        
        if mean_distance < 0.1:
            report.append("• High consistency: Runs converge to similar values")
        elif mean_distance < 0.5:
            report.append("• Moderate consistency: Some variation in final values")
        else:
            report.append("• Low consistency: High variation in final values")
        
        if mean_correlation > 0.9:
            report.append("• High correlation: Runs follow similar optimization paths")
        elif mean_correlation > 0.7:
            report.append("• Moderate correlation: Some similarity in optimization paths")
        else:
            report.append("• Low correlation: High variation in optimization paths")
        
        return "\n".join(report)


# Example usage and demonstration
def demo_vector_run_comparator():
    """Demonstrate the VectorRunComparator with sample data."""
    
    # Generate sample vector runs
    np.random.seed(42)
    
    print("=" * 80)
    print("VECTOR RUN COMPARATOR DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create sample vector runs (4 runs, 50 iterations each, 1D vectors)
    vector_runs = []
    run_labels = ['Standard_Params', 'Tuned_Params', 'High_Exploration', 'Low_Exploration']
    
    # Run 1: Standard parameters
    vector_runs.append(np.random.normal(5, 1, 50))
    
    # Run 2: Tuned parameters (better convergence)
    base_trajectory = np.linspace(5, 4, 50)
    noise = np.random.normal(0, 0.5, 50)
    vector_runs.append(base_trajectory + noise)
    
    # Run 3: High exploration (more noise, slower convergence)
    base_trajectory = np.linspace(5, 4.5, 50)
    noise = np.random.normal(0, 1.5, 50)
    vector_runs.append(base_trajectory + noise)
    
    # Run 4: Low exploration (very little noise, fast convergence)
    base_trajectory = np.linspace(5, 4.2, 50)
    noise = np.random.normal(0, 0.2, 50)
    vector_runs.append(base_trajectory + noise)
    
    # Create comparator
    comparator = VectorRunComparator(vector_runs, run_labels)
    
    print("Creating comprehensive vector run analysis...")
    print(f"Analyzing {comparator.n_runs} runs with {comparator.n_iterations} iterations each")
    print(f"Each run produces {comparator.n_dimensions}-dimensional vectors")
    print()
    
    # Perform analyses
    print("1. CONVERGENCE ANALYSIS")
    print("-" * 40)
    convergence_df = comparator.convergence_analysis()
    print(convergence_df)
    print()
    
    print("2. EUCLIDEAN DISTANCE ANALYSIS")
    print("-" * 40)
    euclidean_df = comparator.euclidean_distance_analysis()
    print(euclidean_df)
    print()
    
    print("3. CORRELATION ANALYSIS")
    print("-" * 40)
    correlation_df = comparator.correlation_analysis()
    print(correlation_df)
    print()
    
    print("4. STATISTICAL SIGNIFICANCE TESTS")
    print("-" * 40)
    stats_df = comparator.statistical_significance_tests()
    print(stats_df)
    print()
    
    print("5. CLUSTERING ANALYSIS")
    print("-" * 40)
    clustering_results = comparator.clustering_analysis()
    print(f"Cophenetic Correlation: {clustering_results['cophenetic_correlation']:.4f}")
    print()
    
    # Create visualizations
    print("Creating comprehensive visualizations...")
    fig = comparator.plot_comprehensive_analysis(figsize=(24, 18))
    plt.suptitle('Vector Run Comparator - Comprehensive Analysis', fontsize=20, y=0.98)
    plt.show()
    
    # Generate comprehensive report
    print("=" * 80)
    print("COMPREHENSIVE REPORT")
    print("=" * 80)
    
    report = comparator.generate_comprehensive_report()
    print(report)
    
    return comparator


# Run demonstration if script is executed directly
if __name__ == "__main__":
    demo_comparator = demo_vector_run_comparator()
