import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson, kstest
import warnings
from typing import List, Dict, Optional, Union

class OptimizationAnalyzer:
    """
    A comprehensive tool for analyzing multiple runs of optimization problems.
    
    This class provides statistical analysis, distribution characterization,
    and visualization for optimization results.
    """
    
    def __init__(self, results: Union[List[float], np.ndarray, Dict[str, List[float]], 
                                   Dict[str, np.ndarray], List[np.ndarray]]):
        """
        Initialize the analyzer with optimization results.
        
        Parameters:
        -----------
        results : Union[List[float], np.ndarray, Dict[str, List[float]], Dict[str, np.ndarray], List[np.ndarray]]
            Single algorithm results as list/array, or multiple algorithms as dict.
            Can also handle vector results (2D arrays) for multi-dimensional optimization.
        """
        if isinstance(results, dict):
            # Handle dictionary of results
            self.results_dict = {}
            self.vector_results = {}
            self.is_vector_optimization = False
            
            for name, res in results.items():
                res_array = np.array(res)
                if res_array.ndim == 1:
                    self.results_dict[name] = res_array
                elif res_array.ndim == 2:
                    self.vector_results[name] = res_array
                    self.is_vector_optimization = True
                    # Also store scalar results (norms, means, etc.)
                    self.results_dict[name] = np.linalg.norm(res_array, axis=1)
                else:
                    raise ValueError(f"Results for {name} must be 1D or 2D arrays")
            
            self.single_algorithm = len(results) == 1
            self.algorithm_names = list(results.keys())
            
        elif isinstance(results, list) and len(results) > 0 and isinstance(results[0], np.ndarray):
            # Handle list of vector results
            self.results_dict = {}
            self.vector_results = {}
            self.is_vector_optimization = True
            
            for i, res in enumerate(results):
                res_array = np.array(res)
                if res_array.ndim == 2:
                    name = f'Algorithm_{i+1}'
                    self.vector_results[name] = res_array
                    self.results_dict[name] = np.linalg.norm(res_array, axis=1)
                else:
                    raise ValueError(f"Vector result {i} must be a 2D array")
            
            self.single_algorithm = len(results) == 1
            self.algorithm_names = list(self.results_dict.keys())
            
        else:
            # Handle scalar results
            self.results_dict = {'Algorithm': np.array(results)}
            self.vector_results = {}
            self.is_vector_optimization = False
            self.single_algorithm = True
            self.algorithm_names = ['Algorithm']
    
    def descriptive_statistics(self, algorithm: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics.
        
        Parameters:
        -----------
        algorithm : str, optional
            Specific algorithm name. If None, analyzes all algorithms.
            
        Returns:
        --------
        pd.DataFrame : Descriptive statistics summary
        """
        if algorithm and algorithm in self.results_dict:
            algorithms_to_analyze = [algorithm]
        else:
            algorithms_to_analyze = self.algorithm_names
        
        stats_data = []
        
        for alg_name in algorithms_to_analyze:
            data = self.results_dict[alg_name]
            
            stats_row = {
                'Algorithm': alg_name,
                'Count': len(data),
                'Mean': np.mean(data),
                'Median': np.median(data),
                'Std': np.std(data, ddof=1),
                'Min': np.min(data),
                'Max': np.max(data),
                'Q1': np.percentile(data, 25),
                'Q3': np.percentile(data, 75),
                'IQR': np.percentile(data, 75) - np.percentile(data, 25),
                'CV': np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else np.inf,
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            }
            stats_data.append(stats_row)
        
        return pd.DataFrame(stats_data)
    
    def normality_tests(self, algorithm: Optional[str] = None, alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform multiple normality tests on the results.
        
        Parameters:
        -----------
        algorithm : str, optional
            Specific algorithm name. If None, tests all algorithms.
        alpha : float
            Significance level for tests (default: 0.05)
            
        Returns:
        --------
        pd.DataFrame : Results of normality tests
        """
        if algorithm and algorithm in self.results_dict:
            algorithms_to_analyze = [algorithm]
        else:
            algorithms_to_analyze = self.algorithm_names
        
        test_results = []
        
        for alg_name in algorithms_to_analyze:
            data = self.results_dict[alg_name]
            
            # Shapiro-Wilk test (best for small samples)
            if len(data) <= 5000:  # Shapiro-Wilk limitation
                shapiro_stat, shapiro_p = shapiro(data)
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            # D'Agostino and Pearson's test
            dagostino_stat, dagostino_p = normaltest(data)
            
            # Anderson-Darling test
            anderson_result = anderson(data, dist='norm')
            anderson_critical = anderson_result.critical_values[2]  # 5% significance level
            anderson_is_normal = anderson_result.statistic < anderson_critical
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
            
            test_row = {
                'Algorithm': alg_name,
                'Shapiro_Stat': shapiro_stat,
                'Shapiro_p': shapiro_p,
                'Shapiro_Normal': shapiro_p > alpha if not np.isnan(shapiro_p) else 'N/A',
                'DAgostino_Stat': dagostino_stat,
                'DAgostino_p': dagostino_p,
                'DAgostino_Normal': dagostino_p > alpha,
                'Anderson_Stat': anderson_result.statistic,
                'Anderson_Normal': anderson_is_normal,
                'KS_Stat': ks_stat,
                'KS_p': ks_p,
                'KS_Normal': ks_p > alpha
            }
            test_results.append(test_row)
        
        return pd.DataFrame(test_results)
    
    def success_rate_analysis(self, target_value: float, 
                            tolerance: float = 0.0, 
                            minimize: bool = True) -> pd.DataFrame:
        """
        Calculate success rates based on achieving target performance.
        
        Parameters:
        -----------
        target_value : float
            Target objective function value
        tolerance : float
            Tolerance around target value
        minimize : bool
            Whether it's a minimization problem
            
        Returns:
        --------
        pd.DataFrame : Success rate analysis
        """
        success_data = []
        
        for alg_name in self.algorithm_names:
            data = self.results_dict[alg_name]
            
            if minimize:
                successes = np.sum(data <= target_value + tolerance)
            else:
                successes = np.sum(data >= target_value - tolerance)
            
            total_runs = len(data)
            success_rate = successes / total_runs
            
            # Confidence interval for success rate (Wilson score interval)
            n = total_runs
            p = success_rate
            z = 1.96  # 95% confidence
            denominator = 1 + z**2/n
            centre = (p + z**2/(2*n)) / denominator
            half_width = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
            
            ci_lower = centre - half_width
            ci_upper = centre + half_width
            
            success_row = {
                'Algorithm': alg_name,
                'Total_Runs': total_runs,
                'Successes': successes,
                'Success_Rate': success_rate,
                'CI_Lower_95': ci_lower,
                'CI_Upper_95': ci_upper
            }
            success_data.append(success_row)
        
        return pd.DataFrame(success_data)
    
    def compare_algorithms(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Compare multiple algorithms using statistical tests.
        
        Parameters:
        -----------
        alpha : float
            Significance level for tests
            
        Returns:
        --------
        pd.DataFrame : Pairwise comparison results
        """
        if self.single_algorithm:
            print("Only one algorithm provided. Cannot perform comparison.")
            return pd.DataFrame()
        
        comparisons = []
        algorithms = self.algorithm_names
        
        for i in range(len(algorithms)):
            for j in range(i+1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                data1, data2 = self.results_dict[alg1], self.results_dict[alg2]
                
                # Mann-Whitney U test (non-parametric)
                mw_stat, mw_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Welch's t-test (doesn't assume equal variances)
                t_stat, t_p = stats.ttest_ind(data1, data2, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                    (len(data2)-1)*np.var(data2, ddof=1)) / 
                                   (len(data1) + len(data2) - 2))
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                
                comparison_row = {
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Mean_Diff': np.mean(data1) - np.mean(data2),
                    'MannWhitney_Stat': mw_stat,
                    'MannWhitney_p': mw_p,
                    'MannWhitney_Significant': mw_p < alpha,
                    'TTest_Stat': t_stat,
                    'TTest_p': t_p,
                    'TTest_Significant': t_p < alpha,
                    'Cohens_d': cohens_d,
                    'Effect_Size': self._interpret_cohens_d(abs(cohens_d))
                }
                comparisons.append(comparison_row)
        
        return pd.DataFrame(comparisons)
    
    def vector_descriptive_statistics(self, algorithm: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate descriptive statistics for vector optimization results.
        
        Parameters:
        -----------
        algorithm : str, optional
            Specific algorithm name. If None, analyzes all algorithms.
            
        Returns:
        --------
        pd.DataFrame : Vector descriptive statistics summary
        """
        if not self.is_vector_optimization:
            print("No vector results available. Use descriptive_statistics() for scalar results.")
            return pd.DataFrame()
        
        if algorithm and algorithm in self.vector_results:
            algorithms_to_analyze = [algorithm]
        else:
            algorithms_to_analyze = self.algorithm_names
        
        stats_data = []
        
        for alg_name in algorithms_to_analyze:
            if alg_name not in self.vector_results:
                continue
                
            vectors = self.vector_results[alg_name]  # Shape: (n_runs, n_dimensions)
            n_runs, n_dims = vectors.shape
            
            # Calculate statistics for each dimension
            for dim in range(n_dims):
                dim_data = vectors[:, dim]
                
                stats_row = {
                    'Algorithm': alg_name,
                    'Dimension': f'Dim_{dim+1}',
                    'Count': n_runs,
                    'Mean': np.mean(dim_data),
                    'Median': np.median(dim_data),
                    'Std': np.std(dim_data, ddof=1),
                    'Min': np.min(dim_data),
                    'Max': np.max(dim_data),
                    'Q1': np.percentile(dim_data, 25),
                    'Q3': np.percentile(dim_data, 75),
                    'IQR': np.percentile(dim_data, 75) - np.percentile(dim_data, 25),
                    'CV': np.std(dim_data, ddof=1) / np.mean(dim_data) if np.mean(dim_data) != 0 else np.inf,
                    'Skewness': stats.skew(dim_data),
                    'Kurtosis': stats.kurtosis(dim_data)
                }
                stats_data.append(stats_row)
            
            # Overall vector statistics
            vector_norms = np.linalg.norm(vectors, axis=1)
            vector_means = np.mean(vectors, axis=0)
            vector_std = np.std(vectors, axis=0)
            
            stats_row = {
                'Algorithm': alg_name,
                'Dimension': 'Overall',
                'Count': n_runs,
                'Mean': np.mean(vector_norms),
                'Median': np.median(vector_norms),
                'Std': np.std(vector_norms, ddof=1),
                'Min': np.min(vector_norms),
                'Max': np.max(vector_norms),
                'Q1': np.percentile(vector_norms, 25),
                'Q3': np.percentile(vector_norms, 75),
                'IQR': np.percentile(vector_norms, 75) - np.percentile(vector_norms, 25),
                'CV': np.std(vector_norms, ddof=1) / np.mean(vector_norms) if np.mean(vector_norms) != 0 else np.inf,
                'Skewness': stats.skew(vector_norms),
                'Kurtosis': stats.kurtosis(vector_norms)
            }
            stats_data.append(stats_row)
        
        return pd.DataFrame(stats_data)
    
    def vector_convergence_analysis(self, algorithm: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze convergence patterns in vector optimization results.
        
        Parameters:
        -----------
        algorithm : str, optional
            Specific algorithm name. If None, analyzes all algorithms.
            
        Returns:
        --------
        pd.DataFrame : Convergence analysis results
        """
        if not self.is_vector_optimization:
            print("No vector results available.")
            return pd.DataFrame()
        
        if algorithm and algorithm in self.vector_results:
            algorithms_to_analyze = [algorithm]
        else:
            algorithms_to_analyze = self.algorithm_names
        
        convergence_data = []
        
        for alg_name in algorithms_to_analyze:
            if alg_name not in self.vector_results:
                continue
                
            vectors = self.vector_results[alg_name]
            n_runs, n_dims = vectors.shape
            
            # Calculate convergence metrics
            vector_norms = np.linalg.norm(vectors, axis=1)
            
            # Distance from best solution
            best_idx = np.argmin(vector_norms)
            best_vector = vectors[best_idx]
            distances_from_best = [np.linalg.norm(vec - best_vector) for vec in vectors]
            
            # Convergence rate (improvement over iterations)
            improvements = np.diff(vector_norms)
            convergence_rate = np.mean(improvements[improvements < 0]) if np.any(improvements < 0) else 0
            
            # Stability (variance in final solutions)
            final_quarter = int(0.75 * n_runs)
            final_vectors = vectors[final_quarter:]
            final_norms = vector_norms[final_quarter:]
            stability = np.std(final_norms)
            
            # Diversity (spread of solutions)
            diversity = np.mean([np.linalg.norm(vec - np.mean(vectors, axis=0)) for vec in vectors])
            
            convergence_row = {
                'Algorithm': alg_name,
                'Total_Runs': n_runs,
                'Dimensions': n_dims,
                'Best_Value': np.min(vector_norms),
                'Worst_Value': np.max(vector_norms),
                'Mean_Distance_From_Best': np.mean(distances_from_best),
                'Max_Distance_From_Best': np.max(distances_from_best),
                'Convergence_Rate': convergence_rate,
                'Stability': stability,
                'Diversity': diversity,
                'Improvement_Ratio': (np.min(vector_norms) - np.max(vector_norms)) / np.max(vector_norms)
            }
            convergence_data.append(convergence_row)
        
        return pd.DataFrame(convergence_data)
    
    def vector_algorithm_comparison(self, alpha: float = 0.05, 
                                  distance_metric: str = 'euclidean',
                                  standardize: bool = True) -> pd.DataFrame:
        """
        Compare vector optimization algorithms using multi-dimensional statistical tests and distance metrics.
        
        Parameters:
        -----------
        alpha : float
            Significance level for tests
        distance_metric : str
            Distance metric for vector comparison ('euclidean', 'manhattan', 'cosine')
        standardize : bool
            Whether to standardize vector components before comparison
            
        Returns:
        --------
        pd.DataFrame : Vector algorithm comparison results
        """
        if not self.is_vector_optimization:
            print("No vector results available for comparison.")
            return pd.DataFrame()
        
        if self.single_algorithm:
            print("Only one algorithm provided. Cannot perform comparison.")
            return pd.DataFrame()
        
        comparisons = []
        algorithms = self.algorithm_names
        
        for i in range(len(algorithms)):
            for j in range(i+1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                
                if alg1 not in self.vector_results or alg2 not in self.vector_results:
                    continue
                
                vectors1 = self.vector_results[alg1].copy()
                vectors2 = self.vector_results[alg2].copy()
                
                # Ensure same dimensionality
                if vectors1.shape[1] != vectors2.shape[1]:
                    print(f"Warning: {alg1} and {alg2} have different dimensions. Skipping comparison.")
                    continue
                
                # Standardize if requested
                if standardize:
                    # Standardize each dimension separately
                    for dim in range(vectors1.shape[1]):
                        all_values = np.concatenate([vectors1[:, dim], vectors2[:, dim]])
                        mean_val = np.mean(all_values)
                        std_val = np.std(all_values)
                        if std_val > 0:
                            vectors1[:, dim] = (vectors1[:, dim] - mean_val) / std_val
                            vectors2[:, dim] = (vectors2[:, dim] - mean_val) / std_val
                
                # Distance-based comparisons
                distance_metrics = self._calculate_distance_metrics(vectors1, vectors2, distance_metric)
                
                # Statistical comparisons (flattened)
                flat1 = vectors1.flatten()
                flat2 = vectors2.flatten()
                
                # Mann-Whitney U test
                mw_stat, mw_p = stats.mannwhitneyu(flat1, flat2, alternative='two-sided')
                
                # Welch's t-test
                t_stat, t_p = stats.ttest_ind(flat1, flat2, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(flat1)-1)*np.var(flat1, ddof=1) + 
                                    (len(flat2)-1)*np.var(flat2, ddof=1)) / 
                                   (len(flat1) + len(flat2) - 2))
                cohens_d = (np.mean(flat1) - np.mean(flat2)) / pooled_std
                
                # Vector-specific metrics
                norms1 = np.linalg.norm(vectors1, axis=1)
                norms2 = np.linalg.norm(vectors2, axis=1)
                
                # Best solution comparison
                best1, best2 = np.min(norms1), np.min(norms2)
                best_diff = best1 - best2
                
                # Mean solution comparison
                mean1, mean2 = np.mean(norms1), np.mean(norms2)
                mean_diff = mean1 - mean2
                
                # Convergence comparison
                conv1 = np.std(norms1[-len(norms1)//4:])  # Last quarter stability
                conv2 = np.std(norms2[-len(norms2)//4:])
                conv_diff = conv1 - conv2
                
                # Component-wise analysis
                component_analysis = self._analyze_vector_components(vectors1, vectors2)
                
                comparison_row = {
                    'Algorithm_1': alg1,
                    'Algorithm_2': alg2,
                    'Dimensions': vectors1.shape[1],
                    'Distance_Metric': distance_metric,
                    'Standardized': standardize,
                    'Best_Diff': best_diff,
                    'Mean_Diff': mean_diff,
                    'Convergence_Diff': conv_diff,
                    'Mean_Distance': distance_metrics['mean_distance'],
                    'Min_Distance': distance_metrics['min_distance'],
                    'Max_Distance': distance_metrics['max_distance'],
                    'Cosine_Similarity': distance_metrics['cosine_similarity'],
                    'Component_Variance_Ratio': component_analysis['variance_ratio'],
                    'Most_Variable_Dimension': component_analysis['most_variable_dim'],
                    'MannWhitney_Stat': mw_stat,
                    'MannWhitney_p': mw_p,
                    'MannWhitney_Significant': mw_p < alpha,
                    'TTest_Stat': t_stat,
                    'TTest_p': t_p,
                    'TTest_Significant': t_p < alpha,
                    'Cohens_d': cohens_d,
                    'Effect_Size': self._interpret_cohens_d(abs(cohens_d))
                }
                comparisons.append(comparison_row)
        
        return pd.DataFrame(comparisons)
    
    def _calculate_distance_metrics(self, vectors1: np.ndarray, vectors2: np.ndarray, 
                                  metric: str = 'euclidean') -> dict:
        """Calculate various distance metrics between vector sets."""
        metrics = {}
        
        # Calculate distances between all pairs
        distances = []
        for v1 in vectors1:
            for v2 in vectors2:
                if metric == 'euclidean':
                    dist = np.linalg.norm(v1 - v2)
                elif metric == 'manhattan':
                    dist = np.sum(np.abs(v1 - v2))
                elif metric == 'cosine':
                    # Cosine distance = 1 - cosine similarity
                    dot_product = np.dot(v1, v2)
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        cosine_sim = dot_product / (norm1 * norm2)
                        dist = 1 - cosine_sim
                    else:
                        dist = 1
                else:
                    raise ValueError(f"Unknown distance metric: {metric}")
                distances.append(dist)
        
        distances = np.array(distances)
        
        metrics['mean_distance'] = np.mean(distances)
        metrics['min_distance'] = np.min(distances)
        metrics['max_distance'] = np.max(distances)
        
        # Cosine similarity (separate from distance)
        if metric == 'cosine':
            metrics['cosine_similarity'] = 1 - metrics['mean_distance']
        else:
            # Calculate cosine similarity separately
            mean1 = np.mean(vectors1, axis=0)
            mean2 = np.mean(vectors2, axis=0)
            dot_product = np.dot(mean1, mean2)
            norm1 = np.linalg.norm(mean1)
            norm2 = np.linalg.norm(mean2)
            if norm1 > 0 and norm2 > 0:
                metrics['cosine_similarity'] = dot_product / (norm1 * norm2)
            else:
                metrics['cosine_similarity'] = 0
        
        return metrics
    
    def _analyze_vector_components(self, vectors1: np.ndarray, vectors2: np.ndarray) -> dict:
        """Analyze which vector components vary most between algorithms."""
        analysis = {}
        
        # Calculate variance for each dimension across both algorithms
        all_vectors = np.vstack([vectors1, vectors2])
        variances = np.var(all_vectors, axis=0)
        
        # Find most variable dimension
        most_variable_dim = np.argmax(variances)
        analysis['most_variable_dim'] = most_variable_dim
        
        # Calculate variance ratio (max/min)
        if np.min(variances) > 0:
            analysis['variance_ratio'] = np.max(variances) / np.min(variances)
        else:
            analysis['variance_ratio'] = np.inf
        
        return analysis
    
    def vector_aggregation_analysis(self, weights: Optional[np.ndarray] = None, 
                                  aggregation_method: str = 'norm') -> pd.DataFrame:
        """
        Analyze vector results using different aggregation approaches.
        
        Parameters:
        -----------
        weights : np.ndarray, optional
            Weights for each dimension. If None, equal weights are used.
        aggregation_method : str
            Method for aggregation ('norm', 'weighted_sum', 'custom')
            
        Returns:
        --------
        pd.DataFrame : Aggregation analysis results
        """
        if not self.is_vector_optimization:
            print("No vector results available for aggregation analysis.")
            return pd.DataFrame()
        
        aggregation_data = []
        
        for alg_name in self.algorithm_names:
            if alg_name not in self.vector_results:
                continue
                
            vectors = self.vector_results[alg_name]
            n_runs, n_dims = vectors.shape
            
            # Set default weights if not provided
            if weights is None:
                weights = np.ones(n_dims) / n_dims
            elif len(weights) != n_dims:
                print(f"Warning: Weight length {len(weights)} doesn't match dimensions {n_dims}. Using equal weights.")
                weights = np.ones(n_dims) / n_dims
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Calculate different aggregation metrics
            if aggregation_method == 'norm':
                # L2 norm (Euclidean distance from origin)
                aggregated_values = np.linalg.norm(vectors, axis=1)
            elif aggregation_method == 'weighted_sum':
                # Weighted sum of components
                aggregated_values = np.sum(vectors * weights, axis=1)
            elif aggregation_method == 'custom':
                # Custom: weighted sum with sign consideration
                aggregated_values = np.sum(vectors * weights, axis=1)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            # Calculate statistics for aggregated values
            stats_row = {
                'Algorithm': alg_name,
                'Aggregation_Method': aggregation_method,
                'Count': n_runs,
                'Mean': np.mean(aggregated_values),
                'Median': np.median(aggregated_values),
                'Std': np.std(aggregated_values, ddof=1),
                'Min': np.min(aggregated_values),
                'Max': np.max(aggregated_values),
                'Q1': np.percentile(aggregated_values, 25),
                'Q3': np.percentile(aggregated_values, 75),
                'CV': np.std(aggregated_values, ddof=1) / np.mean(aggregated_values) if np.mean(aggregated_values) != 0 else np.inf,
                'Skewness': stats.skew(aggregated_values),
                'Kurtosis': stats.kurtosis(aggregated_values)
            }
            
            # Add component-wise analysis
            for dim in range(n_dims):
                dim_data = vectors[:, dim]
                stats_row[f'Dim_{dim+1}_Mean'] = np.mean(dim_data)
                stats_row[f'Dim_{dim+1}_Std'] = np.std(dim_data, ddof=1)
                stats_row[f'Dim_{dim+1}_Weight'] = weights[dim]
            
            aggregation_data.append(stats_row)
        
        return pd.DataFrame(aggregation_data)
    
    def plot_vector_distributions(self, figsize: tuple = (20, 15), 
                                use_pca: bool = True, 
                                use_parallel_coords: bool = True) -> plt.Figure:
        """
        Create comprehensive visualization of vector optimization results with advanced techniques.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        use_pca : bool
            Whether to include PCA dimensionality reduction plots
        use_parallel_coords : bool
            Whether to include parallel coordinate plots for high-dimensional data
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        if not self.is_vector_optimization:
            print("No vector results available. Use plot_distributions() for scalar results.")
            return None
        
        n_algorithms = len(self.algorithm_names)
        n_dims = self.vector_results[self.algorithm_names[0]].shape[1]
        
        # Determine subplot layout based on available features
        if use_pca and use_parallel_coords and n_dims > 2:
            rows, cols = 4, 3
        elif use_pca or use_parallel_coords:
            rows, cols = 3, 3
        else:
            rows, cols = 3, 3
        
        # Create subplots
        fig = plt.figure(figsize=figsize)
        
        # 1. Vector norms distribution
        ax1 = plt.subplot(rows, cols, 1)
        for alg_name in self.algorithm_names:
            if alg_name in self.vector_results:
                norms = np.linalg.norm(self.vector_results[alg_name], axis=1)
                ax1.hist(norms, alpha=0.6, density=True, label=f'{alg_name} (n={len(norms)})', bins=20)
        ax1.set_title('Distribution of Vector Norms')
        ax1.set_xlabel('Vector Norm')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot of vector norms
        ax2 = plt.subplot(rows, cols, 2)
        box_data = [np.linalg.norm(self.vector_results[alg], axis=1) for alg in self.algorithm_names if alg in self.vector_results]
        box_plot = ax2.boxplot(box_data, labels=[alg for alg in self.algorithm_names if alg in self.vector_results], patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        ax2.set_title('Box Plot of Vector Norms')
        ax2.set_ylabel('Vector Norm')
        ax2.grid(True, alpha=0.3)
        
        # 3. Convergence plots for each algorithm
        ax3 = plt.subplot(rows, cols, 3)
        for alg_name in self.algorithm_names:
            if alg_name in self.vector_results:
                norms = np.linalg.norm(self.vector_results[alg_name], axis=1)
                ax3.plot(norms, alpha=0.7, label=alg_name)
        ax3.set_title('Convergence of Vector Norms')
        ax3.set_xlabel('Run Number')
        ax3.set_ylabel('Vector Norm')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4-6. Individual dimension distributions (first 3 dimensions)
        for dim in range(min(3, n_dims)):
            ax = plt.subplot(rows, cols, 4 + dim)
            for alg_name in self.algorithm_names:
                if alg_name in self.vector_results:
                    dim_data = self.vector_results[alg_name][:, dim]
                    ax.hist(dim_data, alpha=0.6, density=True, label=f'{alg_name}', bins=15)
            ax.set_title(f'Distribution of Dimension {dim+1}')
            ax.set_xlabel(f'Value (Dim {dim+1})')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 7. 2D scatter plot (first two dimensions)
        ax7 = plt.subplot(rows, cols, 7)
        for alg_name in self.algorithm_names:
            if alg_name in self.vector_results and self.vector_results[alg_name].shape[1] >= 2:
                vectors = self.vector_results[alg_name]
                ax7.scatter(vectors[:, 0], vectors[:, 1], alpha=0.6, label=alg_name, s=20)
        ax7.set_title('2D Scatter Plot (First Two Dimensions)')
        ax7.set_xlabel('Dimension 1')
        ax7.set_ylabel('Dimension 2')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Heatmap of mean values by algorithm and dimension
        ax8 = plt.subplot(rows, cols, 8)
        mean_matrix = []
        for alg_name in self.algorithm_names:
            if alg_name in self.vector_results:
                mean_vector = np.mean(self.vector_results[alg_name], axis=0)
                mean_matrix.append(mean_vector)
        
        if mean_matrix:
            mean_matrix = np.array(mean_matrix)
            im = ax8.imshow(mean_matrix, cmap='viridis', aspect='auto')
            ax8.set_title('Mean Values Heatmap')
            ax8.set_xlabel('Dimension')
            ax8.set_ylabel('Algorithm')
            ax8.set_yticks(range(len(self.algorithm_names)))
            ax8.set_yticklabels(self.algorithm_names)
            plt.colorbar(im, ax=ax8)
        
        # 9. Diversity analysis
        ax9 = plt.subplot(rows, cols, 9)
        diversity_data = []
        for alg_name in self.algorithm_names:
            if alg_name in self.vector_results:
                vectors = self.vector_results[alg_name]
                center = np.mean(vectors, axis=0)
                distances = [np.linalg.norm(vec - center) for vec in vectors]
                diversity_data.append(distances)
        
        if diversity_data:
            ax9.boxplot(diversity_data, labels=[alg for alg in self.algorithm_names if alg in self.vector_results])
            ax9.set_title('Solution Diversity (Distance from Center)')
            ax9.set_ylabel('Distance from Center')
            ax9.grid(True, alpha=0.3)
        
        # Additional plots for high-dimensional data
        plot_idx = 10
        
        # 10. PCA plot (if requested and dimensions > 2)
        if use_pca and n_dims > 2:
            try:
                from sklearn.decomposition import PCA
                ax_pca = plt.subplot(rows, cols, plot_idx)
                
                # Combine all vectors and apply PCA
                all_vectors = []
                labels = []
                for alg_name in self.algorithm_names:
                    if alg_name in self.vector_results:
                        vectors = self.vector_results[alg_name]
                        all_vectors.append(vectors)
                        labels.extend([alg_name] * len(vectors))
                
                all_vectors = np.vstack(all_vectors)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(all_vectors)
                
                # Plot PCA results
                for alg_name in self.algorithm_names:
                    if alg_name in self.vector_results:
                        mask = np.array(labels) == alg_name
                        ax_pca.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                                     alpha=0.6, label=alg_name, s=20)
                
                ax_pca.set_title(f'PCA Plot (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
                ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax_pca.legend()
                ax_pca.grid(True, alpha=0.3)
                plot_idx += 1
                
            except ImportError:
                print("Warning: sklearn not available for PCA. Install scikit-learn for PCA plots.")
        
        # 11. Parallel coordinate plot (if requested and dimensions > 2)
        if use_parallel_coords and n_dims > 2:
            ax_parallel = plt.subplot(rows, cols, plot_idx)
            
            # Sample a subset of points for clarity
            max_points = 50
            for alg_name in self.algorithm_names:
                if alg_name in self.vector_results:
                    vectors = self.vector_results[alg_name]
                    if len(vectors) > max_points:
                        # Sample random points
                        indices = np.random.choice(len(vectors), max_points, replace=False)
                        vectors = vectors[indices]
                    
                    # Plot parallel coordinates
                    for i, vector in enumerate(vectors):
                        alpha = 0.1 if len(vectors) > 10 else 0.3
                        ax_parallel.plot(range(n_dims), vector, alpha=alpha, 
                                       color=plt.cm.Set1(list(self.algorithm_names).index(alg_name) / len(self.algorithm_names)))
            
            ax_parallel.set_title('Parallel Coordinate Plot')
            ax_parallel.set_xlabel('Dimension')
            ax_parallel.set_ylabel('Value')
            ax_parallel.set_xticks(range(n_dims))
            ax_parallel.set_xticklabels([f'Dim {i+1}' for i in range(n_dims)])
            ax_parallel.grid(True, alpha=0.3)
            
            # Add legend
            handles = [plt.Line2D([0], [0], color=plt.cm.Set1(i / len(self.algorithm_names)), 
                                label=alg_name) for i, alg_name in enumerate(self.algorithm_names)]
            ax_parallel.legend(handles=handles)
            plot_idx += 1
        
        # 12. Distance matrix heatmap
        if plot_idx <= rows * cols:
            ax_dist = plt.subplot(rows, cols, plot_idx)
            
            # Calculate pairwise distances between algorithm means
            mean_vectors = []
            alg_names = []
            for alg_name in self.algorithm_names:
                if alg_name in self.vector_results:
                    mean_vector = np.mean(self.vector_results[alg_name], axis=0)
                    mean_vectors.append(mean_vector)
                    alg_names.append(alg_name)
            
            if len(mean_vectors) > 1:
                mean_vectors = np.array(mean_vectors)
                distance_matrix = np.zeros((len(mean_vectors), len(mean_vectors)))
                
                for i in range(len(mean_vectors)):
                    for j in range(len(mean_vectors)):
                        distance_matrix[i, j] = np.linalg.norm(mean_vectors[i] - mean_vectors[j])
                
                im = ax_dist.imshow(distance_matrix, cmap='viridis', aspect='auto')
                ax_dist.set_title('Distance Matrix Between Algorithm Means')
                ax_dist.set_xticks(range(len(alg_names)))
                ax_dist.set_yticks(range(len(alg_names)))
                ax_dist.set_xticklabels(alg_names, rotation=45)
                ax_dist.set_yticklabels(alg_names)
                plt.colorbar(im, ax=ax_dist)
        
        plt.tight_layout()
        return fig
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return 'Negligible'
        elif d < 0.5:
            return 'Small'
        elif d < 0.8:
            return 'Medium'
        else:
            return 'Large'
    
    def plot_distributions(self, figsize: tuple = (15, 10)) -> plt.Figure:
        """
        Create comprehensive visualization of the distributions.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        n_algorithms = len(self.algorithm_names)
        
        if n_algorithms == 1:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        
        # 1. Histogram with density curve
        ax1 = axes[0]
        for alg_name in self.algorithm_names:
            data = self.results_dict[alg_name]
            ax1.hist(data, alpha=0.6, density=True, label=f'{alg_name} (n={len(data)})', bins=20)
            
            # Add density curve
            x = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            ax1.plot(x, kde(x), linewidth=2)
        
        ax1.set_title('Distribution of Optimization Results')
        ax1.set_xlabel('Objective Function Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2 = axes[1]
        box_data = [self.results_dict[alg] for alg in self.algorithm_names]
        box_plot = ax2.boxplot(box_data, labels=self.algorithm_names, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title('Box Plot of Results')
        ax2.set_ylabel('Objective Function Value')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot for normality (first algorithm only)
        ax3 = axes[2]
        first_alg = self.algorithm_names[0]
        data = self.results_dict[first_alg]
        stats.probplot(data, dist="norm", plot=ax3)
        ax3.set_title(f'Q-Q Plot for Normality - {first_alg}')
        ax3.grid(True, alpha=0.3)
        
        # 4. Violin plot (if multiple algorithms) or convergence plot
        ax4 = axes[3]
        if n_algorithms > 1:
            violin_data = [self.results_dict[alg] for alg in self.algorithm_names]
            parts = ax4.violinplot(violin_data, positions=range(1, n_algorithms+1))
            ax4.set_xticks(range(1, n_algorithms+1))
            ax4.set_xticklabels(self.algorithm_names)
            ax4.set_title('Violin Plot of Results')
            ax4.set_ylabel('Objective Function Value')
        else:
            # Simple statistics plot for single algorithm
            data = self.results_dict[first_alg]
            run_numbers = range(1, len(data) + 1)
            ax4.plot(run_numbers, data, 'o-', alpha=0.7)
            ax4.axhline(y=np.mean(data), color='r', linestyle='--', label=f'Mean: {np.mean(data):.4f}')
            ax4.axhline(y=np.median(data), color='g', linestyle='--', label=f'Median: {np.median(data):.4f}')
            ax4.set_title('Results by Run Number')
            ax4.set_xlabel('Run Number')
            ax4.set_ylabel('Objective Function Value')
            ax4.legend()
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, target_value: Optional[float] = None, 
                       tolerance: float = 0.0, minimize: bool = True) -> str:
        """
        Generate a comprehensive text report of the analysis.
        
        Parameters:
        -----------
        target_value : float, optional
            Target value for success rate analysis
        tolerance : float
            Tolerance for success rate analysis
        minimize : bool
            Whether it's a minimization problem
            
        Returns:
        --------
        str : Formatted analysis report
        """
        report = []
        report.append("=" * 60)
        report.append("OPTIMIZATION RESULTS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Check if this is vector optimization
        if self.is_vector_optimization:
            report.append("VECTOR OPTIMIZATION ANALYSIS")
            report.append("=" * 40)
            report.append("")
            
            # Vector Descriptive Statistics
            report.append("VECTOR DESCRIPTIVE STATISTICS")
            report.append("-" * 30)
            vector_desc_stats = self.vector_descriptive_statistics()
            report.append(vector_desc_stats.to_string(index=False))
            report.append("")
            
            # Vector Convergence Analysis
            report.append("VECTOR CONVERGENCE ANALYSIS")
            report.append("-" * 30)
            convergence_stats = self.vector_convergence_analysis()
            report.append(convergence_stats.to_string(index=False))
            report.append("")
            
            # Vector Algorithm Comparison (if multiple algorithms)
            if not self.single_algorithm:
                report.append("VECTOR ALGORITHM COMPARISON")
                report.append("-" * 30)
                vector_comparisons = self.vector_algorithm_comparison()
                report.append(vector_comparisons.to_string(index=False))
                report.append("")
        
        # Scalar Descriptive Statistics
        report.append("SCALAR DESCRIPTIVE STATISTICS")
        report.append("-" * 30)
        desc_stats = self.descriptive_statistics()
        report.append(desc_stats.to_string(index=False))
        report.append("")
        
        # Normality Tests
        report.append("NORMALITY TESTS")
        report.append("-" * 30)
        norm_tests = self.normality_tests()
        report.append(norm_tests.to_string(index=False))
        report.append("")
        
        # Success Rate Analysis (if target provided)
        if target_value is not None:
            report.append("SUCCESS RATE ANALYSIS")
            report.append("-" * 30)
            success_rates = self.success_rate_analysis(target_value, tolerance, minimize)
            report.append(success_rates.to_string(index=False))
            report.append("")
        
        # Algorithm Comparison (if multiple algorithms)
        if not self.single_algorithm:
            report.append("SCALAR ALGORITHM COMPARISON")
            report.append("-" * 30)
            comparisons = self.compare_algorithms()
            report.append(comparisons.to_string(index=False))
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        desc_stats = self.descriptive_statistics()
        for _, row in desc_stats.iterrows():
            alg_name = row['Algorithm']
            cv = row['CV']
            
            if cv > 0.5:
                report.append(f"• {alg_name}: High variability (CV={cv:.3f}). Consider more runs or parameter tuning.")
            elif cv < 0.1:
                report.append(f"• {alg_name}: Very consistent results (CV={cv:.3f}). Good stability.")
            else:
                report.append(f"• {alg_name}: Moderate variability (CV={cv:.3f}). Acceptable performance.")
        
        # Vector-specific recommendations
        if self.is_vector_optimization:
            report.append("")
            report.append("VECTOR-SPECIFIC RECOMMENDATIONS")
            report.append("-" * 30)
            convergence_stats = self.vector_convergence_analysis()
            for _, row in convergence_stats.iterrows():
                alg_name = row['Algorithm']
                diversity = row['Diversity']
                stability = row['Stability']
                
                if diversity > np.mean(convergence_stats['Diversity']) * 1.5:
                    report.append(f"• {alg_name}: High solution diversity (diversity={diversity:.3f}). Good exploration.")
                elif diversity < np.mean(convergence_stats['Diversity']) * 0.5:
                    report.append(f"• {alg_name}: Low solution diversity (diversity={diversity:.3f}). May need more exploration.")
                
                if stability > np.mean(convergence_stats['Stability']) * 1.5:
                    report.append(f"• {alg_name}: High variability in final solutions (stability={stability:.3f}). Consider more runs.")
                elif stability < np.mean(convergence_stats['Stability']) * 0.5:
                    report.append(f"• {alg_name}: Very stable final solutions (stability={stability:.3f}). Good convergence.")
        
        return "\n".join(report)

# Example usage and demonstration
def demo_optimization_analyzer():
    """Demonstrate the OptimizationAnalyzer with single algorithm multiple vector runs."""
    
    # Generate sample optimization results for demonstration
    np.random.seed(42)
    
    print("=" * 80)
    print("SINGLE ALGORITHM MULTIPLE VECTOR RUNS ANALYSIS")
    print("=" * 80)
    print()
    
    # Single algorithm with multiple vector runs example
    # This simulates multiple runs/configurations of the same algorithm
    # Each "run" represents a different configuration or independent execution
    
    # Generate multiple vector results from the same algorithm (e.g., different parameter settings)
    single_algorithm_vector_runs = [
        np.random.normal([5, 3, 7], [1, 0.8, 1.2], (30, 3)),  # Run 1: Standard parameters
        np.random.normal([4.5, 2.8, 6.5], [0.8, 0.6, 1.0], (30, 3)),  # Run 2: Tuned parameters
        np.random.normal([5.2, 3.1, 7.1], [1.2, 1.0, 1.4], (30, 3)),  # Run 3: Different initialization
        np.random.normal([4.8, 2.9, 6.8], [0.9, 0.7, 1.1], (30, 3))   # Run 4: Alternative configuration
    ]
    
    # Create analyzer for single algorithm with multiple vector runs
    single_alg_vector_analyzer = OptimizationAnalyzer(single_algorithm_vector_runs)
    
    print("Creating analysis for single algorithm with multiple vector runs...")
    print(f"Algorithm: Genetic Algorithm with 4 different configurations")
    print(f"Each run has {single_algorithm_vector_runs[0].shape[0]} iterations and {single_algorithm_vector_runs[0].shape[1]} dimensions")
    print()
    
    # Vector descriptive statistics
    vector_desc_stats = single_alg_vector_analyzer.vector_descriptive_statistics()
    print("Vector Descriptive Statistics (by Run):")
    print(vector_desc_stats)
    print()
    
    # Vector convergence analysis
    convergence_stats = single_alg_vector_analyzer.vector_convergence_analysis()
    print("Vector Convergence Analysis (by Run):")
    print(convergence_stats)
    print()
    
    # Vector algorithm comparison (comparing different runs of the same algorithm)
    print("Vector Run Comparisons (Same Algorithm, Different Configurations):")
    print("Using Euclidean distance with standardization:")
    vector_comparisons = single_alg_vector_analyzer.vector_algorithm_comparison(
        distance_metric='euclidean', standardize=True)
    print(vector_comparisons)
    print()
    
    # Distance-based comparisons with different metrics
    print("Distance-based Comparisons with Different Metrics:")
    for metric in ['euclidean', 'manhattan', 'cosine']:
        print(f"\n{metric.upper()} Distance Analysis:")
        comp = single_alg_vector_analyzer.vector_algorithm_comparison(
            distance_metric=metric, standardize=True)
        if not comp.empty:
            print(f"Mean Distance: {comp['Mean_Distance'].iloc[0]:.4f}")
            print(f"Cosine Similarity: {comp['Cosine_Similarity'].iloc[0]:.4f}")
    print()
    
    # Aggregation analysis
    print("Vector Aggregation Analysis:")
    print("L2 Norm Aggregation:")
    norm_agg = single_alg_vector_analyzer.vector_aggregation_analysis(aggregation_method='norm')
    print(norm_agg[['Algorithm', 'Mean', 'Std', 'Min', 'Max']])
    print()
    
    print("Weighted Sum Aggregation (equal weights):")
    weighted_agg = single_alg_vector_analyzer.vector_aggregation_analysis(aggregation_method='weighted_sum')
    print(weighted_agg[['Algorithm', 'Mean', 'Std', 'Min', 'Max']])
    print()
    
    # Create enhanced vector visualizations
    print("Creating enhanced visualizations with PCA and parallel coordinates...")
    vector_fig = single_alg_vector_analyzer.plot_vector_distributions(
        figsize=(20, 15), use_pca=True, use_parallel_coords=True)
    plt.suptitle('Single Algorithm - Multiple Vector Runs Analysis (Enhanced)', fontsize=16, y=0.98)
    plt.show()
    
    # Generate comprehensive report
    print("=" * 80)
    print("COMPREHENSIVE REPORT")
    print("=" * 80)
    
    # Single algorithm vector report
    single_alg_vector_report = single_alg_vector_analyzer.generate_report(target_value=8.0, minimize=True)
    print(single_alg_vector_report)
    
    return single_alg_vector_analyzer

# Run demonstration if script is executed directly
if __name__ == "__main__":
    demo_analyzer = demo_optimization_analyzer()