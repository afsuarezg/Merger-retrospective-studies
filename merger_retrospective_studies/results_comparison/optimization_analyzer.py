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
    
    def __init__(self, results: Union[List[float], np.ndarray, Dict[str, List[float]]]):
        """
        Initialize the analyzer with optimization results.
        
        Parameters:
        -----------
        results : Union[List[float], np.ndarray, Dict[str, List[float]]]
            Single algorithm results as list/array, or multiple algorithms as dict
        """
        if isinstance(results, dict):
            self.results_dict = {name: np.array(res) for name, res in results.items()}
            self.single_algorithm = False
            self.algorithm_names = list(results.keys())
        else:
            self.results_dict = {'Algorithm': np.array(results)}
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
        # report.append()
        
        # Descriptive Statistics
        report.append("DESCRIPTIVE STATISTICS")
        report.append("-" * 30)
        desc_stats = self.descriptive_statistics()
        report.append(desc_stats.to_string(index=False))
        # report.append()
        
        # Normality Tests
        report.append("NORMALITY TESTS")
        report.append("-" * 30)
        norm_tests = self.normality_tests()
        report.append(norm_tests.to_string(index=False))
        # report.append()
        
        # Success Rate Analysis (if target provided)
        if target_value is not None:
            report.append("SUCCESS RATE ANALYSIS")
            report.append("-" * 30)
            success_rates = self.success_rate_analysis(target_value, tolerance, minimize)
            report.append(success_rates.to_string(index=False))
            # report.append()
        
        # Algorithm Comparison (if multiple algorithms)
        if not self.single_algorithm:
            report.append("ALGORITHM COMPARISON")
            report.append("-" * 30)
            comparisons = self.compare_algorithms()
            report.append(comparisons.to_string(index=False))
            #report.append()
        
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
        
        return "\n".join(report)

# Example usage and demonstration
def demo_optimization_analyzer():
    """Demonstrate the OptimizationAnalyzer with sample data."""
    
    # Generate sample optimization results for demonstration
    np.random.seed(42)
    
    # Simulate results from three different optimization algorithms
    algorithm_results = {
        'Genetic_Algorithm': np.random.normal(10, 2, 50),  # Mean=10, std=2
        # 'Particle_Swarm': np.random.normal(12, 1.5, 50),  # Mean=12, std=1.5
        # 'Simulated_Annealing': np.random.exponential(8, 50) + 5  # Exponential distribution
    }
    
    # Create analyzer
    analyzer = OptimizationAnalyzer(algorithm_results)
    
    # Generate comprehensive analysis
    print("Creating comprehensive optimization analysis...")
    
    # Basic descriptive statistics
    desc_stats = analyzer.descriptive_statistics()
    print("\nDescriptive Statistics:")
    print(desc_stats)
    
    # Normality tests
    norm_tests = analyzer.normality_tests()
    print("\nNormality Tests:")
    print(norm_tests)
    
    # Success rate analysis (target: achieve value <= 10)
    success_rates = analyzer.success_rate_analysis(target_value=10.0, minimize=True)
    print("\nSuccess Rate Analysis (Target ≤ 10):")
    print(success_rates)
    
    # Algorithm comparison
    comparisons = analyzer.compare_algorithms()
    print("\nAlgorithm Comparisons:")
    print(comparisons)
    
    # Generate full report
    report = analyzer.generate_report(target_value=10.0, minimize=True)
    print("\n" + report)
    
    # Create visualizations
    fig = analyzer.plot_distributions(figsize=(15, 10))
    plt.show()
    
    return analyzer

# Run demonstration if script is executed directly
if __name__ == "__main__":
    demo_analyzer = demo_optimization_analyzer()