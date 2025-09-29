"""
Vector Run Comparator Module

A comprehensive tool for comparing one-dimensional vectors resulting from 
different optimization algorithm runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Dict, Any, Optional, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings


class VectorComparator:
    """A comprehensive class for comparing multiple 1D vectors from optimization runs."""
    
    def __init__(self, vectors: List[Union[np.ndarray, List]], 
                 labels: Optional[List[str]] = None,
                 normalize: bool = False,
                 normalization_method: str = 'l2'):
        """Initialize the VectorComparator with multiple vectors."""
        if not vectors:
            raise ValueError("At least one vector must be provided")
        
        # Convert to numpy arrays and validate
        self.vectors = []
        for i, vec in enumerate(vectors):
            if isinstance(vec, list):
                vec = np.array(vec)
            elif not isinstance(vec, np.ndarray):
                raise ValueError(f"Vector {i} must be a numpy array or list")
            
            if vec.ndim != 1:
                raise ValueError(f"Vector {i} must be 1-dimensional")
            
            self.vectors.append(vec)
        
        # Check if all vectors have the same length
        lengths = [len(vec) for vec in self.vectors]
        if len(set(lengths)) > 1:
            warnings.warn(f"Vectors have different lengths: {lengths}. "
                         "Some comparisons may not be meaningful.")
        
        # Set labels
        if labels is None:
            self.labels = [f"Vector_{i}" for i in range(len(vectors))]
        else:
            if len(labels) != len(vectors):
                raise ValueError("Number of labels must match number of vectors")
            self.labels = labels
        
        # Normalize if requested
        if normalize:
            self.vectors = self._normalize_vectors(normalization_method)
        
        self.n_vectors = len(self.vectors)
        self.vector_length = len(self.vectors[0]) if self.vectors else 0
    
    def _normalize_vectors(self, method: str) -> List[np.ndarray]:
        """Normalize all vectors using the specified method."""
        normalized = []
        for vec in self.vectors:
            if method == 'l2':
                norm = np.linalg.norm(vec)
                normalized.append(vec / norm if norm > 0 else vec)
            elif method == 'standardize':
                mean, std = np.mean(vec), np.std(vec)
                normalized.append((vec - mean) / std if std > 0 else vec)
            elif method == 'minmax':
                min_val, max_val = np.min(vec), np.max(vec)
                normalized.append((vec - min_val) / (max_val - min_val) 
                                if max_val > min_val else vec)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        return normalized
    
    def magnitude_comparison(self) -> Dict[str, Any]:
        """Calculate and compare Euclidean norms (L2 norms) of all vectors."""
        norms = [np.linalg.norm(vec) for vec in self.vectors]
        rankings = np.argsort(norms)[::-1]  # Descending order
        
        return {
            'norms': dict(zip(self.labels, norms)),
            'rankings': dict(zip(self.labels, rankings)),
            'max_norm': max(norms),
            'min_norm': min(norms),
            'norm_ratio': max(norms) / min(norms) if min(norms) > 0 else np.inf,
            'norm_std': np.std(norms)
        }
    
    def component_wise_analysis(self, reference_idx: int = 0) -> Dict[str, Any]:
        """Perform element-wise analysis between vectors and a reference vector."""
        if reference_idx >= self.n_vectors:
            raise ValueError(f"Reference index {reference_idx} out of range")
        
        ref_vector = self.vectors[reference_idx]
        ref_label = self.labels[reference_idx]
        
        results = {'reference': ref_label}
        
        for i, (vec, label) in enumerate(zip(self.vectors, self.labels)):
            if i == reference_idx:
                continue
            
            # Element-wise differences
            diff = vec - ref_vector
            ratio = np.divide(vec, ref_vector, out=np.zeros_like(vec), 
                            where=ref_vector != 0)
            
            # Find positions with largest differences
            max_diff_idx = np.argmax(np.abs(diff))
            max_ratio_idx = np.argmax(np.abs(ratio - 1))
            
            results[label] = {
                'differences': diff,
                'ratios': ratio,
                'max_difference': np.max(np.abs(diff)),
                'max_difference_idx': max_diff_idx,
                'max_ratio': np.max(np.abs(ratio)),
                'max_ratio_idx': max_ratio_idx,
                'mean_absolute_difference': np.mean(np.abs(diff)),
                'mean_absolute_ratio_deviation': np.mean(np.abs(ratio - 1))
            }
        
        return results
    
    def statistical_summary(self) -> pd.DataFrame:
        """Calculate comprehensive statistical summaries for all vectors."""
        stats_data = []
        
        for vec, label in zip(self.vectors, self.labels):
            stats = {
                'Vector': label,
                'Length': len(vec),
                'Mean': np.mean(vec),
                'Median': np.median(vec),
                'Std': np.std(vec),
                'Min': np.min(vec),
                'Max': np.max(vec),
                'Range': np.max(vec) - np.min(vec),
                'Q1': np.percentile(vec, 25),
                'Q3': np.percentile(vec, 75),
                'IQR': np.percentile(vec, 75) - np.percentile(vec, 25),
                'Min_Idx': np.argmin(vec),
                'Max_Idx': np.argmax(vec),
                'Skewness': self._calculate_skewness(vec),
                'Kurtosis': self._calculate_kurtosis(vec)
            }
            stats_data.append(stats)
        
        return pd.DataFrame(stats_data)
    
    def _calculate_skewness(self, vec: np.ndarray) -> float:
        """Calculate skewness of a vector."""
        mean = np.mean(vec)
        std = np.std(vec)
        if std == 0:
            return 0
        return np.mean(((vec - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, vec: np.ndarray) -> float:
        """Calculate kurtosis of a vector."""
        mean = np.mean(vec)
        std = np.std(vec)
        if std == 0:
            return 0
        return np.mean(((vec - mean) / std) ** 4) - 3
    
    def distance_matrix(self, metric: str = 'euclidean') -> pd.DataFrame:
        """Calculate distance matrix between all pairs of vectors."""
        if metric not in ['euclidean', 'manhattan']:
            raise ValueError("Metric must be 'euclidean' or 'manhattan'")
        
        # Convert to 2D array for pdist
        vectors_2d = np.array(self.vectors)
        
        if metric == 'euclidean':
            distances = pdist(vectors_2d, metric='euclidean')
        else:  # manhattan
            distances = pdist(vectors_2d, metric='cityblock')
        
        # Convert to square matrix
        dist_matrix = squareform(distances)
        
        return pd.DataFrame(dist_matrix, 
                          index=self.labels, 
                          columns=self.labels)
    
    def similarity_matrix(self, metric: str = 'cosine') -> pd.DataFrame:
        """Calculate similarity matrix between all pairs of vectors."""
        if metric not in ['cosine', 'pearson']:
            raise ValueError("Metric must be 'cosine' or 'pearson'")
        
        n = len(self.vectors)
        sim_matrix = np.eye(n)  # Initialize with identity matrix
        
        for i in range(n):
            for j in range(i + 1, n):
                if metric == 'cosine':
                    # Cosine similarity
                    dot_product = np.dot(self.vectors[i], self.vectors[j])
                    norm_i = np.linalg.norm(self.vectors[i])
                    norm_j = np.linalg.norm(self.vectors[j])
                    
                    if norm_i == 0 or norm_j == 0:
                        similarity = 0
                    else:
                        similarity = dot_product / (norm_i * norm_j)
                else:  # pearson
                    # Pearson correlation
                    if len(self.vectors[i]) < 2:
                        similarity = 0
                    else:
                        corr, _ = pearsonr(self.vectors[i], self.vectors[j])
                        similarity = corr if not np.isnan(corr) else 0
                
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity
        
        return pd.DataFrame(sim_matrix, 
                          index=self.labels, 
                          columns=self.labels)
    
    def normalize_vectors(self, method: str = 'l2') -> 'VectorComparator':
        """Create a new VectorComparator with normalized vectors."""
        normalized_vectors = self._normalize_vectors(method)
        return VectorComparator(normalized_vectors, self.labels)
    
    def plot_vectors(self, figsize: Tuple[int, int] = (12, 8), 
                    style: str = 'line') -> None:
        """Create visualizations comparing all vectors."""
        plt.figure(figsize=figsize)
        
        if style == 'line':
            for vec, label in zip(self.vectors, self.labels):
                plt.plot(vec, label=label, marker='o', markersize=4)
            plt.xlabel('Component Index')
            plt.ylabel('Value')
            plt.title('Vector Comparison - Line Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        elif style == 'bar':
            x = np.arange(len(self.vectors[0]))
            width = 0.8 / len(self.vectors)
            
            for i, (vec, label) in enumerate(zip(self.vectors, self.labels)):
                plt.bar(x + i * width, vec, width, label=label, alpha=0.8)
            
            plt.xlabel('Component Index')
            plt.ylabel('Value')
            plt.title('Vector Comparison - Bar Chart')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_distance_heatmap(self, metric: str = 'euclidean', 
                             figsize: Tuple[int, int] = (8, 6)) -> None:
        """Create heatmap visualization of distance/similarity matrix."""
        plt.figure(figsize=figsize)
        
        if metric in ['euclidean', 'manhattan']:
            matrix = self.distance_matrix(metric)
            title = f'Distance Matrix ({metric.title()})'
            cmap = 'Reds'
        else:  # cosine or pearson
            matrix = self.similarity_matrix(metric)
            title = f'Similarity Matrix ({metric.title()})'
            cmap = 'RdYlBu_r'
        
        sns.heatmap(matrix, annot=True, cmap=cmap, center=0, 
                   square=True, fmt='.3f')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_component_analysis(self, reference_idx: int = 0, 
                               figsize: Tuple[int, int] = (15, 10)) -> None:
        """Create detailed component-wise analysis plots."""
        analysis = self.component_wise_analysis(reference_idx)
        ref_label = self.labels[reference_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: All vectors
        axes[0, 0].plot(self.vectors[reference_idx], label=ref_label, 
                       linewidth=2, marker='o')
        for i, (vec, label) in enumerate(zip(self.vectors, self.labels)):
            if i != reference_idx:
                axes[0, 0].plot(vec, label=label, alpha=0.7, marker='s')
        axes[0, 0].set_title('All Vectors')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Differences from reference
        for label, data in analysis.items():
            if label != 'reference':
                axes[0, 1].plot(data['differences'], label=f'{label} - {ref_label}')
        axes[0, 1].set_title('Differences from Reference')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Ratios to reference
        for label, data in analysis.items():
            if label != 'reference':
                axes[1, 0].plot(data['ratios'], label=f'{label} / {ref_label}')
        axes[1, 0].set_title('Ratios to Reference')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Statistical summary
        stats = self.statistical_summary()
        x_pos = np.arange(len(stats))
        axes[1, 1].bar(x_pos, stats['Mean'], alpha=0.7, label='Mean')
        axes[1, 1].errorbar(x_pos, stats['Mean'], yerr=stats['Std'], 
                           fmt='none', color='red', alpha=0.7)
        axes[1, 1].set_title('Mean Values with Standard Deviation')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(stats['Vector'], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_comprehensive_overview(self, reference_idx: int = 0, 
                                   figsize: Tuple[int, int] = (20, 16)) -> None:
        """
        Create a comprehensive overview plot showing all analysis results in one figure.
        
        Parameters
        ----------
        reference_idx : int, default=0
            Index of the reference vector for component analysis.
        figsize : Tuple[int, int], default=(20, 16)
            Figure size for the comprehensive plot.
        """
        fig = plt.figure(figsize=figsize)
        
        # Create a grid layout for all subplots
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Line plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        for vec, label in zip(self.vectors, self.labels):
            ax1.plot(vec, label=label, marker='o', markersize=3)
        ax1.set_title('Vector Comparison - Line Plot', fontsize=10)
        ax1.set_xlabel('Component Index')
        ax1.set_ylabel('Value')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Bar chart (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(self.vectors[0]))
        width = 0.8 / len(self.vectors)
        for i, (vec, label) in enumerate(zip(self.vectors, self.labels)):
            ax2.bar(x + i * width, vec, width, label=label, alpha=0.8)
        ax2.set_title('Vector Comparison - Bar Chart', fontsize=10)
        ax2.set_xlabel('Component Index')
        ax2.set_ylabel('Value')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Euclidean distance heatmap (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        euclidean_dist = self.distance_matrix('euclidean')
        sns.heatmap(euclidean_dist, annot=True, cmap='Reds', center=0, 
                   square=True, fmt='.2f', ax=ax3, cbar_kws={'shrink': 0.8})
        ax3.set_title('Euclidean Distance Matrix', fontsize=10)
        
        # 4. Manhattan distance heatmap (top far right)
        ax4 = fig.add_subplot(gs[0, 3])
        manhattan_dist = self.distance_matrix('manhattan')
        sns.heatmap(manhattan_dist, annot=True, cmap='Reds', center=0, 
                   square=True, fmt='.2f', ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Manhattan Distance Matrix', fontsize=10)
        
        # 5. Cosine similarity heatmap (second row left)
        ax5 = fig.add_subplot(gs[1, 0])
        cosine_sim = self.similarity_matrix('cosine')
        sns.heatmap(cosine_sim, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.3f', ax=ax5, cbar_kws={'shrink': 0.8})
        ax5.set_title('Cosine Similarity Matrix', fontsize=10)
        
        # 6. Pearson correlation heatmap (second row center)
        ax6 = fig.add_subplot(gs[1, 1])
        pearson_sim = self.similarity_matrix('pearson')
        sns.heatmap(pearson_sim, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.3f', ax=ax6, cbar_kws={'shrink': 0.8})
        ax6.set_title('Pearson Correlation Matrix', fontsize=10)
        
        # 7. Magnitude comparison (second row right)
        ax7 = fig.add_subplot(gs[1, 2])
        mag_comp = self.magnitude_comparison()
        norms = list(mag_comp['norms'].values())
        labels_list = list(mag_comp['norms'].keys())
        bars = ax7.bar(range(len(norms)), norms, alpha=0.7, color='skyblue')
        ax7.set_title('Vector Magnitudes (L2 Norms)', fontsize=10)
        ax7.set_xlabel('Vectors')
        ax7.set_ylabel('Euclidean Norm')
        ax7.set_xticks(range(len(labels_list)))
        ax7.set_xticklabels(labels_list, rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, norm) in enumerate(zip(bars, norms)):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{norm:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 8. Statistical summary (second row far right)
        ax8 = fig.add_subplot(gs[1, 3])
        stats = self.statistical_summary()
        x_pos = np.arange(len(stats))
        ax8.bar(x_pos, stats['Mean'], alpha=0.7, label='Mean', color='lightcoral')
        ax8.errorbar(x_pos, stats['Mean'], yerr=stats['Std'], 
                    fmt='none', color='red', alpha=0.7)
        ax8.set_title('Mean Values ± Std Dev', fontsize=10)
        ax8.set_xlabel('Vectors')
        ax8.set_ylabel('Value')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(stats['Vector'], rotation=45)
        ax8.grid(True, alpha=0.3)
        
        # 9. Component differences from reference (third row left)
        ax9 = fig.add_subplot(gs[2, 0])
        analysis = self.component_wise_analysis(reference_idx)
        ref_label = self.labels[reference_idx]
        for label, data in analysis.items():
            if label != 'reference':
                ax9.plot(data['differences'], label=f'{label} - {ref_label}', alpha=0.8)
        ax9.set_title(f'Differences from {ref_label}', fontsize=10)
        ax9.set_xlabel('Component Index')
        ax9.set_ylabel('Difference')
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
        ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 10. Component ratios to reference (third row center)
        ax10 = fig.add_subplot(gs[2, 1])
        for label, data in analysis.items():
            if label != 'reference':
                ax10.plot(data['ratios'], label=f'{label} / {ref_label}', alpha=0.8)
        ax10.set_title(f'Ratios to {ref_label}', fontsize=10)
        ax10.set_xlabel('Component Index')
        ax10.set_ylabel('Ratio')
        ax10.legend(fontsize=8)
        ax10.grid(True, alpha=0.3)
        ax10.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # 11. Range comparison (third row right)
        ax11 = fig.add_subplot(gs[2, 2])
        ranges = stats['Max'] - stats['Min']
        bars = ax11.bar(range(len(ranges)), ranges, alpha=0.7, color='lightgreen')
        ax11.set_title('Vector Ranges (Max - Min)', fontsize=10)
        ax11.set_xlabel('Vectors')
        ax11.set_ylabel('Range')
        ax11.set_xticks(range(len(stats)))
        ax11.set_xticklabels(stats['Vector'], rotation=45)
        ax11.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, range_val) in enumerate(zip(bars, ranges)):
            ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{range_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 12. IQR comparison (third row far right)
        ax12 = fig.add_subplot(gs[2, 3])
        bars = ax12.bar(range(len(stats)), stats['IQR'], alpha=0.7, color='orange')
        ax12.set_title('Interquartile Range (IQR)', fontsize=10)
        ax12.set_xlabel('Vectors')
        ax12.set_ylabel('IQR')
        ax12.set_xticks(range(len(stats)))
        ax12.set_xticklabels(stats['Vector'], rotation=45)
        ax12.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, iqr) in enumerate(zip(bars, stats['IQR'])):
            ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{iqr:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 13. Skewness comparison (fourth row left)
        ax13 = fig.add_subplot(gs[3, 0])
        bars = ax13.bar(range(len(stats)), stats['Skewness'], alpha=0.7, color='purple')
        ax13.set_title('Skewness', fontsize=10)
        ax13.set_xlabel('Vectors')
        ax13.set_ylabel('Skewness')
        ax13.set_xticks(range(len(stats)))
        ax13.set_xticklabels(stats['Vector'], rotation=45)
        ax13.grid(True, alpha=0.3)
        ax13.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, (bar, skew) in enumerate(zip(bars, stats['Skewness'])):
            ax13.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{skew:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 14. Kurtosis comparison (fourth row center)
        ax14 = fig.add_subplot(gs[3, 1])
        bars = ax14.bar(range(len(stats)), stats['Kurtosis'], alpha=0.7, color='brown')
        ax14.set_title('Kurtosis', fontsize=10)
        ax14.set_xlabel('Vectors')
        ax14.set_ylabel('Kurtosis')
        ax14.set_xticks(range(len(stats)))
        ax14.set_xticklabels(stats['Vector'], rotation=45)
        ax14.grid(True, alpha=0.3)
        ax14.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, (bar, kurt) in enumerate(zip(bars, stats['Kurtosis'])):
            ax14.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{kurt:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 15. Min/Max comparison (fourth row right)
        ax15 = fig.add_subplot(gs[3, 2])
        x_pos = np.arange(len(stats))
        ax15.bar(x_pos - 0.2, stats['Min'], 0.4, label='Min', alpha=0.7, color='lightblue')
        ax15.bar(x_pos + 0.2, stats['Max'], 0.4, label='Max', alpha=0.7, color='lightcoral')
        ax15.set_title('Min/Max Values', fontsize=10)
        ax15.set_xlabel('Vectors')
        ax15.set_ylabel('Value')
        ax15.set_xticks(x_pos)
        ax15.set_xticklabels(stats['Vector'], rotation=45)
        ax15.legend(fontsize=8)
        ax15.grid(True, alpha=0.3)
        
        # 16. Summary statistics table (fourth row far right)
        ax16 = fig.add_subplot(gs[3, 3])
        ax16.axis('off')
        
        # Create a summary table
        summary_text = f"Summary Statistics\n\n"
        summary_text += f"Number of vectors: {self.n_vectors}\n"
        summary_text += f"Vector length: {self.vector_length}\n"
        summary_text += f"Reference vector: {ref_label}\n\n"
        
        # Add key statistics
        summary_text += f"Largest norm: {mag_comp['max_norm']:.3f}\n"
        summary_text += f"Smallest norm: {mag_comp['min_norm']:.3f}\n"
        summary_text += f"Norm ratio: {mag_comp['norm_ratio']:.3f}\n\n"
        
        # Add most similar/different pairs
        euclidean_dist_no_diag = euclidean_dist.values
        np.fill_diagonal(euclidean_dist_no_diag, np.inf)
        min_dist_idx = np.unravel_index(np.argmin(euclidean_dist_no_diag), euclidean_dist_no_diag.shape)
        max_dist_idx = np.unravel_index(np.argmax(euclidean_dist_no_diag), euclidean_dist_no_diag.shape)
        
        summary_text += f"Most similar: {self.labels[min_dist_idx[0]]} & {self.labels[min_dist_idx[1]]}\n"
        summary_text += f"Most different: {self.labels[max_dist_idx[0]]} & {self.labels[max_dist_idx[1]]}\n"
        
        ax16.text(0.05, 0.95, summary_text, transform=ax16.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add overall title
        fig.suptitle('Comprehensive Vector Analysis Overview', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, include_plots: bool = True, 
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        report = {
            'summary': {
                'n_vectors': self.n_vectors,
                'vector_length': self.vector_length,
                'labels': self.labels
            },
            'magnitude_comparison': self.magnitude_comparison(),
            'statistical_summary': self.statistical_summary().to_dict('records'),
            'distance_matrix': self.distance_matrix().to_dict(),
            'similarity_matrix': self.similarity_matrix().to_dict(),
            'component_analysis': self.component_wise_analysis()
        }
        
        if include_plots:
            # Note: In a real implementation, you might want to save plots
            # and include their paths in the report
            report['plots_generated'] = True
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_report = self._convert_numpy_for_json(report)
                json.dump(json_report, f, indent=2)
        
        return report
    
    def _convert_numpy_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        else:
            return obj


# Example usage and testing
if __name__ == "__main__":
    # Create sample vectors for testing
    np.random.seed(42)
    vectors = [
        np.random.normal(0, 1, 10),
        np.random.normal(0.5, 1.2, 10),
        np.random.normal(-0.3, 0.8, 10),
        np.random.normal(0.2, 1.2, 10),
        np.random.normal(-0.8, 0.8, 10)
    ]
    labels = ["Run 1", "Run 2", "Run 3", "Run 4", "Run 5"]
    
    # Create comparator
    comparator = VectorComparator(vectors, labels)
    
    # Generate report
    print("Vector Comparison Report")
    print("=" * 50)
    
    # Statistical summary
    stats = comparator.statistical_summary()
    print("\nStatistical Summary:")
    print(stats)
    
    # Magnitude comparison
    mag_comp = comparator.magnitude_comparison()
    print(f"\nMagnitude Comparison:")
    print(f"Norms: {mag_comp['norms']}")
    print(f"Rankings: {mag_comp['rankings']}")
    
    # Distance matrix
    distances = comparator.distance_matrix()
    print(f"\nEuclidean Distance Matrix:")
    print(distances)
    
    # Similarity matrix
    similarities = comparator.similarity_matrix()
    print(f"\nCosine Similarity Matrix:")
    print(similarities)
    
    # Visualization demonstrations
    print(f"\nGenerating visualizations...")
    print("=" * 50)
    
    # 1. Line plot comparison
    print("1. Line Plot - Vector Comparison")
    print("-" * 30)
    comparator.plot_vectors(style='line', figsize=(10, 6))
    
    # 2. Bar chart comparison
    print("2. Bar Chart - Vector Comparison")
    print("-" * 30)
    comparator.plot_vectors(style='bar', figsize=(12, 6))
    
    # 3. Distance matrix heatmap
    print("3. Distance Matrix Heatmap (Euclidean)")
    print("-" * 30)
    comparator.plot_distance_heatmap(metric='euclidean', figsize=(8, 6))
    
    # 4. Manhattan distance heatmap
    print("4. Distance Matrix Heatmap (Manhattan)")
    print("-" * 30)
    comparator.plot_distance_heatmap(metric='manhattan', figsize=(8, 6))
    
    # 5. Cosine similarity heatmap
    print("5. Similarity Matrix Heatmap (Cosine)")
    print("-" * 30)
    comparator.plot_distance_heatmap(metric='cosine', figsize=(8, 6))
    
    # 6. Pearson correlation heatmap
    print("6. Similarity Matrix Heatmap (Pearson)")
    print("-" * 30)
    comparator.plot_distance_heatmap(metric='pearson', figsize=(8, 6))
    
    # 7. Component-wise analysis (comprehensive 4-panel plot)
    print("7. Component-wise Analysis (4-panel plot)")
    print("-" * 30)
    comparator.plot_component_analysis(reference_idx=0, figsize=(15, 10))
    
    # 8. Component-wise analysis with different reference
    print("8. Component-wise Analysis (Reference: Run 2)")
    print("-" * 30)
    comparator.plot_component_analysis(reference_idx=1, figsize=(15, 10))
    
    # 9. Comprehensive overview plot (all plots in one figure)
    print("9. Comprehensive Overview Plot (All Analysis in One Figure)")
    print("-" * 30)
    comparator.plot_comprehensive_overview(reference_idx=0, figsize=(20, 16))
    
    print("\nAll visualizations completed!")
    print("Note: Close each plot window to proceed to the next visualization.")