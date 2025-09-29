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
        np.random.normal(-0.3, 0.8, 10)
    ]
    labels = ["Run 1", "Run 2", "Run 3"]
    
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
    
    print("\nAll visualizations completed!")
    print("Note: Close each plot window to proceed to the next visualization.")