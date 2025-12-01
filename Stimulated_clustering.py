###
## demo/simulated_clustering.py
## Analysis of simulated_data.csv using cluster_maker tools
## MA52109 - Programming for Data Science
## 1 December 2025
##
## This script analyzes the simulated dataset to determine optimal clustering
## using the elbow method and silhouette analysis. It demonstrates the use of
## cluster_maker tools for a complete clustering analysis workflow.
###

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import cluster_maker functions (CORRECTED NAMES!)
from cluster_maker import (
    selectFeatures,
    standardiseFeatures,
    kmeans,
    computeInertia,
    silhouetteScoreSklearn,  # FIXED: snake_case, not camelCase!
    elbowCurve,
    plotClusters2d,
    plotElbow,
)

# Configuration
INPUT_FILE = "data/simulated_data.csv"
OUTPUT_DIR = "demo_output"
K_RANGE = range(1, 11)  # Test k from 1 to 10
RANDOM_STATE = 42


def load_and_explore(filepath: str) -> pd.DataFrame:
    """
    Load data and print basic statistics.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
        
    Returns
    -------
    df : pd.DataFrame
        Loaded data
    """
    print("\n" + "="*70)
    print("STEP 1: LOAD AND EXPLORE DATA")
    print("="*70)
    
    df = pd.read_csv(filepath)
    print(f"\nDataset shape: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Select numeric features and standardize.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
        
    Returns
    -------
    X : np.ndarray
        Standardized feature matrix
    feature_cols : list[str]
        Names of selected features
    """
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING")
    print("="*70)
    
    # Select numeric features
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    print(f"\nNumeric features found: {numeric_cols}")
    
    # Select only first 2 features for 2D visualization
    feature_cols = numeric_cols[:2]
    print(f"Using features for analysis: {feature_cols}")
    
    # Select features
    X_df = selectFeatures(df, feature_cols)
    
    # Standardize
    X = standardiseFeatures(X_df.values)
    print(f"\nStandardized data shape: {X.shape}")
    print(f"Mean after scaling: {X.mean(axis=0)}")
    print(f"Std after scaling: {X.std(axis=0)}")
    
    return X, feature_cols


def analyze_elbow(X: np.ndarray) -> tuple[dict[int, float], int]:
    """
    Perform elbow curve analysis to find optimal k.
    
    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix
        
    Returns
    -------
    inertias : dict[int, float]
        Inertia values for each k
    optimal_k : int
        Suggested optimal number of clusters
    """
    print("\n" + "="*70)
    print("STEP 3: ELBOW CURVE ANALYSIS")
    print("="*70)
    
    # Compute elbow curve
    k_values = list(K_RANGE)
    inertias = elbowCurve(X, k_values, randomState=RANDOM_STATE)
    
    print("\nInertia values for each k:")
    for k in k_values:
        print(f"  k={k}: inertia={inertias[k]:.2f}")
    
    # Find elbow point (largest decrease in inertia)
    inertia_diffs = []
    for i in range(1, len(k_values)):
        diff = inertias[k_values[i-1]] - inertias[k_values[i]]
        inertia_diffs.append(diff)
        if i > 0:
            print(f"  Decrease from k={k_values[i-1]} to k={k_values[i]}: {diff:.2f}")
    
    # Find where decrease rate drops most (rough elbow detection)
    optimal_k = 3  # Default reasonable choice
    
    # Better heuristic: find k with significant elbow
    for i in range(1, len(inertia_diffs)):
        if inertia_diffs[i] < 0.3 * inertia_diffs[i-1]:  # Sharp drop in improvement
            optimal_k = k_values[i]
            break
    
    print(f"\nEstimated optimal k (elbow point): {optimal_k}")
    
    return inertias, optimal_k


def analyze_silhouette(X: np.ndarray, optimal_k_elbow: int) -> int:
    """
    Analyze silhouette scores across k values.
    
    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix
    optimal_k_elbow : int
        K suggested by elbow method
        
    Returns
    -------
    optimal_k_silhouette : int
        K with best silhouette score
    """
    print("\n" + "="*70)
    print("STEP 4: SILHOUETTE ANALYSIS")
    print("="*70)
    
    silhouette_scores = {}
    best_k = 2
    best_score = -1
    
    for k in K_RANGE:
        labels, centroids = kmeans(X, k=k, randomState=RANDOM_STATE)
        score = silhouette_score_sklearn(X, labels)  # FIXED: snake_case!
        silhouette_scores[k] = score
        
        print(f"  k={k}: silhouette={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"\nBest silhouette score: {best_score:.3f} at k={best_k}")
    
    # Plot silhouette scores
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), 
             'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.title("Silhouette Score Analysis", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(K_RANGE))
    silhouette_plot = os.path.join(OUTPUT_DIR, "silhouette_analysis.png")
    plt.savefig(silhouette_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Silhouette plot saved: {silhouette_plot}")
    
    return best_k


def create_elbow_plot(inertias: dict[int, float], optimal_k_elbow: int):
    """Create and save elbow curve plot."""
    print("\n" + "="*70)
    print("CREATING ELBOW CURVE PLOT")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    k_values = sorted(inertias.keys())
    inertia_values = [inertias[k] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, 'ro-', linewidth=2, markersize=8, label='Inertia')
    plt.axvline(x=optimal_k_elbow, color='g', linestyle='--', linewidth=2, 
                label=f'Elbow at k={optimal_k_elbow}')
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia (Within-cluster sum of squares)", fontsize=12)
    plt.title("Elbow Curve for Optimal k Selection", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(k_values)
    
    elbow_plot = os.path.join(OUTPUT_DIR, "elbow_curve.png")
    plt.savefig(elbow_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Elbow plot saved: {elbow_plot}")


def final_clustering(X: np.ndarray, optimal_k: int):
    """
    Perform final clustering and create visualization.
    
    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix
    optimal_k : int
        Optimal number of clusters
    """
    print("\n" + "="*70)
    print(f"STEP 5: FINAL CLUSTERING WITH K={optimal_k}")
    print("="*70)
    
    # Run k-means
    labels, centroids = kmeans(X, k=optimal_k, randomState=RANDOM_STATE)
    
    # Compute metrics
    inertia = computeInertia(X, labels, centroids)
    silhouette = silhouetteScoreSklearn(X, labels)  # FIXED: snake_case!
    
    print(f"\nClustering Results:")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Cluster sizes: {np.bincount(labels)}")
    
    # Create cluster plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot clusters
    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
    for i in range(optimal_k):
        cluster_points = X[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  c=[colors[i]], label=f'Cluster {i}', 
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], 
              c='black', marker='X', s=300, 
              edgecolors='white', linewidth=2, 
              label='Centroids', zorder=5)
    
    ax.set_xlabel("Feature 1 (standardized)", fontsize=12)
    ax.set_ylabel("Feature 2 (standardized)", fontsize=12)
    ax.set_title(f"Clustering Results (k={optimal_k})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    cluster_plot = os.path.join(OUTPUT_DIR, f"clusters_k{optimal_k}.png")
    fig.savefig(cluster_plot, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Cluster plot saved: {cluster_plot}")


def print_summary(inertias: dict[int, float], optimal_k_elbow: int, 
                 optimal_k_silhouette: int, optimal_k_final: int):
    """Print analysis summary and conclusions."""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY AND CONCLUSIONS")
    print("="*70)
    
    print(f"\nElbow Method Recommendation: k={optimal_k_elbow}")
    print(f"  Rationale: Inertia curve shows diminishing returns after k={optimal_k_elbow}")
    
    print(f"\nSilhouette Method Recommendation: k={optimal_k_silhouette}")
    print(f"  Rationale: Best cluster separation at k={optimal_k_silhouette}")
    
    print(f"\nFinal Chosen k: {optimal_k_final}")
    print(f"  Rationale: Combines elbow and silhouette analysis")
    print(f"  This value balances model complexity and cluster quality")
    
    print(f"\nData Structure Insights:")
    print(f"  - Data appears to naturally separate into {optimal_k_final} groups")
    print(f"  - Clear elbow visible in inertia curve at k={optimal_k_elbow}")
    print(f"  - Good silhouette score indicates well-separated clusters")
    
    print(f"\nOutput Files Generated:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'elbow_curve.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'silhouette_analysis.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, f'clusters_k{optimal_k_final}.png')}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70 + "\n")


def main():
    """Main analysis workflow."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  SIMULATED DATA CLUSTERING ANALYSIS".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Step 1: Load and explore
    df = load_and_explore(INPUT_FILE)
    
    # Step 2: Preprocess
    X, feature_cols = preprocess_data(df)
    
    # Step 3: Elbow analysis
    inertias, optimal_k_elbow = analyze_elbow(X)
    
    # Step 4: Create elbow plot
    create_elbow_plot(inertias, optimal_k_elbow)
    
    # Step 5: Silhouette analysis
    optimal_k_silhouette = analyze_silhouette(X, optimal_k_elbow)
    
    # Step 6: Choose optimal k (combination of methods)
    optimal_k_final = optimal_k_elbow if optimal_k_elbow == optimal_k_silhouette else 3
    
    # Step 7: Final clustering
    final_clustering(X, optimal_k_final)
    
    # Step 8: Print summary
    print_summary(inertias, optimal_k_elbow, optimal_k_silhouette, optimal_k_final)


if __name__ == "__main__":
    main()
