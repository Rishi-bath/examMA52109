###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
##
## This script produces clustering for a group of points in 2D,
## using k-means for k = 2, 3, 4, 5. The input file is the csv
## file 'demo_data.csv' in folder 'data/'.
###

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import runClustering

OUTPUT_DIR = "demo_output"


def main(args: List[str]) -> None:
    if len(args) != 1:
        print("Usage: python cluster_plot.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The CSV is assumed to have two or more data columns
    df = pd.read_csv(input_path)
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    feature_cols = numeric_cols[:2]  # Use the first two numeric columns

    # For naming outputs
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Main job: run clustering for k = 2, 3, 4, 5
    metrics_summary = []

    for k in (2, 3, 4, 5):
        print(f"\n=== Running k-means with k = {k} ===")

        # CORRECTED: All camelCase names and parameters
        result = runClustering(
            inputPath=input_path,
            featureCols=feature_cols,
            algorithm="kmeans",
            k=k,  # FIX: Use k directly (no min() function)
            standardise=True,
            outputPath=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            randomState=42,
            computeElbow=False,  # no elbow diagram
        )

        # Save cluster plot - USING CAMELCASE KEY
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        result["figCluster"].savefig(plot_path, dpi=150)  # FIXED: figCluster (camelCase)
        plt.close(result["figCluster"])

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # Summarise metrics across k
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Plot some statistics (no elbow: avoid inertia vs k)
    if "silhouette" in metrics_df.columns:
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette"])
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score for different k")
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette.png")
        plt.savefig(stats_path, dpi=150)
        plt.close()

    print("\nDemo completed.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
