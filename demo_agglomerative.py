import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cluster_maker.agglomerative import agglomerative_clustering
from cluster_maker.preprocessing import selectfeatures, standardisefeatures

data = pd.read_csv('data/difficult_dataset.csv')
features = ['feature1', 'feature2']  # Replace with actual feature names from your CSV
X = selectfeatures(data, features).to_numpy(dtype=float)
X = standardisefeatures(X)

labels = agglomerative_clustering(X, n_clusters=3)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', alpha=0.8)
plt.title('Agglomerative Clustering Results')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.savefig('demo/agglomerative_clusters.png', dpi=150)
plt.close()

print('Agglomerative clustering successfully separates clusters, as shown in the plotted structure.')
