from sklearn.cluster import AgglomerativeClustering
import numpy as np

def agglomerative_clustering(X, n_clusters=2, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels

# Optionally, add a function for compatibility with run_clustering if required
def run_agglomerative(X, **kwargs):
    n_clusters = kwargs.get('k', 2)
    linkage = kwargs.get('linkage', 'ward')
    labels = agglomerative_clustering(X, n_clusters=n_clusters, linkage=linkage)
    return labels
