# pca_precompute.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import torch

# Load the CLIP embeddings from your precomputed pickle file
df = pd.read_pickle('image_embeddings.pickle')
embeddings = np.stack(df['embedding'].values)  # shape: [num_images, embedding_dim]

# Number of principal components
k = 50
pca = PCA(n_components=k)
pca.fit(embeddings)

# Save PCA components and mean
np.save('pca_components.npy', pca.components_.T)  # shape: embedding_dim x k
np.save('pca_mean.npy', pca.mean_)                # shape: embedding_dim

# Transform all embeddings
pca_embeddings = pca.transform(embeddings)  # shape: [num_images, k]
np.save('pca_embeddings.npy', pca_embeddings)

def nearest_neighbors(query_embedding, embeddings, top_k=5):
    # query_embedding: 1D vector in the same space as embeddings
    distances = euclidean_distances([query_embedding], embeddings).flatten()
    nearest_indices = np.argsort(distances)[:top_k]
    return nearest_indices, distances[nearest_indices]

# Example usage:
if __name__ == '__main__':
    query_idx = 0
    query_embedding = pca_embeddings[query_idx]
    top_indices, top_distances = nearest_neighbors(query_embedding, pca_embeddings)
    print("Top indices:", top_indices)
    print("Top distances:", top_distances)
