# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Example gene expression data (rows = samples, columns = genes)
# Replace this with actual gene expression data
data = {
    'Gene1': [5.1, 3.4, 4.2, 6.1, 5.9, 4.3],
    'Gene2': [3.2, 2.9, 3.5, 4.0, 3.9, 2.7],
    'Gene3': [1.1, 1.3, 1.2, 1.0, 1.2, 1.1],
    'Gene4': [4.1, 4.5, 4.0, 5.2, 4.9, 4.3],
    'Gene5': [0.5, 0.8, 0.7, 0.6, 0.5, 0.6],
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA (reduce to 2 components for visualization)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Plot the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', marker='o')
plt.title('PCA of Gene Expression Data')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True)
plt.show()

# Explained variance ratio (How much variance is captured by each principal component)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Principal components (loadings)
print("Principal Components (loadings):")
print(pca.components_)

# Plotting the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()
