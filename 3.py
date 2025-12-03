from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create DataFrame for clarity
df = pd.DataFrame(X, columns=feature_names)

# Apply PCA to reduce 4 features → 2 principal components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df)

# Create a DataFrame for PCA result
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Print explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Plot the 2 principal components
plt.figure(figsize=(8,6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['target'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset (4 features → 2)')
plt.colorbar(label='Target Class')
plt.show()
