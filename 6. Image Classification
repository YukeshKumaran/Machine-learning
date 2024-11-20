# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_openml

# Load the dataset (example: CIFAR-10 or another dataset of animal images)
# Replace this with your actual image dataset
data = fetch_openml('mnist_784')  # Example dataset: MNIST (digits 0-9)
X = data.data  # Image pixel data
y = data.target.astype(int)  # Image labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Reduce to 50 components (tune this value)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize K-NN classifier with optimal K (found through cross-validation/grid search)
knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='euclidean')

# Train the K-NN model
knn.fit(X_train_pca, y_train)

# Predict on the test data
y_pred = knn.predict(X_test_pca)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation scores
cv_scores = cross_val_score(knn, X_train_pca, y_train, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())

# Optional: Plot the first few predictions vs actual values
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test.iloc[i].reshape(28, 28), cmap='gray')  # Adjust image reshape depending on dataset
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.show()
