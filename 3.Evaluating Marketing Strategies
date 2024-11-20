# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Sample initial market data (replace with actual data)
# For simplicity, assume 'market_conditions' includes features like interest rate, inflation, etc.
data = {
    'interest_rate': [3.5, 4.0, 3.0, 5.0, 4.2, 3.7, 3.9],
    'unemployment_rate': [5.2, 5.0, 4.8, 6.0, 5.5, 5.1, 4.9],
    'gdp_growth': [2.5, 2.0, 3.0, 1.8, 2.2, 2.4, 2.7],
    'stock_market': ['up', 'down', 'up', 'down', 'up', 'down', 'up'],  # Target variable
}

df = pd.DataFrame(data)

# Convert categorical target variable to numeric (up -> 1, down -> 0)
df['stock_market'] = df['stock_market'].map({'up': 1, 'down': 0})

# Define features (X) and target variable (y)
X = df.drop('stock_market', axis=1)
y = df['stock_market']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the initial Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate initial model performance
print("Initial Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the initial Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=['Down', 'Up'], filled=True)
plt.title("Initial Decision Tree")
plt.show()

# New market data to update the model (market conditions change)
new_data = {
    'interest_rate': [4.1, 3.8, 5.0, 3.3, 4.5, 3.9, 4.0],
    'unemployment_rate': [5.4, 5.2, 6.1, 5.0, 5.6, 5.3, 5.0],
    'gdp_growth': [2.3, 2.1, 1.9, 2.5, 2.8, 2.4, 2.6],
    'stock_market': ['up', 'down', 'up', 'down', 'up', 'down', 'up'],
}

df_new = pd.DataFrame(new_data)
df_new['stock_market'] = df_new['stock_market'].map({'up': 1, 'down': 0})

# Define features and target for the new data
X_new = df_new.drop('stock_market', axis=1)
y_new = df_new['stock_market']

# Retrain the model with new data
X_combined = pd.concat([X, X_new], axis=0)
y_combined = pd.concat([y, y_new], axis=0)

# Split combined data into new training and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Retrain the Decision Tree model with the updated data
model_updated = DecisionTreeClassifier(random_state=42)
model_updated.fit(X_train_new, y_train_new)

# Predict on the updated test data
y_pred_updated = model_updated.predict(X_test_new)

# Evaluate the updated model
print("\nUpdated Model Evaluation:")
print("Accuracy:", accuracy_score(y_test_new, y_pred_updated))
print("\nClassification Report:\n", classification_report(y_test_new, y_pred_updated))

# Visualize the updated Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model_updated, feature_names=X.columns, class_names=['Down', 'Up'], filled=True)
plt.title("Updated Decision Tree")
plt.show()

# Compare feature importance between initial and updated models
print("\nFeature Importances (Initial Model):", model.feature_importances_)
print("Feature Importances (Updated Model):", model_updated.feature_importances_)
