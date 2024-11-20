# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample real estate data (replace with actual data)
# 'location', 'size', 'bedrooms' are features and 'price' is the target variable
data = {
    'location': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'C', 'B', 'A'],
    'size': [1500, 1800, 1600, 1400, 1700, 1500, 1450, 1650, 1800, 1600],
    'bedrooms': [3, 4, 3, 2, 4, 3, 3, 4, 4, 3],
    'price': [400000, 500000, 450000, 380000, 490000, 470000, 460000, 510000, 520000, 450000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert categorical feature 'location' into numerical format using one-hot encoding
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Define features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the house prices on the test set
y_pred = model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Calculate other evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display the evaluation metrics
print(f"R-squared: {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

# Plot actual vs predicted values for visualization
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
