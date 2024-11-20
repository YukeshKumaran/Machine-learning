# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Example dataset: daily temperatures (You can replace this with actual weather data)
# The dataset contains dates and temperature for simplicity
# Create a dataset with seasonal patterns
days = np.arange(1, 366)  # Day of the year (1 to 365)
temperature = 10 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 2, 365)  # Simulated temperature data with seasonality

# Convert to DataFrame
df = pd.DataFrame({'Day': days, 'Temperature': temperature})

# Plot the original data to visualize the seasonal pattern
plt.figure(figsize=(10, 6))
plt.plot(df['Day'], df['Temperature'], label='Temperature', color='blue')
plt.title('Daily Temperature with Seasonal Variation')
plt.xlabel('Day of the Year')
plt.ylabel('Temperature')
plt.grid(True)
plt.show()

# Polynomial Regression to account for seasonality
# Create polynomial features to capture the cyclical behavior
poly = PolynomialFeatures(degree=4)  # Degree 4 to capture higher-order seasonal effects
X_poly = poly.fit_transform(df[['Day']])  # Polynomial features for the day

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, df['Temperature'], test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot the predicted vs actual temperatures
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual Temperature')
plt.scatter(X_test[:, 1], y_pred, color='red', label='Predicted Temperature')
plt.title('Polynomial Regression for Temperature Prediction')
plt.xlabel('Day of the Year')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model performance (R-squared)
r_squared = model.score(X_test, y_test)
print(f'R-squared: {r_squared:.4f}')
