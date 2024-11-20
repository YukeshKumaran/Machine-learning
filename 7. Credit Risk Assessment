# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Sample data for credit risk assessment (replace with actual data)
# Features: credit_score, income_level, employment_status (binary: 1 = employed, 0 = unemployed)
data = {
    'credit_score': [750, 650, 580, 700, 720, 690, 710, 640, 680, 800],
    'income_level': [50000, 40000, 30000, 45000, 60000, 55000, 48000, 35000, 52000, 65000],
    'employment_status': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],  # 1 = employed, 0 = unemployed
    'default': [0, 1, 1, 0, 0, 1, 0, 1, 0, 0]  # 1 = default, 0 = no default
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define features (X) and target variable (y)
X = df[['credit_score', 'income_level', 'employment_status']]
y = df['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get the coefficients (log-odds) from the model
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

# Print the coefficients and intercept
print("\nCoefficients:", coefficients)
print("Intercept:", intercept)

# Interpret the coefficients: Calculate odds ratios by exponentiating the coefficients
odds_ratios = np.exp(coefficients)
print("\nOdds Ratios (Exp(Coefficients)):")
print(f"Credit Score: {odds_ratios[0]:.4f}")
print(f"Income Level: {odds_ratios[1]:.4f}")
print(f"Employment Status: {odds_ratios[2]:.4f}")

# You can also visualize the feature importance (coefficients)
import matplotlib.pyplot as plt

# Create a bar plot for coefficient importance
features = ['Credit Score', 'Income Level', 'Employment Status']
plt.bar(features, odds_ratios)
plt.xlabel('Features')
plt.ylabel('Odds Ratios')
plt.title('Feature Importance (Odds Ratios) for Credit Risk Assessment')
plt.show()
