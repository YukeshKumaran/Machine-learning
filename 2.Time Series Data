import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Simulated time series data
data = {
    "Month": range(1, 37),
    "Consumption": [200, 220, 210, 240, 250, 230, 260, 270, 280, 300, 320, 310] * 3
}

df = pd.DataFrame(data)

# Features and target
X = df[["Month"]]
y = df["Consumption"]

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
future_months = pd.DataFrame({"Month": range(37, 49)})
future_predictions = model.predict(future_months)

print("Future Predictions:", future_predictions)
