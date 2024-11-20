# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Sample data (replace this with actual email dataset)
# 'text' column contains the email text, 'label' column contains the spam (1) or not spam (0) labels
data = {
    'text': [
        'Free money, click here to claim', 
        'Hi, we are looking forward to your feedback', 
        'Limited time offer, buy now!',
        'Hey, what are you up to?',
        'Get rich quick with this simple trick',
        'Dear user, your account is in danger',
        'Meeting tomorrow at 3pm',
        'Exclusive offer just for you',
        'Reminder about our meeting tomorrow',
        'You have won a lottery, claim your prize'
    ],
    'label': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1 = spam, 0 = not spam
}

# Create DataFrame
df = pd.DataFrame(data)

# Define features (X) and target variable (y)
X = df['text']
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with TF-IDF Vectorizer and SVM model
# If using a linear kernel, you can specify 'linear' for the kernel
svm_model = make_pipeline(
    TfidfVectorizer(stop_words='english'),  # Convert email text to numerical features
    SVC(kernel='linear', random_state=42)   # Linear kernel SVM
)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# If you want to use a non-linear kernel (e.g., RBF kernel), simply change the kernel:
# svm_model_rbf = make_pipeline(
#     TfidfVectorizer(stop_words='english'),
#     SVC(kernel='rbf', random_state=42)
# )
# svm_model_rbf.fit(X_train, y_train)
# y_pred_rbf = svm_model_rbf.predict(X_test)
# print("\nRBF Kernel Accuracy:", accuracy_score(y_test, y_pred_rbf))
