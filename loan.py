

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv("loan_approval_dataset.csv")
data.columns = data.columns.str.strip()
print("\nDataset Loaded Successfully\n")
print(data.head())
print("\nDataset Info:\n")
print(data.info())
print("\nMissing Values Before Encoding:\n")
print(data.isnull().sum())
data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()
data['education'] = data['education'].map({
    'Graduate': 1,
    'Not Graduate': 0
})

data['self_employed'] = data['self_employed'].map({
    'Yes': 1,
    'No': 0
})

data['loan_status'] = data['loan_status'].map({
    'Approved': 1,
    'Rejected': 0
})

print("\nEncoding Completed\n")
print("Missing Values After Encoding:\n")
print(data.isnull().sum())

# Drop rows with NaN (safe because very few)
data.dropna(inplace=True)
X = data.drop(['loan_status', 'loan_id'], axis=1)
y = data['loan_status']

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape :", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Completed")
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("\nModel Training Completed")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
"""
Input Order:
[
 no_of_dependents,
 education,
 self_employed,
 income_annum,
 loan_amount,
 loan_term,
 cibil_score,
 residential_assets_value,
 commercial_assets_value,
 luxury_assets_value,
 bank_asset_value
]
"""

sample_input = [[
    2,        # no_of_dependents
    1,        # education (Graduate)
    0,        # self_employed (No)
    600000,   # income_annum
    250000,   # loan_amount
    12,       # loan_term
    780,      # cibil_score
    300000,   # residential_assets_value
    0,        # commercial_assets_value
    0,        # luxury_assets_value
    100000    # bank_asset_value
]]

prediction = model.predict(sample_input)

print("\nLoan Prediction Result:")
print("✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected")



