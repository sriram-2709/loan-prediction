"""
PROJECT TITLE : Loan Approval Prediction using Machine Learning
DOMAIN        : Artificial Intelligence & Machine Learning
PROBLEM TYPE  : Binary Classification
ALGORITHM     : Logistic Regression
BACKEND       : Not Required
LANGUAGE      : Python
"""

# --------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# STEP 2: LOAD DATASET
# --------------------------------------------------

data = pd.read_csv("loan_approval_dataset.csv")

# Clean column names (VERY IMPORTANT)
data.columns = data.columns.str.strip()

print("\nDataset Loaded Successfully\n")
print(data.head())


# --------------------------------------------------
# STEP 3: BASIC DATA CHECKS
# --------------------------------------------------

print("\nDataset Info:\n")
print(data.info())

print("\nMissing Values Before Encoding:\n")
print(data.isnull().sum())


# --------------------------------------------------
# STEP 4: CLEAN TEXT DATA
# --------------------------------------------------

"""
Remove unwanted spaces and normalize text
to avoid NaN during encoding
"""

data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()


# --------------------------------------------------
# STEP 5: ENCODE CATEGORICAL COLUMNS
# --------------------------------------------------

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


# --------------------------------------------------
# STEP 6: HANDLE NaN CREATED DURING ENCODING
# --------------------------------------------------

print("Missing Values After Encoding:\n")
print(data.isnull().sum())

# Drop rows with NaN (safe because very few)
data.dropna(inplace=True)


# --------------------------------------------------
# STEP 7: FEATURE SELECTION
# --------------------------------------------------

X = data.drop(['loan_status', 'loan_id'], axis=1)
y = data['loan_status']

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape :", y.shape)


# --------------------------------------------------
# STEP 8: TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Completed")


# --------------------------------------------------
# STEP 9: MODEL TRAINING
# --------------------------------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("\nModel Training Completed")


# --------------------------------------------------
# STEP 10: MODEL EVALUATION
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))




# --------------------------------------------------
# STEP 12: MANUAL LOAN PREDICTION
# --------------------------------------------------

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


# --------------------------------------------------
# END OF PROGRAM
# --------------------------------------------------
