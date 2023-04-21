import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Regression Analysis
# Split the data into train-test sets 
X_train, X_test, y_train, y_test = train_test_split(ds_fraud.drop(columns = ["nameOrig", "nameDest", "isFraud", 
    "isFlaggedFraud"]), ds_fraud["isFraud"], test_size = 0.2, random_state = 42)

# Perform Logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)

# The training accuracy is 0.9982788143877837
print(log_reg.score(X_train, y_train))

# Predict using the regression model
y_pred = log_reg.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))

# The test accuracy is 
print(log_reg.score(X_test, y_test))

# Get the classification report
print(metrics.classification_report(y_test, y_pred, target_names = ["Not Fraudulent", "Fraudulent"]))

# TODO: LDA, QDA, k-means, SVM 