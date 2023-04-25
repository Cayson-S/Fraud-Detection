#################################################################################
# Author: Cayson Seipel
#
# Summary: The goal of this project is to predict whether transactions are fraudulent or
# not. I use two models (logistic regression and support vector machine) to
# model the data. 
#################################################################################

from data.make_dataset import CleanData
from features.build_features import BuildFeatures
from models.train_model import TrainModels
from models.predict_model import PredictModels
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


#################################################################################
# Import Data
#################################################################################

# Show all of the columns when printing to the console
pd.set_option("display.max_columns", 20)

# Import the data
# The data can be found at https://www.kaggle.com/datasets/ealaxi/paysim1
fraud_data = CleanData.load_data("Fraud_Data.csv")


#################################################################################
# Data Cleansing
#################################################################################

# Convert all of the numeric columns into numeric datatypes (the code would throw errors because the data was nonnumeric)
fraud_data = CleanData.convert_to_numeric(fraud_data, ["step", "amount", "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"])

# rename a column to be consistent with other columns 
fraud_data = CleanData.rename_col(fraud_data, "oldbalanceOrg", "oldbalanceOrig")


#################################################################################
# Feature Engineering
#################################################################################

# Convert the step column into day and hour fields  
# The original column was a date column representing hours since the beginning of the month (there is only one month of data)
fraud_data_clean = BuildFeatures.convert_to_date(fraud_data, "step")

# Convert the transaction type to dummy variables
fraud_data_clean = BuildFeatures.convert_to_dummy(fraud_data_clean)


#################################################################################
# Data Visualization
#################################################################################
corr_matrix = fraud_data_clean.corr()

sns.heatmap(corr_matrix, cmap = "YlGnBu", annot = True)
plt.show()


#################################################################################
# Classification Preparation
#################################################################################

# Remove highly correlated columns
# The columns removed are already captured by the amount and original balance columns
# The data is imbalanced, so I downsample the non-fraud data to balance the classes  
fraud_data_downsample = TrainModels.downsample(fraud_data_clean.drop(["newbalanceOrig", "newbalanceDest"], 
                                                                      axis = 1), "isFraud")

# Split the data into train-test sets 
X_train, X_test, y_train, y_test = train_test_split(fraud_data_downsample[["amount", "oldbalanceOrig", "oldbalanceDest", 
                                                                           "day", "hour", "CASH_OUT", "DEBIT", "PAYMENT", 
                                                                           "CASH_IN", "TRANSFER"]], 
                                                    fraud_data_downsample["isFraud"], test_size = 0.2, random_state = 42)

# Scale the training data for the two models
scaler, X_train_scaled = TrainModels.scale_data(X_train, ["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour"])

# Scale the test data for both models
X_test_scaled = TrainModels.scale_data(X_test, ["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour"], scaler)[1]


#################################################################################
# Logistic Regression and Support Vector Machine
#################################################################################

# Train a logistic regression
# Through using backwards stepwise feature elimination, all but the following were eliminated for having high p-values
fraud_reg = TrainModels.logistic_model(X_train_scaled[["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour", "TRANSFER"]], y_train)

# Predict with the logistic model on the test data
log_report = PredictModels.logistic_predict(X_test_scaled[["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour", "TRANSFER"]], y_test, fraud_reg)

# Train the smv model
fraud_svm = TrainModels.svm_model(X_train_scaled, y_train)

# Predict on the SVM model using the test data
svm_report = PredictModels.svm_predict(X_test_scaled, y_test, fraud_svm)


#################################################################################
# Classification Reports
#################################################################################

# print the logistic classification report
print("==========================================================")
print("The Logistic Classification Report:")
print(log_report[1])

# print the SVM classification report
print("==========================================================")
print("The SVM Classification Report:")
print(svm_report[1])
