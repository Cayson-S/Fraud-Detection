from data.make_dataset import CleanData
from features.build_features import BuildFeatures
from models.train_model import TrainModels
from models.predict_model import predictModels
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", 20)

# Import the data
# The data can be found at https://www.kaggle.com/datasets/ealaxi/paysim1
fraud_data = CleanData.load_data("Fraud_Data.csv")

# Clean the data
fraud_data = CleanData.convert_to_numeric(fraud_data, ["step", "amount", "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"])
fraud_data = CleanData.rename_col(fraud_data, "oldbalanceOrg", "oldbalanceOrig")

# Feature engineering
fraud_data_clean = BuildFeatures.convert_to_date(fraud_data, "step")
fraud_data_clean = BuildFeatures.convert_to_dummy(fraud_data_clean)

corr_matrix = fraud_data_clean.corr()

sns.heatmap(corr_matrix, cmap = "YlGnBu", annot = True)
plt.show()

# Remove highly correlated columns
# The columns removed are already captured by the amount and original balance columns
fraud_data_downsample = TrainModels.downsample(fraud_data_clean.drop(["newbalanceOrig", "newbalanceDest"], 
                                                                      axis = 1), "isFraud")

# Split the data into train-test sets 
X_train, X_test, y_train, y_test = train_test_split(fraud_data_downsample[["amount", "oldbalanceOrig", "oldbalanceDest", 
                                                                           "day", "hour", "CASH_OUT", "DEBIT", "PAYMENT", 
                                                                           "CASH_IN", "TRANSFER"]], 
                                                    fraud_data_downsample["isFraud"], test_size = 0.2, random_state = 42)

# Scale the training data for the two models
scaler, X_train_scaled = TrainModels.scale_data(X_train, ["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour"])


# Train a logistic regression
# By using backwards stepwise feature elimination, all but the following were eliminated for having high p-values or coefficients equalling zero
fraud_reg = TrainModels.logistic_model(X_train_scaled[["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour", "TRANSFER"]], y_train)

# Train the smv model
fraud_svm = TrainModels.svm_model(X_train_scaled, y_train)

# Scale the test data
scaler, X_test_scaled = TrainModels.scale_data(X_test, ["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour"], scaler)

# Predict on each of the two models and get the classification reports
log_report = predictModels.logistic_predict(X_test_scaled[["amount", "oldbalanceOrig", "oldbalanceDest", "day", "hour", "TRANSFER"]], y_test, fraud_reg)
svm_report = predictModels.svm_predict(X_test_scaled, y_test, fraud_svm)

# print the logistic classification report
print("==========================================================")
print("The Logistic Classification Report:")
print(log_report[1])

# print the svm classification report
print("==========================================================")
print("The SVM Classification Report:")
print(svm_report[1])
