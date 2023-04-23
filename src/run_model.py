from data.make_dataset import CleanData
from features.build_features import BuildFeatures
from models.train_model import TrainModels
from models.predict_model import predictModels
from sklearn.model_selection import train_test_split

# Import the data
# The data can be found at https://www.kaggle.com/datasets/ealaxi/paysim1
fraud_data = CleanData.load_data("Fraud_Data.csv")

# Clean the data
fraud_data = CleanData.convert_to_numeric(fraud_data, ["step", "amount", "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"])
fraud_data = CleanData.rename_col(fraud_data, "oldbalanceOrg", "oldbalanceOrig")

# Feature engineering
fraud_data_final = BuildFeatures.convert_to_date(fraud_data, "step")
fraud_data_final = BuildFeatures.convert_to_dummy(fraud_data_final)

# Split the data into train-test sets 
X_train, X_test, y_train, y_test = train_test_split(fraud_data_final[["amount", "oldbalanceOrig", "newbalanceOrig", 
                                                                     "oldbalanceDest", "newbalanceDest", "day", "hour", 
                                                                     "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]], 
                                                    fraud_data_final["isFraud"], test_size = 0.2, random_state = 42)

# Train and test a logistic regression
scaler, fraud_reg = TrainModels.logistic_model(X_train, y_train, ["amount", "oldbalanceOrig", "newbalanceOrig", 
                                              "oldbalanceDest", "newbalanceDest", "day", "hour"])

log_report = predictModels.logistic_predict(X_test, y_test, scaler, ["amount", "oldbalanceOrig", "newbalanceOrig", "oldbalanceDest", 
                               "newbalanceDest", "day", "hour"], fraud_reg)

# print the classification report
print(log_report)