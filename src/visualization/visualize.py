import ydata_profiling as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# The date can be found at https://www.kaggle.com/datasets/ealaxi/paysim1
ds_fraud = pd.read_csv("./data/external/Fraud_Data.csv")

# Data cleansing 
# Convert the numeric data into appropriate numeric types
ds_fraud[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", 
                              "isFraud", "isFlaggedFraud"]].apply(pd.to_numeric)

# Feature engineering
# Convert the type parameter to dummy variables
ds_fraud = pd.get_dummies(ds_fraud, columns = ["type"], prefix = "", prefix_sep = "")

# Look only at fraud data
is_fraud = ds_fraud.loc[ds_fraud["isFraud"] == 1]

# Get a bar plot of fraud data by transaction type
for col in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
    plt.bar(col, sum(is_fraud[col]), 0.4, color = "red")

plt.show()

# One-hot encoding - remove one of the unnecessary columns
ds_fraud.drop("CASH_IN", axis = 1, inplace = True)

# Creates up to two boxplots from the supplied data
# If a filename is supplied, it is saved to "./reports/figures/"
def transaction_boxplots(data1: pd.Series, data2: pd.Series, title: str, y_label: str, x_label1: str,
                         x_label2: str = None, file_name: str = None):
    plt.figure(figsize = (9, 5))
    plt.suptitle(title, x = 0.35)
    plt.ticklabel_format(style = "plain")
    ax1 = plt.subplot(131)
    plt.tight_layout() 
    plt.boxplot(data1)
    plt.xticks(ticks = [])
    plt.xlabel(x_label1)
    plt.ylabel(y_label)
    
    # Data for a second boxplot was supplied
    if data2.any():
        ax2 = plt.subplot(132, sharey = ax1)
        plt.tight_layout()
        plt.boxplot(data2)
        plt.xticks(ticks = [])
        plt.xlabel(x_label2)
    
    if file_name != None:
        plt.savefig("./reports/figures/" + file_name, dpi = "figure", format = "png", bbox_inches = "tight")
    else:
        plt.show()

# Create a Pandas Profiling report
# The results are in fraud_detection_profile.html in the reports directory 
profile = ds_fraud.profile_report()
profile.to_file(output_file = "fraud_data_profile.html")

# Get some summary statistics on the original dataframe and save it to a csv
ds_fraud.describe().to_csv("./reports/summary_stats_fraud_data.csv")

# Side-by-side boxplot comparison of transaction amounts by fraud type
transaction_boxplots(data1 = ds_fraud[ds_fraud["isFraud"] == 0]["amount"], 
   data2 = ds_fraud[ds_fraud["isFraud"] == 1]["amount"], title = "Transaction Amounts by Fraud Type",
   x_label1 = "Non-Fraudulent", x_label2 = "Fraudulent", 
   y_label = "Transaction Amount (In Millions Of Local Currency)", file_name = "amount_fraud_comparison.png")

# Side-by-side boxplot comparison of oldbalanceOrg by fraud type
transaction_boxplots(data1 = ds_fraud[ds_fraud["isFraud"] == 0]["oldbalanceOrg"], 
    data2 = ds_fraud[ds_fraud["isFraud"] == 1]["oldbalanceOrg"], title = "Originator Original Balance by Transaction Fraud Type",
    x_label1 = "Non-Fraudulent", x_label2 = "Fraudulent", 
    y_label = "Original Balance (In Millions Of Local Currency)", file_name = "original_balance_fraud_comparison.png")

# Side-by-side boxplot comparison of newbalanceOrg by fraud type
transaction_boxplots(data1 = ds_fraud[ds_fraud["isFraud"] == 0]["newbalanceOrig"], 
    data2 = ds_fraud[ds_fraud["isFraud"] == 1]["newbalanceOrig"], title = "Originator New Balance by Transaction Fraud Type",
    x_label1 = "Non-Fraudulent", x_label2 = "Fraudulent", 
    y_label = "New Balance (In Millions Of Local Currency)", file_name = "new_balance_fraud_comparison.png")

# Side-by-side boxplot comparison of oldbalanceDest by fraud type
transaction_boxplots(data1 = ds_fraud[ds_fraud["isFraud"] == 0]["oldbalanceDest"], 
    data2 = ds_fraud[ds_fraud["isFraud"] == 1]["oldbalanceDest"], title = "Destination Original Balance by Transaction Fraud Type",
    x_label1 = "Non-Fraudulent", x_label2 = "Fraudulent", 
    y_label = "Original Balance (In Millions Of Local Currency)", file_name = "original_destination_fraud_comparison.png")

# Side-by-side boxplot comparison of newbalanceDest by fraud type
transaction_boxplots(data1 = ds_fraud[ds_fraud["isFraud"] == 0]["newbalanceDest"], 
    data2 = ds_fraud[ds_fraud["isFraud"] == 1]["newbalanceDest"], title = "Destination New Balance by Transaction Fraud Type",
    x_label1 = "Non-Fraudulent", x_label2 = "Fraudulent", 
    y_label = "New Balance (In Millions Of Local Currency)", file_name = "new_destination_fraud_comparison.png")

# -----------------------------------------------------------------------------------------

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