import ydata_profiling as pp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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