#%%

import ydata_profiling as pp
import pandas as pd
import matplotlib.pyplot as plt

# The date can be found at https://www.kaggle.com/datasets/ealaxi/paysim1
ds_fraud = pd.read_csv("./data/external/Fraud_Data.csv")

# Create a Pandas Profiling report
# The results are in fraud_detection_profile.html in the reports directory 
#profile = ds_fraud.profile_report()
#profile.to_file(output_file = "fraud_data_profile.html")

# Data cleansing 

# Convert the numeric data into appropriate numeric types
ds_fraud = ds_fraud[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "newbalanceDest", 
                              "isFraud", "isFlaggedFraud"]].apply(pd.to_numeric)

# Get some summary statistics on the original dataframe
print(ds_fraud.describe())

# What makes fraudulent transactions what they are?
ds_only_fraud = ds_fraud.loc[ds_fraud["isFraud"] == 1]

# Check that the dataframe was created correctly
print(ds_only_fraud.head())

# Get some summary statistics on the new dataframe
print(ds_only_fraud.describe())



"""
plt.hist(ds_only_fraud["amount"], bins = 7)
plt.show()

plt.hist(ds_only_fraud["oldbalanceOrg"], bins = 7)
plt.show()

plt.hist(ds_only_fraud["newbalanceOrig"], bins = 7)
plt.show()

plt.hist(ds_only_fraud["newbalanceDest"], bins = 7)
plt.show()

plt.hist(ds_only_fraud["isFlaggedFraud"], bins = 2)
plt.show()
"""