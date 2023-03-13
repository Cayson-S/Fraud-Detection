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


# What makes fraudulent transactions what they are?
ds_only_fraud = ds_fraud.loc[ds_fraud["isFraud"] == 1]

# Check that the dataframe was created correctly
print(ds_only_fraud.head())

# Get some summary statistics on the new dataframe
print(ds_only_fraud.describe())

#for col in ds_only_fraud.columns:
#    ds_only_fraud.hist(column = col)

    

