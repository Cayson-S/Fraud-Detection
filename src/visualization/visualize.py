import ydata_profiling as pp
import pandas as pd

# The date can be found at https://www.kaggle.com/datasets/ealaxi/paysim1
ds_fraud = pd.read_csv("./data/Fraud_Data.csv")

# Create a Pandas Profiling report
# The results are in fraud_detection_profile.html in the reports directory 
#profile = ds_fraud.profile_report()
#profile.to_file(output_file = "fraud_data_profile.html")

#https://github.com/Cayson-S/Fraud-Detection