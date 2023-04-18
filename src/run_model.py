from data.make_dataset import CleanData
from features.build_features import BuildFeatures
from models.train_model import TrainModel

# Import the data
# The data can be found at https://www.kaggle.com/datasets/ealaxi/paysim1
fraud_data = CleanData.load_data("Fraud_Data.csv")

# Clean the data
fraud_data = CleanData.convert_to_numeric(fraud_data, ["step", "amount", "oldbalanceOrg", "newbalanceOrig", 
        "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"])


fraud_data_final = BuildFeatures.convert_to_date(fraud_data, "step")
fraud_data_final = BuildFeatures.convert_to_dummy(fraud_data_final)

print(fraud_data_final.head())