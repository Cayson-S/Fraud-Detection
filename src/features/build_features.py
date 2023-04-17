import pandas as pd

class BuildFeatures:
    def __init__(self, name):
        self.name = name

    def import_data(data_loc: string = "./data/external/Fraud_Data.csv") -> pd.DataFrame:
        data = pd.read_csv(data_loc)

        return data

    def rename_():