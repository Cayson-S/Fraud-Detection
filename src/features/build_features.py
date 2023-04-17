import pandas as pd

class BuildFeatures:
    def __init__(self, name):
        self.name = name

    def convert_to_date(data: pd.DataFrame, col: str) -> pd.Dataframe:
        # This function converts a column into a date field (an hour and a day column)
        # The original column is an integer type representing the hours since the beginning of the month
        # :param data: the dataset
        # :type data: pd.Dataframe
        # :param col: column location to be converted to a date field
        # :type col: string 
        # :returns: dataframe
        # :rtype: pd.dataframe 

        # determine the day and hour of the transaction
        # Add the data to new columns
        for i in range(len(data)):
            data["day"][i] = data[col][i]//24
            data["hour"][i] = data[col][i] % 24
        
        # Remove the old column from the dataset
        data.drop(col, index = 1, inplace = True)

        return data