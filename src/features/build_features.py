import pandas as pd
import sys

class BuildFeatures:
    def __init__(self, name):
        self.name = name

    def convert_to_date(data: pd.DataFrame, col: str) -> pd.DataFrame:
        # This function converts a column into a date field (an hour and a day column)
        # The original column is an integer type representing the hours since the beginning of the month
        # :param data: the dataset
        # :type data: pd.Dataframe
        # :param col: column location to be converted to a date field
        # :type col: string 
        # :returns: dataframe
        # :rtype: pd.dataframe 
        
        # Copy the data so as not to change the original dataset
        data_copy = data.copy()

        # determine the day and hour of the transaction
        # Add the data to new columns
        data_copy["day"] = data_copy[col] // 24
        data_copy["hour"] = data_copy[col] % 24
        
        # Remove the old column from the dataset
        data_copy.drop(col, axis = 1, inplace = True)

        return data_copy

    def convert_to_dummy(data: pd.DataFrame, remove_cash_in: bool = True) -> pd.DataFrame:
        # This function converts a categorical column into dummy variables
        # The original column is a string representing the column to be one-hot encoded
        # IMPORTANT: The dataset must contain the column "type" which has multiple rows
        #            One of the rows must be "CASH_IN"
        # :param data: the dataset
        # :type data: pd.Dataframe
        # :returns: dataframe
        # :rtype: pd.dataframe 
        
        # Copy the data so as not to change the original dataset
        data_copy = data.copy()

        # Ensure that the data entered fits the function
        try:
            # one-hot encoding - convert the type parameter to dummy variables
            data_copy = pd.get_dummies(data_copy, columns = ["type"], prefix = "", prefix_sep = "")
        except:
            print("the column \"type\" does not exist in the data.")
            sys.exit(1)

        if (remove_cash_in == True):
            # Ensure that the data contains a "CASH_IN" column
            try:
                data_copy.drop("CASH_IN", axis = 1, inplace = True)
            except:
                print("\"CASH_IN\" does not exist in the data.")

        return data_copy