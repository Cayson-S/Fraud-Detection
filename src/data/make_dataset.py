# -*- coding: utf-8 -*-
import pandas as pd

class CleanData:
    def __init__(self, name):
        self.name = name
    
    def load_data(file: str) -> pd.DataFrame:
        # This function reads the data from the external file
        # :param file: file name
        # :type file: str
        # :returns: dataframe
        # :rtype: pd.dataframe 

        file_name = "../data/external/" + file
        data = pd.read_csv(file_name, sep = " ", header = True)

        return data
    
    def convert_to_numeric(data: pd.Dataframe, cols: list) -> pd.DataFrame:
        # This function converts a list of columns to a numeric datatype
        # :param data: the dataset
        # :type data: pd.Dataframe
        # :param cols: columns to be converted to a numeric datatype
        # :type cols: list 
        # :returns: dataframe
        # :rtype: pd.dataframe 

        # Convert the numeric data into appropriate numeric types
        data[cols].apply(pd.to_numeric)

        return data
    
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