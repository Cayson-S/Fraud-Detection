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

        file_name = "../fraud_detection/data/external/" + file
        data = pd.read_csv(file_name, sep = ",")

        return data
    
    def convert_to_numeric(data: pd.DataFrame, cols: list) -> pd.DataFrame:
        # This function converts a list of columns to a numeric datatype
        # :param data: the dataset
        # :type data: pd.Dataframe
        # :param cols: columns to be converted to a numeric datatype
        # :type cols: list 
        # :returns: dataframe
        # :rtype: pd.dataframe 

        data_copy = data.copy()

        # Convert the numeric data into appropriate numeric types
        data_copy[cols].apply(pd.to_numeric)

        return data_copy

    def rename_col(data: pd.DataFrame, col: str, col_name: str) -> pd.DataFrame:
        # This function renames a column
        # :param data: the dataset
        # :type data: pd.Dataframe
        # :param col: column to be renamed
        # :type col: string
        # :param col_name: new column name
        # :type col_name: string
        # :returns: dataframe
        # :rtype: pd.dataframe 

        # Rename a column 
        data_copy = data.rename(columns = {col: col_name})

        return data_copy