import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics

class TrainModel:
    def logistic_model(X: pd.DataFrame, y: pd.DataFrame) -> LogisticRegression:
        # This function fits a logistic regression model to the given data
        # It is recommended that training data be used (instead of the whole dataset)
        # :param X: the predictors (from the training set)
        # :type X: pd.Dataframe
        # :param y: the response variable (from the training set)
        # :type y: pd.Dataframe 
        # :returns: logistic model
        # :rtype: LogisticRegression

        # Fit the logistic model
        log_reg = LogisticRegression().fit(X, y)

        # Get the training accuracy
        print("The logistic regression training accuracy is: ", log_reg.score(X, y))

        return return log_reg