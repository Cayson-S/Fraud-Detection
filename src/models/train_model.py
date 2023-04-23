import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics

class TrainModels:
    def logistic_model(X: pd.DataFrame, y: pd.DataFrame, scalable_features: list) -> tuple:
        # This function fits a logistic regression model to the given data
        # It is recommended that training data be used (instead of the whole dataset)
        # :param X: the predictors (from the training set)
        # :type X: pd.Dataframe
        # :param y: the response variable (from the training set)
        # :type y: pd.Dataframe 
        # :param unscalable_features: the columns that should not be scaled (dummies et cetera)
        # :type unscalable_features: list
        # :returns: the scaler to be used for the test data and the logistic model
        # :rtype: a tuple containing StandardScaler and LogisticRegression

        # Scale the data
        X_scaled = X.copy()
        scaler = StandardScaler().fit(X_scaled[scalable_features])
        X_scaled[scalable_features] = scaler.transform(X_scaled[scalable_features])

        # Fit the logistic model
        log_reg = LogisticRegression(max_iter = 200).fit(X_scaled, y)

        # Get the training accuracy
        print("The logistic regression training accuracy is: ", log_reg.score(X_scaled, y))

        return scaler, log_reg