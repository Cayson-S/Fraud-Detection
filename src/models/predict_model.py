import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class predictModels:
    def logistic_predict(X: pd.DataFrame, y: pd.DataFrame, scaler: StandardScaler, scalable_features: list, 
                         log_model: LogisticRegression) -> metrics.classification_report:
        # This function predicts from an already fit logistic regression model
        # :param X: the prediction data
        # :type X: pd.DataFrame
        # :param y: the response variable (from the test set)
        # :type y: pd.DataFrame 
        # :param scaler: the function fit on training data from which to scale the test data
        # :type scaler: StandardScalar 
        # :param scalable_features: the list of columns that should be scalded
        # :type scalable_features: list
        # :param log_model: the logistic regression model
        # :type log_model: LogisticRegression 
        # :returns: the classification report for the logistic regression with the test data
        # :rtype: metrics.classification_report
        
        # Scale the test data
        X_scaled = X.copy()
        X_scaled[scalable_features] = scaler.transform(X_scaled[scalable_features])

        # Predict the response variable
        y_pred = log_model.predict(X_scaled)

        # Print the accuracy
        print(log_model.score(X_scaled, y))
        
        # return the classification report
        return metrics.classification_report(y, y_pred, target_names = ["Not Fraudulent", "Fraudulent"])

# TODO: LDA, QDA, k-means, SVM 