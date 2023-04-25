import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

class TrainModels:
    def scale_data(to_scale: pd.DataFrame, scalable_features: list, scaler: StandardScaler = None) -> tuple:
        # :param to_scale: the dataset to scale
        # :type to_scale: pd.Dataframe
        # :param scalable_features: the columns that should be scaled
        # :type scalable_features: list
        # :returns: the scaler to be used for the test data and the scaled data
        # :rtype: a tuple containing StandardScaler and pd.DataFrame
        
        # Scale the data
        data_scaled = to_scale.copy()

        # No scaler was given to the function
        if not scaler:
            scaler = StandardScaler().fit(data_scaled[scalable_features])
        
        data_scaled[scalable_features] = scaler.transform(data_scaled[scalable_features])

        return scaler, data_scaled

    def downsample(data: pd.DataFrame, y: str) -> pd.DataFrame:
        # This function downsamples the data to fix unbalanced classes
        # :param data: the dataset to be downsampled
        # :type data: pd.Dataframe
        # :param y: the response variable
        # :type y: pd.Dataframe 
        # :returns: downsampled data
        # :rtype: pd.DataFrame

        fraud_data = data[data[y] == 1]

        downsampled_data = pd.concat([resample(data[data[y] == 0], replace = False, 
                                      n_samples = len(fraud_data), random_state = 42), fraud_data])

        return downsampled_data

    def logistic_model(X: pd.DataFrame, y: pd.DataFrame, num_iter: int = 100) -> sm.Logit:
        # This function fits a logistic regression model to the given data
        # It is recommended that training data be used (instead of the whole dataset)
        # :param X: the predictors (from the training set and, ideally, scaled)
        # :type X: pd.Dataframe
        # :param y: the response variable (from the training set)
        # :type y: pd.Dataframe 
        # :param num_iter: the maximum number of iterations allowed for the model
        # :type num_iter: int 
        # :returns: the logistic model
        # :rtype: LogisticRegression

        best_feautures = X.copy()

        # Fit the statsmodels logistic model to determine features that have high p-values
        logit = sm.Logit(y, sm.add_constant(best_feautures)).fit_regularized(method = "l1", maxiter = num_iter)

        # Print the model summary to check p-values
        print("The logistic regression summary: ", logit.summary2())



        return logit

    #def knn_model(X: pd.DataFrame, y: pd.DataFrame, num_iter: int = 2000) -> svm:
        # This function fits a Support Vector Machine to the given data
        # It is recommended that training data be used (instead of the whole dataset)
        # :param X: the predictors (from the training set)
        # :type X: pd.Dataframe
        # :param y: the response variable (from the training set)
        # :type y: pd.Dataframe 
        # :param num_iter: the maximum number of iterations allowed for the model
        # :type num_iter: int 
        # :returns: the support vector machine
        # :rtype: svm

        #svm_model = svm.LinearSVC(random_state = 42, tol = 1e-5, max_iter = num_iter).fit(X, y)

        #return svm_model
