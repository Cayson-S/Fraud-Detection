#################################################################################
# Author: Cayson Seipel
#
# Summary: This file contains the TrainModels class. This class contains four 
# main functions. The first function scales the data using scikit-learn's 
# StandardScaler. The second function downsamples the data so that both classes
# are equally represented in the data. The third function fits a logistic
# regression model to the given data. Finally, the fourth function fits a
# support vector machine to the given data.
#################################################################################

import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import statsmodels.api as sm

class TrainModels:
    def __init__(self, name):
        self.name = name

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

    def logistic_model(X: pd.DataFrame, y: pd.DataFrame) -> sm.Logit:
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

        # Fit the statsmodels logistic model to determine features that have high p-values
        log_reg = sm.Logit(y, sm.add_constant(X)).fit_regularized()

        # Print the model summary to check p-values
        print("The logistic regression summary: ", log_reg.summary2())

        return log_reg

    def svm_model(X: pd.DataFrame, y: pd.DataFrame, C_val: float = 3.0, gamma_val: float = 3.0) -> SVC:
        # This function fits a Support Vector Machine to the given data
        # It is recommended that training data be used (instead of the whole dataset)
        # :param X: the predictors (from the training set)
        # :type X: pd.Dataframe
        # :param y: the response variable (from the training set)
        # :type y: pd.Dataframe 
        # :param C_val: the regularization parameter (greater than 0)
        # :type C_val: float
        # :param gamma: the kernel coefficient (greater than 0)
        # :type gamma: float
        # :returns: the support vector machine
        # :rtype: svm

        svm_train = SVC(kernel = "rbf", C = C_val, gamma = gamma_val).fit(X, y)

        return svm_train
