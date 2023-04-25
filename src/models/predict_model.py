#################################################################################
# Author: Cayson Seipel
#
# Summary: This file contains the PredictModels class. This class contains two
# main functions. The first function classifies test data from a given logistic
# regression. The second function classifies test data from a given support
# vector machine. For both functions, a confusion matrix is generated.
#################################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

class PredictModels:
    def __init__(self, name):
        self.name = name

    def logistic_predict(X: pd.DataFrame, y: pd.DataFrame, log_model: sm.Logit) -> tuple:
        # This function predicts from an already fit logistic regression model
        # :param X: the prediction data
        # :type X: pd.DataFrame
        # :param y: the response variable (from the test set)
        # :type y: pd.DataFrame 
        # :param log_model: the logistic regression model
        # :type log_model: LogisticRegression 
        # :returns: the confusion matrix and classification report for the logistic regression
        # :rtype: a tuple containing metrics.confusion_matrix and metrics.classification_report

        # Predict the response variable
        y_pred = log_model.predict(sm.add_constant(X))

        # Print the accuracy
        as_binary = (y_pred >= 0.5).astype(int)
        
        # Show the confusion matrix
        cm = metrics.confusion_matrix(y, as_binary, labels = [0, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
        cm_display.plot(values_format = ".1f")
        plt.show()

        # return the classification report
        return cm, metrics.classification_report(y, as_binary, target_names = ["Not Fraudulent", "Fraudulent"])

    def svm_predict(X: pd.DataFrame, y: pd.DataFrame, svm_trained: SVC) -> tuple:
        # This function predicts from an already fit svm
        # :param X: the prediction data
        # :type X: pd.DataFrame
        # :param y: the response variable (from the test set)
        # :type y: pd.DataFrame 
        # :param svm_trained: the trained support vector machine
        # :type svm_trained: SVC 
        # :returns: the confusion matrix and classification report for the svm
        # :rtype: a tuple containing metrics.confusion_matrix and metrics.classification_report

        # Predict the response variable
        y_pred = svm_trained.predict(X)
        
        # Show the confusion matrix
        cm = metrics.confusion_matrix(y, y_pred, labels = [0, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
        cm_display.plot(values_format = ".1f")
        plt.show()

        # return the classification report
        return cm, metrics.classification_report(y, y_pred, target_names = ["Not Fraudulent", "Fraudulent"])