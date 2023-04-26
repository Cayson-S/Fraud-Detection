#################################################################################
# Author: Cayson Seipel
#
# Summary: This file contains the PredictModels class. This class contains three
# main functions. The first function returns a performance report containing
# a confusion matrix and a classification report. The second function
# classifies test data from a given logistic regression. The second function 
# classifies test data from a given support vector machine. For both the second 
# and third functions, a performance report is generated.
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

    def performance_report(y: pd.DataFrame, y_pred: pd.DataFrame, file_name: str = None) -> tuple:
        # Create a confusion matrix from the supplied data
        # The results are displayed to the console unless a file name is supplied
        # :param y: The target data
        # :type y: pd.DataFrame
        # :param y_pred: The predicted classes
        # :type y_pred: pd.DataFrame
        # :param file_name: The file title for the produced graph(s) 
        # :type file_name: str
        # :returns: the confusion matrix and classification report for the given model
        # :rtype: a tuple containing metrics.confusion_matrix and metrics.classification_report
        
        # Show the confusion matrix
        cm = metrics.confusion_matrix(y, y_pred, labels = [0, 1])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
        cm_display.plot(values_format = ".1f")
        
        # If a file name is specified, save the graphs to that file
        # Otherwise, show the plot
        if file_name != None:
            plt.savefig("./reports/figures/" + file_name, dpi = "figure", format = "png")
        else:
            plt.show()
        
        plt.clf()
        plt.close()

        return cm, metrics.classification_report(y, y_pred, target_names = ["Not Fraudulent", "Fraudulent"])

    def logistic_predict(X: pd.DataFrame, y: pd.DataFrame, log_model: sm.Logit, file_name: str = None) -> tuple:
        # This function predicts from an already fit logistic regression model
        # :param X: the prediction data
        # :type X: pd.DataFrame
        # :param y: the response variable (from the test set)
        # :type y: pd.DataFrame 
        # :param log_model: the logistic regression model
        # :type log_model: LogisticRegression 
        # :param file_name: The file title for the produced graph(s) 
        # :type file_name: str
        # :returns: the confusion matrix and classification report for the logistic regression
        # :rtype: a tuple containing metrics.confusion_matrix and metrics.classification_report

        # Predict the response variable
        y_pred = log_model.predict(sm.add_constant(X))

        # Print the accuracy
        as_binary = (y_pred >= 0.5).astype(int)

        # Get the performance report
        report = PredictModels.performance_report(y, as_binary, file_name)

        # return the performance report
        return report

    def svm_predict(X: pd.DataFrame, y: pd.DataFrame, svm_trained: SVC, file_name: str = None) -> tuple:
        # This function predicts from an already fit svm
        # :param X: the prediction data
        # :type X: pd.DataFrame
        # :param y: the response variable (from the test set)
        # :type y: pd.DataFrame 
        # :param svm_trained: the trained support vector machine
        # :type svm_trained: SVC 
        # :param file_name: The file title for the produced graph(s) 
        # :type file_name: str
        # :returns: the confusion matrix and classification report for the svm
        # :rtype: a tuple containing metrics.confusion_matrix and metrics.classification_report

        # Predict the response variable
        y_pred = svm_trained.predict(X)
        
        # Get the performance report
        report = PredictModels.performance_report(y, y_pred, file_name)

        # return the classification report
        return report