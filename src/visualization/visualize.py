#################################################################################
# Author: Cayson Seipel
#
# Summary: This file contains the Visualize class which contains several
# methods for visualizing data. transaction_boxplots creates boxplots
# for the supplied data. It also supports side-by-side boxplots if a second
# dataset is supplied. The second function creates a pandas profile for the
# data and saves it directly to a file in ./reports. Finally, the third 
#################################################################################

import ydata_profiling as pp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualize:
    def __init__(self, name):
        self.name = name

    def transaction_boxplots(data1: pd.Series, data2: pd.Series, title: str, y_label: str, x_label1: str,
                             x_label2: str = None, file_name: str = None) -> None:
        # Creates up to two boxplots from the supplied data
        # If a filename is supplied, it is saved to "./reports/figures/"
        # :param data1: the dataset to visualize
        # :type data1: pd.Series
        # :param data2: the second dataset to visualize
        # :type data2: pd.Series
        # :param title: The title of the graph
        # :type title: str
        # :param y_label: The title for the y-axis
        # :type y_label: str
        # :param x_label1: The title for the x-axis (the first graph)
        # :type x_label1: str
        # :param x_label2: The title for the x-axis (the second graph)
        # :type x_label2: str
        # :param file_name: The file title for the produced graph(s) 
        # :type file_name: str
        # :returns: nothing is returned
        # :rtype: None

        plt.figure(figsize = (9, 5))
        plt.suptitle(title, x = 0.35)
        plt.ticklabel_format(style = "plain")
        ax1 = plt.subplot(131)
        plt.tight_layout() 
        plt.boxplot(data1)
        plt.xticks(ticks = [])
        plt.xlabel(x_label1)
        plt.ylabel(y_label)
    
        # Data for a second boxplot was supplied
        if data2.any():
            ax2 = plt.subplot(132, sharey = ax1)
            plt.tight_layout()
            plt.boxplot(data2)
            plt.xticks(ticks = [])
            plt.xlabel(x_label2)
    
        # If a file name is specified, save the graphs to that file
        # Otherwise, show the plot
        if file_name != None:
            plt.savefig("./reports/figures/" + file_name, dpi = "figure", format = "png", bbox_inches = "tight")
        else:
            plt.show()

        plt.clf()
        plt.close()

        return None

    def data_profile(data: pd.DataFrame) -> None:
        # Create a Pandas Profiling report
        # The results are saved to fraud_detection_profile.html in the reports directory 
        # :param data: The dataset to be profiled
        # :type data: pd.DataFrame
        # :returns: nothing is returned
        # :rtype: None

        profile = data.profile_report()
        profile.to_file(output_file = "fraud_data_profile.html")

        return None

    def correlation_matrix(data: pd.DataFrame, file_name: str = None) -> None:
        # Create a correlation matrix from the supplied data
        # The results are displayed to the console unless a file name is supplied
        # :param data: The dataset to graph
        # :type data: pd.DataFrame
        # :param file_name: The file title for the produced graph(s) 
        # :type file_name: str
        # :returns: nothing is returned
        # :rtype: None

        corr_matrix = data.corr()

        sns.heatmap(corr_matrix, cmap = "YlGnBu", annot = True, fmt = "0.2f", annot_kws = {"fontsize": 8})

        # If a file name is specified, save the graphs to that file
        # Otherwise, show the plot
        if file_name != None:
            plt.savefig("./reports/figures/" + file_name, dpi = "figure", format = "png", bbox_inches = "tight")
        else:
            plt.show()

        return None
