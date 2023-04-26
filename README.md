# Fraud Detection Project

Author: Cayson Seipel

Date: 4/26/2023

This project looks to classify transaction data as fraudulent. The data is available to download from https://www.kaggle.com/datasets/ealaxi/paysim1

## Background

<p>&emsp; Monetary transactions are frequent occurrences as customers purchase products and services. While this behavior is normal, bad actors can subvert this process by making fraudulent transactions. To minimize the damage done by these bad actors, mobile money services must proactively flag transactions that are likely fraudulent. Unfortunately, fraudulent transactions can be almost indistinguishable from legitimate ones. They can also occur at any time. This combination of factors makes transactions impossible for humans to speedily analyze for fraud. Thus, to protect their customers, mobile money services must invest in a data science solution that utilizes statistical analyses to accurately identify and block fraudulent transactions.</p>

<p>&emsp;The simulated data was created from a month of confidential transaction data sourced from a mobile money service company. The company operates in Africa. The goal of this project is to predict whether transactions are fraudulent or not by utilizing several statistical classification models. The resulting models can be used on other transaction data from the company to flag fraudulent transactions more accurately.</p>

## Data Preparation

<p>&emsp;Before any analyses could be performed on the data, I needed to prepare the data. The data was relatively clean, so I did not have to perform many operations to make the data usable. First, I transformed the numeric fields into numeric datatypes (integers and floats), and I renamed a column to follow the naming conventions of the data. Second, I converted a date field – specifying how many hours since the beginning of the month a transaction occurred – into a day and hour field for better interpretability. Finally, I converted a categorical feature describing the transaction type into multiple dummy variables. That is, each transaction type was given a column that specified whether any given transaction was of that transaction type. After completing these steps, the data could be explored through visualizations.</p>

## Data Visualization

<p>&emsp;To better understand the data, I visualized fraudulent and non-fraudulent data and compared their transaction details. With a few exceptions, the fraudulent data’s transaction details were very similar to data that was not fraudulent. The features that were different include the currency amount of a transaction, the balance in the originating account, and the balance in the destination account. These fields will be necessary for the analyses. Additionally, it did not appear that the transaction types were linearly separable. That is, it did not appear that a straight line could be drawn to separate fraudulent from non-fraudulent transactions. The fact that the data is likely non-linear means that the classification model that best predicts a transaction’s type will likely be non-linear. I also created a correlation matrix to determine if there was multicollinearity in the data. There was a strong correlation between the new balances of the originator and destination accounts and the old balances of both accounts. As the information in the new balance fields is already captured by the old balances for both accounts and the total monetary amount of each transaction, I dropped both features.</p>

## Model Choice and Details

<p>&emsp;Based on the information gleaned on which types of models will likely perform better on the data, I chose to use logistic regression and a support vector machine (SVM) with a non-linear kernel. The logistic regression will likely not perform as well because it is a linear classification model. However, logistic models are much easier to explain than more complex models, such as SVMs. To allow the models to be tested, I split the data, with 20 percent being held out for testing and the rest being used to train the models. I also scaled the data so that each non-categorical feature would have a mean of zero and a standard deviation of one. Scaling the data ensures that different feature scales do not ‘overpower’ the applied models to make features on a smaller scale effectively insignificant when they may be significant. The data is now ready to be modeled.</p>

<p>&emsp;Two models were fit to the training data: logistic regression and a support vector machine (SVM) using a non-linear kernel. Both models are classification models. The first model, logistic regression, follows the equation shown in Equation 1, where β represents the intercept and each coefficient and X represents each feature for a given data point (transaction). Importantly, the function returns a value between 0 and 1, which is the odds that the given transaction is fraudulent. A transaction is assumed to be fraudulent if the returned odds are larger than 0.5. Thus, if a transaction was input into the logistic model and the model returned 0.7, the model suggests that there is a 70 percent chance that the transaction is fraudulent.</p>

<br></br>

<b>Equation 1</b>
Logistic Regression

<img width="380" alt="image" src="https://user-images.githubusercontent.com/71890506/234708160-29905461-d0a0-4c51-b736-16a1a910c0d5.png">

<br></br>

<p>&emsp;The second model, a support vector machine (SVM), was also trained on the data. In general, an SVM separates data by placing a hyperplane between classes. When using am SVM with a non-linear kernel, the SVM expands the data to a higher dimension where the data becomes linearly separable. An example of a non-linear SVM is shown in Figure 1. In the figure, the non-linear SVM can separate the purple class from the blue class surrounding it. Because of its statistical complexity, this method is much more difficult to explain than the logistic regression. However, it does output a prediction value (fraudulent or non-fraudulent) that is easy to understand.</p>

<br></br>

<b>Figure 1</b>

Non-Linear SVM

<img width="380" alt="image" src="https://user-images.githubusercontent.com/71890506/234708397-2f057bd1-a719-40dc-ae7c-27e3e8017fc7.png">

<i>Note.</i> This image was sourced from Liu, Y. (2023). 

<br></br>

## Model Performance and Comparison

<p>&emsp;After training both models, they must be tested to determine if they can accurately predict fraud using data they have not yet been exposed to. In addition, the original data includes information on whether the current fraud-prediction model has predicted each transaction as fraudulent. This model will be the baseline to test the logistic and support vector machine (SVM) models. A report was created for each model to compare them. The associated confusion matrices are shown in Figure 3, Figure 4, and Figure 5.</p>

<br></br>

<b>Figure 2</b>

Base Model Confusion Matrix

![image](https://user-images.githubusercontent.com/71890506/234709417-87360e61-6482-4669-9315-bfca088eacf8.png)

<br></br>

<b>Figure 3</b>

Logistic Regression Confusion Matrix

![image](https://user-images.githubusercontent.com/71890506/234709453-e6cc3910-d77a-42db-8df8-0510ac98cc9f.png)

<br></br>

<b>Figure 3</b>
Support Vector Machine Confusion Matrix

![image](https://user-images.githubusercontent.com/71890506/234709373-330ad87d-5e8b-4b7e-9b37-e4eb75a06ea5.png)

<br></br>

<p>&emsp;For each confusion matrix, the top-left box represents the transactions correctly predicted to be non-fraudulent. The bottom-right box represents the transactions correctly identified as fraudulent. The other two boxes represent incorrect predictions. In Figure 3, the base model does not accurately predict non-fraudulent transactions. Instead, it selected almost 8200 transactions as non-fraudulent when they were the opposite. This base model predicted close to zero percent of the total fraudulent transactions.</p>

<p>&emsp;Figure 4 represents the predictions made using the testing data on the logistic regression model. This model performed significantly better than the base model. The logistic model was able to accurately flag 79 percent of all fraudulent transactions for a total of almost 1300 fraudulent transactions. This drastic increase in accuracy suggests that fraudulent transactions contain specific features that make them unique compared to non-fraudulent transactions. </p>

<p>&emsp;Figure 5 represents the predictions made using the testing data on the support vector machine (SVM) model. As suspected, the nonlinear model could accurately separate fraudulent from non-fraudulent transactions. The SVM was able to flag 94 percent of all fraudulent transactions. This accuracy resulted in a total of over 1500 transactions being correctly flagged. </p>

## Recommendation

<p>&emsp;Given the results, I would recommend an overhaul of the current fraud detection system. Specifically, I would suggest that the current model be replaced with the support vector machine model. While logistic regression is easier to explain, the chosen model would be implemented in the backend, flagging transactions for potential fraud and requesting some action. Since this model would not require an explanation to perform these actions, I recommend implementing the more accurate model (the support vector machine). Concerning ethics and bias, if the method is implemented so that flagged transactions are completely stopped, then there may be some ethical concerns surrounding people who consistently perform transactions that might be deemed as fraudulent. An example of this is an individual who is living paycheck to paycheck who empties their bank account after every pay period. No ethical or bias considerations have been identified if this model only asks for more information about the transaction instead of completely shutting down a potentially fraudulent transaction.</p>

## Next Steps

<p>&emsp;Given that the support vector machine model is the best option to be implemented by the mobile money service, the model should be tested using real data from the company. If the test results continue to prove the model is accurate, it should replace the original model. However, if the model is shown to be inaccurate when using non-synthetic data, information should be collected on how the real data is different. A new model should be fit that takes this information into account.</p>

## Conclusion

<p>&emsp;In summation, I utilized two different models to predict fraud from synthetic transaction data. This data was created by a mobile money service that operates in an African country. The current model utilized by the company cannot accurately ascertain the fraudulence of a transaction. As such, both models that I fit to the data (logistic regression and a support vector machine) significantly outperformed it. Based on these results, I suggest replacing the base model with the highest-performing model found. The highest-performing model was the support vector machine which accurately flagged 94 percent of all fraudulent transactions. Successful implementation of the top model will save time and money for the mobile money services’ customers. These customers will now be less likely to be a victim of fraud. It will also increase the reputation of the money service in that it is now capable of better protecting its customers.</p>

## References

Liu, Y. (2023). <i>svm</i> [Lecture Notes]. Middle Tennessee State University.



## Project Organization

<p>&emsp; To use this project, run run_model.py. This file imports all of the necessary scripts and performs the entire analysis. The visualizations are saved under ./reports/figures. The data is stored in ./data/external.</p>


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── run_model.py   <- Runs all of the scripts.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
