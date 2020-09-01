# Sentiment Analysis, NLP and Named Entity Recognition

__Title__: Tales from crypto <br />
__Submitted by__: Amar Munipalle <br />

## 1. Introduction

The underlying dataset classifies mortgage loans into various risk categories. Explatory variables include DSR, home ownership patterns, borrower income and derogatory notes on file. Key challenge in dataset is the imbalanced nature of the class and usage of 
Imbalanced class here meaning - significantly more which are low risk vs. current in dataset

| Class |Numbers |
| ---- | ------- |
| Low Risk | 75036|
| High Risk |  2500 |


### 1.1 Technical Tools Used
Imbalanced learn - a python package offering a number of re-sampling techniques commonly used in datasets showing strong between-class imbalance.   
Standard scikit learn machine learning tools

* [Imbalanced Learn](https://pypi.org/project/imbalanced-learn/)

### 1.2 Pre-processing prior to application of logistic regression

![Credit Risk](images/credit-risk.jpg)

a. Features extracted from dataset by excluding risk classification
b. Standard scaler package is applied on the non categorical components of dataset (all except homeowner)
c. Onehot encoder applied to homeowner to derive a sparse dummy variable matrix
d. Logistic regression applied to both the scaled and non scaled datasets
e. Scaled model performs significantly better per balance accuracy scores and recall metrics below. Stated differently this model predicts the high risk scores better in this imbalanced asset class
| Metric |Scaled |Non Scaled |
| -------| ------|-----------|
| BAS    | 0.954 |   0.989   |
| Recall |  0.91 |     0.98  |


## 2. Performance of oversampling and undersampling models

The following oversampling, undersampling and combination models were applied

* [Random Oversampling](https://pypi.org/project/imbalanced-learn/)
* [SMOTE ](https://pypi.org/project/imbalanced-learn/#id30)
* [SMOTENN](https://pypi.org/project/imbalanced-learn/#id33)

### Comparison of sampling models

| Metric |Naive-ROS |SMOTE-OS |Cluster-US|SMOTENN-Comb|
| -------| ---------|---------|----------|------------|
| BAS    | 0.994    |   0.95 |  0.982   |  0.994     |
| Recall |  0.99    |   0.99|  0.99   |  0.99      |
| Geomet |  0.99    |   0.95|  0.98   |  0.99      |

The combination model which combines over and undersampling performs strongly. The Random Oversampling performance is comparable, but has a lower recall score for the low risk class which renders Combination more powerful.
Jupyter notebook includes confusion matrices, balanced accuracy score and classification reports for each of the sampling techniques. The key metrics to monitor are Recall for negative class as this speaks directly to correctly predict (or miss) hgher risk loans which present significant costs to the company and hence are the most important metric to monitor.

### 2.2 Ensemble Learning Models

As the dataset is identical, a similar set of data processing steps is performed. For these sets of models scaling and over/undersampling methods are not applied given the high-variance / low bias features of the individual learners. For e.g. Balanced Random Forest randomly under-samples each boostrap sample to balance it. Hence addtional sample manipulation techniques are not required.

* [Balanced Random Forest](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html#)

Adaboost is another ensemble learning model that is used.
* [Adaboost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

Images below present sample confusion matrices and feature maps of Balanced Random Forest

<p align="center">
<img src="images/ConfMatrix_BRF.png" width="600" height="300"/>
</p>

<p align="center">
Figure 1.Confusion Matrix Sample
</p>

<p align="center">
<img src="images/Features_BRF.png" width="600" height="300"/>
</p>

<p align="center">
Figure 2. Feature strenght sample
</p>


### 2.3 Comparison of Balanced Random Forest and Adaboost

| Metric |BRF       |  ADA    |
| -------| ---------|---------|
| BAS    | 0.993    |  0.994  |
| Recall |  0.99    |   0.99  |
| Geomet |  0.99    |   0.99  | 

Adaboost is stronger performer though both models are robust.

Key features in order of importance are
Feature #1: Interest Rate
Feature #2: Total Debt
Feature #3: Loan Size
Feature #4: Borrower Income
Feature #5: Debt to Income

<p align="center">
<img src="images/Features_BRF.png" width="600" height="300"/>
</p>
