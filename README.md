
# Breast Cancer Classification with Machine Learning Methods

## Introduction

This repository contains the code and results of a comprehensive study on the classification of breast cancer using machine learning methods, as part of a diploma thesis project. The data used in the study were extracted from fine-needle aspiration (FNA) samples, and a total of 13 machine learning algorithms were employed to classify the samples as benign or malignant.

## Folders & Files

1. `Breast Cancer Testing`: This folder contains all the files from the early stages of the project where various experiments and tests were conducted.

2. `Cross Validation (Wrong)`: This folder contains the first completed attempt of the study, but with a mistake in the way the parameters of the machine learning algorithms were optimized. The results of this attempt are not representative due to data leakage during cross-validation.

3. `Nested Cross Validation`: In this folder, nested cross-validation is used to optimize the parameters of the algorithms, resulting in unbiased results. A test is performed with data not seen by the algorithm and the results are considered to be representative.

## Data

The dataset used for this project was the [Wisconsin Diagnostic Breast Cancer (WDBC) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), which contains 569 samples and 30 features.

The data used in the study were extracted from fine-needle aspiration (FNA) samples, and consist of features calculated from the images of the samples. The data set used in this study was limited, and 10-fold cross validation was used for evaluating the performance of the algorithms. Additionally, a nested 5-fold cross validation was used for tuning the parameters of the algorithms.
## Feature Selection & Dimensionality Reduction

The thesis emphasizes on the reduction of the number of features while maintaining high accuracy in the results. Three different feature sets were used, including one with all features, one with a subset of seven features, and one with features extracted using the Principal Component Analysis method. The results of the study provide insight into the most effective and efficient machine learning methods for breast cancer classification, as well as the impact of reducing the number of features on the performance of the algorithms.
## Algorithms

The study employed 13 different machine learning algorithms, including Gaussian Naive Bayes, Linear & Quadratic Discriminant Analysis, Ridge Classifier, k-Nearest Neighbors, Support Vector Machines, Decision Tree, Random Forest, Gradient Tree Boosting, Adaboost & XGBoost, Stochastic Gradient Descent & Multi-Layer Perceptron. The performance of each algorithm was evaluated using the F1-score as the primary metric, which is a measure of the balance between precision and recall. Additional metrics such as accuracy, precision, and recall were also used.

## Requirements

The code in this repository was developed using Python 3.x and the following packages:

- Pandas
- Scikit-Learn
- NumPy
- SciPy
- Matplotlib
- Seaborn
## Usage

1. Clone the repository to your local machine.
2. Install the required packages using pip or conda.
3. Open the Jupyter Notebook file in the repository and run the code cells.
4. The results of the study and the performance of the algorithms can be found in the notebook.
