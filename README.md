Support Vector Machines (SVM) for Breast Cancer Classification
This repository contains a Jupyter notebook (notebooks/SVM_breast_cancer.ipynb) that demonstrates the application of Support Vector Machines (SVM) for classifying breast cancer as either malignant or benign based on features extracted from digitized images of breast mass FNA (Fine Needle Aspirate).

Project Overview
The goal of this project is to build and evaluate an SVM model to accurately predict breast cancer diagnosis. The notebook covers the following steps:

Dataset Loading and Preparation: Loading the dataset and performing initial data inspection.
Data Preprocessing: Handling missing values (if any), encoding categorical features, and splitting the data into training and testing sets.
Feature Scaling: Standardizing the features to ensure optimal performance of the SVM algorithm.
SVM Model Building: Implementing and training SVM models with different kernels (Linear and RBF).
Hyperparameter Tuning: Using Grid Search with cross-validation to find the best hyperparameters for the SVM model.
Model Evaluation: Assessing the performance of the trained models using metrics like confusion matrix, classification report, and accuracy score.
Visualization: Visualizing the decision boundary of the SVM model using PCA for dimensionality reduction.
Cross-Validation Evaluation: Performing cross-validation to get a more robust estimate of the model's performance.
Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which is a widely used dataset for binary classification. It contains features computed from digitized images of fine needle aspirates (FNA) of breast masses.

Features: 30 numerical features describing characteristics of the cell nuclei (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension - mean, standard error, and worst values).
Target: Diagnosis (M = malignant, B = benign).
How to Run the Notebook
You can run this notebook in a few ways:

Google Colab: The easiest way to run this notebook is by opening it directly in Google Colab. You can upload the SVM_breast_cancer.ipynb file to your Google Drive or open it directly from GitHub if you have connected your GitHub account to Colab.
Jupyter Notebook/Lab: You can download the notebook file and run it locally using Jupyter Notebook or JupyterLab.
Prerequisites
To run the notebook, you need to have Python and the following libraries installed:

pandas
numpy
matplotlib
scikit-learn
nbconvert (if you want to execute the notebook from the command line)
