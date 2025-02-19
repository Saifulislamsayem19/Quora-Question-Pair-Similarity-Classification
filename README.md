# Quora Question Pair Similarity Classification

This repository contains code and a report for a project focused on identifying duplicate question pairs using machine learning. The project utilizes the Quora Question Pairs dataset and explores several models, including Artificial Neural Networks (ANN), Logistic Regression, and Long Short-Term Memory (LSTM) networks.

## Project Overview

The goal of this project is to build a model that can accurately determine whether two given questions from Quora are duplicates of each other.  This is a binary classification problem, where the target variable indicates whether a question pair is a duplicate (1) or not (0).

## Dataset

The Quora Question Pairs dataset is used for this project. 

## Code

The code for this project is implemented in a Jupyter Notebook (`quora_question_pairs.ipynb`). The notebook includes the following sections:

* **Exploratory Data Analysis (EDA):**  Visualizations and analysis of the dataset to understand its characteristics.  This includes distributions of question lengths, word clouds, common words, and Jaccard similarity.
* **Text Preprocessing:**  Cleaning and preparing the text data for modeling (lowercasing, special character removal, extra space removal, tokenization, stop word removal, lemmatization).
* **Feature Extraction:**  Converting text data into numerical features using TF-IDF for Logistic Regression and ANN, and tokenization/padding for LSTM.
* **Model Building and Training:** Training different machine learning models (ANN, Logistic Regression, LSTM).
* **Model Evaluation:** Evaluating model performance using metrics like accuracy, precision, recall, F1-score, AUC-ROC, and confusion matrices.  *See the Performance Evaluation Report below for detailed results.*
* **Model Tuning:** Hyperparameter optimization for the ANN model using Keras Tuner.
* **Prediction and Submission:** Generating predictions on the test set and creating a submission file.

## Performance Evaluation Report

The performance evaluation report is included in the notebook itself (as markdown cells) and summarizes the results of the model evaluation.  It includes a table of metrics for each model, confusion matrices, and a brief analysis of the results.  Key findings are summarized below:

* **ANN (Tuned):** Achieved the highest accuracy (e.g., 80.87%) and AUC-ROC (e.g., 0.8777) after hyperparameter tuning. It showed a good balance between precision (e.g., 80.05%) and recall (e.g., 65.21%).
* **Logistic Regression:** Served as a baseline model with an accuracy of (e.g., 73.80%) and AUC-ROC of (e.g., 0.7798).
* **LSTM:**  Provided reasonable performance, although slightly lower than the tuned ANN, with an accuracy of (e.g., 76.20%) and AUC-ROC of (e.g., 0.8355).
