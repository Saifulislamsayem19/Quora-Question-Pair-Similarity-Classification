# Quora Question Pair Similarity Classification

This repository contains the code and report for a project on question pair similarity classification. The goal is to classify pairs of questions as either "duplicate" (semantically similar) or "not duplicate" using machine learning models. The project utilizes the Quora Question Pairs dataset and explores several models, including Artificial Neural Networks (ANN), Logistic Regression, and Long Short-Term Memory (LSTM) networks.  A key focus of this project is hyperparameter tuning and optimization of the ANN model using RandomizedSearchCV and Keras Tuner.

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
* **Hyperparameter Tuning and Optimization:**  A critical component of this project is the optimization of the ANN model.  Two techniques were used:
    * **RandomizedSearchCV:**  Exploring a range of hyperparameter combinations to find promising settings.
    * **Keras Tuner:**  A more efficient and automated hyperparameter optimization technique to refine the ANN model's architecture and training parameters.
* **Prediction and Submission:** Generating predictions on the test set and creating a submission file.

## Model Tuning and Hyperparameter Optimization

To improve the performance of the Artificial Neural Network (ANN) model, hyperparameter tuning was performed using two techniques:

* **RandomizedSearchCV:** This technique was used to search for optimal hyperparameters within a defined search space, including the number of neurons, dropout rate, activation function, optimizer, batch size, and epochs.

* **Keras Tuner:** Keras Tuner was employed to further optimize the ANN model's hyperparameters. It provides a more efficient and flexible approach to hyperparameter tuning compared to RandomizedSearchCV.

The tuning process helped to identify the best combination of hyperparameters for the ANN model, leading to improved performance on the validation set.

## Performance Evaluation

Model performance was evaluated on a held-out validation set using the following metrics:

* Accuracy
* Precision
* Recall
* F1-score
* AUC-ROC
* Confusion Matrix

The following table summarizes the performance of the trained models on the validation set:

| Model                 | Accuracy | Precision | Recall  | F1-score |
|-----------------------|----------|-----------|---------|----------|
| LSTM                  | 75.86%   | 0.6365    | 0.8057  | 0.7112   |
| Logistic Regression   | 71.40%   | 0.5980    | 0.6860  | 0.6390   |
| ANN                   | 79.93%   | 0.7198    | 0.7469  | 0.7331   |
| Best ANN (Keras Tuner)| 81.11%   | 0.7531    | 0.7262  | 0.7394   |

The Best ANN model, optimized using Keras Tuner, achieved the highest accuracy of 81.11%.  Further details on model performance, including confusion matrices and AUC-ROC curves, can be found in the Jupyter Notebook.

## Requirements

The following libraries are required to run the code.  It is **highly recommended** to create a virtual environment to manage dependencies:

```bash
python3 -m venv .venv  # Create a virtual environment (Linux/macOS)
.venv\Scripts\activate  # Activate the virtual environment (Windows)
source .venv/bin/activate # Activate the virtual environment (Linux/macOS)
