# Machine Learning: Customer Purchase Prediction

## About
This repository contains a complete **predictive analytics pipeline** built in Python. It trains a machine learning algorithm to predict whether a customer will make a purchase based on their demographic and behavioral data. The script completely automates the workflow—from **data preprocessing and hyperparameter tuning** to **model evaluation and final report generation**.

## Features
* **Random Forest Classifier:** A robust ensemble learning algorithm chosen for its high accuracy and resistance to overfitting.
* **Automated Hyperparameter Tuning:** Utilizes `GridSearchCV` to automatically test and select the most optimal model parameters.
* **Comprehensive Evaluation:** Calculates exact metrics for **Accuracy, Precision, Recall, and F1-Score**.
* **Diagnostic Visualizations:** Automatically generates and saves a **Confusion Matrix** and an **ROC Curve**.
* **Automated Word Reporting:** Uses `python-docx` to instantly compile all metrics, justifications, and visual plots into a ready-to-submit `.docx` file.

## Prerequisites
Ensure you have Python installed on your system. You will need to install the following core data science libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn python-docx
