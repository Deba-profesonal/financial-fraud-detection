# Financial Fraud Detection System

A Data Science and Machine Learning project for detecting fraudulent credit card transactions using Python and classification algorithms.

## 1. Project Overview

Financial fraud detection is a binary classification problem that aims to identify suspicious transactions and distinguish them from legitimate transactions.

This project applies data analysis, exploratory data analysis (EDA), data preprocessing, and machine learning to classify credit card transactions.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Google Colab
- Git and GitHub

## 3. Dataset Description

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions and 31 columns.

| Property | Value |
|---|---:|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Input features | 30 |
| Target variable | `Class` |

The `Class` column represents the transaction label:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

Fraudulent transactions account for approximately 0.173% of the dataset, making this a highly imbalanced classification problem.

## 4. Project Development Process

### Step 1: Dataset Loading and Inspection
- Loaded the dataset using Pandas.
- Inspected dataset dimensions, columns, and data types.
- Checked missing values and duplicate records.
- Analyzed the distribution of legitimate and fraudulent transactions.

### Step 2: Exploratory Data Analysis
- Visualized class distribution.
- Examined transaction amount statistics.
- Compared transaction amount distributions across classes.
- Investigated duplicate records.
- Analyzed feature correlations with the target variable.

### Step 3: Data Preprocessing
- Separated input features and target labels.
- Applied a stratified 80:20 train-test split.
- Standardized input features for Logistic Regression using training data.
- Retained the original feature scale for Random Forest.

### Step 4: Logistic Regression
- Trained a Logistic Regression baseline.
- Used balanced class weights to address class imbalance.
- Evaluated the model using classification metrics and ROC-AUC and PR-AUC.

### Step 5: Random Forest
- Trained a Random Forest classifier with balanced class weights.
- Evaluated its predictions on the same test set.
- Compared performance against Logistic Regression.

## 5. Model Evaluation Results

The following results were obtained using the initial stratified random train-test split.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 97.55% | 99.95% |
| Fraud Precision | 6.10% | 96.10% |
| Fraud Recall | 91.84% | 75.51% |
| Fraud F1-score | 0.1144 | 0.8457 |
| PR-AUC | 0.7190 | 0.8589 |
| ROC-AUC | 0.9721 | 0.9572 |

### Results Analysis

- Random Forest achieved higher fraud precision, fraud F1-score, and PR-AUC at the evaluated settings.
- Logistic Regression achieved higher fraud recall, but its low precision indicates a large number of false-positive alerts.
- Random Forest provided a better balance between precision and recall at the default classification threshold used in this experiment.
- Accuracy was interpreted alongside class-specific metrics because of the dataset's severe class imbalance.

## 6. Conclusion

This project demonstrates how machine learning can be applied to highly imbalanced financial transaction data.

The experiments show that model selection involves a trade-off between identifying fraudulent transactions and avoiding false alarms. Random Forest achieved stronger fraud precision and PR-AUC in the reported evaluation, while Logistic Regression detected a larger proportion of fraudulent transactions at its evaluated threshold.

The results highlight the importance of using appropriate evaluation metrics rather than relying solely on accuracy.

---
