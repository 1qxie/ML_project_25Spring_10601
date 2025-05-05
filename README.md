# Machine Learning Project: Predicting Project Funding Success

## Overview

This project applies machine learning techniques to predict whether education-related crowdfunding projects (from DonorsChoose.org) will be fully funded. Using real-world data from the KDD Cup 2014, we implemented a full ML pipeline including data cleaning, feature engineering, modeling, and evaluation. The goal is to build interpretable and accurate models to support funding decision-making and educational equity.

---

## Project Structure

### Notebooks

* **`description_datasets_ML_Project.ipynb`**
  Provides an exploratory overview of all datasets used in the project. Includes summary statistics and initial observations.

* **`data_cleaning.ipynb`**
  Cleans raw input data, handles missing values, renames variables, and merges files from the [KDD Cup 2014 dataset](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data) to prepare the feature matrix.

### Python Scripts

* **`feature_slection.py`**
  Selects relevant features using variance filtering and Random Forest feature importance.

* **`data_splitting.py`**
  Splits the dataset into stratified training and test sets, preserving class balance.

* **`baseline_model.py`**
  Trains and evaluates baseline models (Logistic Regression, Random Forest) to provide performance benchmarks.

* **`advanced_model.py`**
  Implements advanced models using Random Forest and XGBoost pipelines with hyperparameter tuning. Evaluates models using ROC AUC, F1, and PR curves.

---

## Workflow Summary

1. **Data Exploration**
   Understand and visualize key patterns in the dataset.

2. **Data Cleaning & Preprocessing**
   Handle missing values, data types, and merge multiple sources.

3. **Feature Engineering**
   Create features such as price per student, price variability, and seasonal indicators.

4. **Feature Selection**
   Identify the most predictive variables using tree-based importance scores.

5. **Data Splitting**
   Stratified 80/20 split for model training and evaluation.

6. **Modeling**

   * Build baseline models using Logistic Regression and Random Forest
   * Develop advanced models using XGBoost and tuned Random Forest pipelines

7. **Evaluation**
   Use classification metrics (precision, recall, F1-score) and AUC curves to compare model performance and guide final selection.

---

## Requirements

This project uses the following Python libraries and tools:

* `pandas`, `numpy`, `os`
* `matplotlib.pyplot`, `seaborn`
* `sklearn.model_selection`: `train_test_split`, `StratifiedKFold`, `GridSearchCV`
* `sklearn.preprocessing`: `LabelEncoder`, `StandardScaler`, `OneHotEncoder`
* `sklearn.compose`: `ColumnTransformer`
* `sklearn.pipeline`: `Pipeline`
* `sklearn.impute`: `SimpleImputer`
* `sklearn.feature_selection`: `VarianceThreshold`
* `sklearn.linear_model`: `LogisticRegression`
* `sklearn.ensemble`: `RandomForestClassifier`
* `xgboost`: `XGBClassifier`
* `sklearn.metrics`: `classification_report`, `confusion_matrix`, `roc_curve`, `precision_recall_curve`, `auc`

