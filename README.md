# Bank-Churn-Prediction-ML-project-


---

# Bank Churn Prediction Project

This repository contains a machine learning project aimed at predicting customer churn in a bank. The project utilizes various techniques to handle class imbalance and trains a Random Forest Classifier to predict whether a customer will churn based on various features.

## Table of Contents

1. [Installation](#installation)
2. [Data](#data)
3. [Preprocessing](#preprocessing)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)


## Installation

To get started with this project, clone the repository and install the required dependencies. You can use the following commands:

```bash
git clone https://github.com/YOUR_USERNAME/bank-churn-prediction.git
cd bank-churn-prediction
pip install -r requirements.txt
```

## Data

The dataset used in this project is the Bank Churn Modelling dataset, which can be found [here](https://github.com/YBIFoundation/Dataset/raw/main/Bank%20Churn%20Modelling.csv).

### Features:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `Num Of Products`
- `Has Credit Card`
- `Is Active Member`
- `Estimated Salary`
- `Churn` (Target variable)

## Preprocessing

1. **Handling Missing Values:**
   The dataset is checked for missing values, and none were found.

2. **Data Encoding:**
   Categorical variables (`Geography` and `Gender`) are converted to dummy variables.

3. **Handling Class Imbalance:**
   - **Undersampling:** `RandomUnderSampler` is used to balance the classes by undersampling the majority class.
   - **Oversampling:** `RandomOverSampler` is used to balance the classes by oversampling the minority class.

## Model Training

### Original Data

1. **Train-Test Split:**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2529)
   ```

2. **Model Creation and Training:**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rfc = RandomForestClassifier()
   rfc.fit(X_train, y_train)
   ```

### Oversampled Data

1. **Train-Test Split:**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, random_state=2529)
   ```

2. **Model Creation and Training:**
   ```python
   rfc.fit(X_train, y_train)
   ```

## Evaluation

The model is evaluated using the classification report, which includes precision, recall, f1-score, and support.

### Evaluation on Original Data

```python
from sklearn.metrics import classification_report
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Evaluation on Oversampled Data

```python
y_pred = rfc.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Results

- **Original Data:** The model's performance is evaluated on the original imbalanced dataset.
- **Oversampled Data:** The model's performance is evaluated on the oversampled balanced dataset.

