# Model Training: Fit the Logistic Regression Model on Clean Data

This project demonstrates how to train a Logistic Regression model on a clean dataset using Python and scikit-learn. It also highlights common security and robustness issues that can appear in machine learning workflows when data validation, model integrity, and safe persistence are not properly handled.

## Project Objective

The goal is to:
- load a dataset,
- split it into features and target labels,
- train a Logistic Regression classifier,
- evaluate the model accuracy,
- save and load the trained model safely.

## Requirements

Install the required Python packages:

```bash
pip install pandas scikit-learn
```

## Dataset

This example assumes a CSV file named `user_data.csv` containing a feature matrix and a target column.

Example structure:

```csv
age,income,subscription
25,40000,0
30,52000,1
...
```

## Example Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
# Best practice: validate file format, column names, and missing values before use
data = pd.read_csv('user_data.csv')

# Split into features and target
# Best practice: confirm that the target column is present and correctly typed
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model to disk
# Best practice: use secure serialization and integrity checks
filename = 'finalized_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Load the model from disk for later use
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')
```

## What the Model Does

Logistic Regression is a supervised learning algorithm used for binary classification problems. It estimates the probability that a sample belongs to a particular class and uses a decision threshold to classify it.

## Model Evaluation

The model is evaluated by measuring its accuracy on the testing set:

```python
result = loaded_model.score(X_test, y_test)
print(f'Model Accuracy: {result:.2f}')
```

## Security and ML Best Practices

This project also demonstrates a few common pitfalls in ML pipelines:

- No input validation on uploaded or generated data
- Lack of sanitization for CSV content
- Unencrypted model persistence
- No integrity checks before loading a model
- Fixed random states and poor reproducibility controls

A production-ready implementation should include:
- schema validation,
- missing-value checks,
- secure model storage,
- digital signing or checksum verification,
- reproducible training pipelines.

## Summary

This project is a simple example of training a Logistic Regression classifier on clean data and evaluating its performance. It also serves as a reminder that ML systems need proper security practices in addition to accurate model training.
