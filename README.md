# Customer Churn Prediction Model
### Overview
This is a machine learning model that predicts the likelihood of a credit customer churning based on their historical data. The model uses a Random Forest Classifier to classify customers as either churners or non-churners.

### Dataset
The model is trained on a dataset called Churn.csv, which contains customer data with various features and a target variable Churn indicating whether the customer has churned or not.

### Model Performance
The model's performance is evaluated using the following metrics:
 > Accuracy
 > Precision
 > Recall
 > F1-score

### How to Use:
 - Download the Churn.csv dataset and place it in the same directory as the Python script.
 - Run the Python script to train the model and generate predictions.
 - The model's performance metrics will be printed to the console.



# SMS Spam Detection Model
### Overview
This is a simple SMS Spam Detection model built using Python and scikit-learn. The model uses a Multinomial Naive Bayes classifier to classify SMS messages as either spam or not spam.

### Dataset
The model is trained on a dataset of labeled SMS messages, where each message is accompanied by a label indicating whether it is spam or not. The dataset is stored in a CSV file named spam.csv.

## Features
The model uses the text of the SMS message as the feature for classification.
The text is preprocessed by converting it to lowercase and removing non-alphabetic characters.

## Model
The model uses a Multinomial Naive Bayes classifier to classify the SMS messages.
The classifier is trained on a training set of messages and evaluated on a test set.

### Usage
To use the model, simply run the Python script and the model will be trained and evaluated on the provided dataset.



# Credit Card Fraud Detection Model
### Overview
This is a Credit Card Fraud Detection model built using Python and scikit-learn. The model uses two different classification algorithms, Logistic Regression and Decision Tree, to detect fraudulent transactions.

### Dataset
The model is trained on a dataset of credit card transactions, where each transaction is labeled as either fraudulent (1) or not fraudulent (0). The dataset is stored in a CSV file named creditcard.csv.

## Models
The model uses two classification algorithms:
 - Logistic Regression
 - Decision Tree

### Usage
To use the model, simply run the Python script and the model will be trained and evaluated on the provided dataset.

### Dependencies
 * Python 3.x
 * scikit-learn
 * pandas
