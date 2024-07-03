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

### Dependencies
 * Python 3.x
 * scikit-learn
 * pandas
   
** License **
This model is licensed under the MIT License.
