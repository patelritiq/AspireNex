import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

churn_data = pd.read_csv('Churn.csv')

churn_data_cleaned = churn_data.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
for column in churn_data_cleaned.columns:
    if churn_data_cleaned[column].dtype == object:
        churn_data_cleaned[column] = label_encoder.fit_transform(churn_data_cleaned[column])

features = churn_data_cleaned.drop('Churn', axis=1)
target = churn_data_cleaned['Churn']

rf_model = RandomForestClassifier()

# Training
rf_model.fit(features, target)

predictions = rf_model.predict(features)

# Calculation
accuracy = accuracy_score(target, predictions)
precision = precision_score(target, predictions)
recall = recall_score(target, predictions)
f1 = f1_score(target, predictions)

# Print results
print("\n" + "="*12 + " CUSTOMER CHURN PREDICTION " + "="*12)
print("\n" + "-"*8 + " Model Evaluation Results " + "-"*8)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}\n")
print("-"*50 + "\n")
