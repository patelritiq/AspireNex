import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

credit_data = pd.read_csv('creditcard.csv')

# Preparation
features = credit_data.drop(['Class'], axis=1)
target = credit_data['Class']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features = pd.DataFrame(features_scaled, columns=features.columns)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
log_reg_predictions = log_reg_model.predict(X_test)

print("\n" + "="*12 + " CREDIT CARD FRAUD DETECTION " + "="*12)
print("\n" + "-"*13 + " Logistic Regression Results " + "-"*13)
print(f"\nAccuracy: {accuracy_score(y_test, log_reg_predictions):.4f}")
print("-"*55)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, log_reg_predictions))
print("-"*55)
print("\nClassification Report:\n", classification_report(y_test, log_reg_predictions))

#Decision Tree model
dec_tree_model = DecisionTreeClassifier()
dec_tree_model.fit(X_train, y_train)
dec_tree_predictions = dec_tree_model.predict(X_test)

print("\n" + "-"*13 + " Decision Tree Results " + "-"*13)
print(f"\nAccuracy: {accuracy_score(y_test, dec_tree_predictions):.4f}")
print("-"*55)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dec_tree_predictions))
print("-"*55)
print("\nClassification Report:\n", classification_report(y_test, dec_tree_predictions))
print("="*55 + "\n")


