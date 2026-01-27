import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Renamimg
data.columns = ['Label', 'Message', 'Extra1', 'Extra2', 'Extra3']

data = data[['Label', 'Message']]

# Displaying the first few rows to check the changes
print("\n" + "="*16 + " SPAM SMS DETECTION " + "="*16)
print("\nFirst 5 entries in the dataset:")
print("-"*60)
print(data.head())
print("-"*60 + "\n")

# Preprocessing
data['Message'] = data['Message'].str.lower().str.replace('[^a-zA-Z\s]', '', regex=True)

messages_train, messages_test, labels_train, labels_test = train_test_split(data['Message'], data['Label'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
messages_train_tfidf = tfidf_vectorizer.fit_transform(messages_train)
messages_test_tfidf = tfidf_vectorizer.transform(messages_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(messages_train_tfidf, labels_train)

labels_pred = naive_bayes.predict(messages_test_tfidf)
model_accuracy = accuracy_score(labels_test, labels_pred)

print(f'Model Accuracy: {model_accuracy*100:.2f}%')
print("="*50 + "\n")
