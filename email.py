import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('spamem.csv')

X = data['Body']
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = CountVectorizer()

X_train_counts = vectorizer.fit_transform(X_train)

X_test_counts = vectorizer.transform(X_test)


rfc = RandomForestClassifier(n_estimators=100,criterion='gini')

rfc.fit(X_train_counts, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = rfc.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report}")
