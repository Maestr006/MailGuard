import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

X_train_tfidf, X_test_tfidf, y_train, y_test = joblib.load('models/data_split.pkl')

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("=== Naive Bayes ===")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/naive_bayes.pkl')
print("Model saved")