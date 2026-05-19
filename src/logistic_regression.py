import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train_tfidf, X_test_tfidf, y_train, y_test = joblib.load('models/data_split.pkl')

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_tfidf , y_train)

y_pred = model.predict(X_test_tfidf)
print("=== logistic regression ===")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_regression.pkl')
print("Model saved ")
