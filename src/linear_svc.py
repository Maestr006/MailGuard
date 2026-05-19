import joblib
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

X_train_tfidf, X_test_tfidf, y_train, y_test = joblib.load('models/data_split.pkl')

model = CalibratedClassifierCV(LinearSVC(class_weight='balanced'))
model.fit(X_train_tfidf , y_train)

y_pred = model.predict(X_test_tfidf)
print("=== linear SVC ===")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/linear_svc.pkl')
print("Model saved ")