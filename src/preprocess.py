import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



ling  = pd.read_csv('data/lingSpam.csv')[['Body', 'Label']]
enron = pd.read_csv('data/enronSpamSubset.csv')[['Body', 'Label']]


df = pd.concat([ling, enron], ignore_index=True)
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col]) 

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df['Body']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    token_pattern=r'[a-zA-Z]{3,}'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), 'models/data_split.pkl')

# Verify
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Nulls:", df.isnull().sum().sum())
print("Label distribution:\n", df['Label'].value_counts())
print("Training matrix:", X_train_tfidf.shape)

