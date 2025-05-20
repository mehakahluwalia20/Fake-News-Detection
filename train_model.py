import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Label data
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine and clean
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df[['text', 'label']].dropna()

# Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
