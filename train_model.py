import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score
import joblib

# Load dataset
df = pd.read_csv('./data/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename columns

# Convert categorical labels to numeric values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Assign features and labels
message = df['message']
label = df['label']

# Split data
x_train, x_test, y_train, y_test = train_test_split(message, label, test_size=0.2, random_state=42)

# Pipeline: Vectorizer + Classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train model
model.fit(x_train, y_train)

# Save model
joblib.dump(model, './models/model.pkl')

# Evaluate model
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
