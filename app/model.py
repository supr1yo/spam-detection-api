import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
model = joblib.load(model_path)

def predict_spam(message: str) -> str:
    return model.predict([message])[0]

