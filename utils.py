import joblib
import os

MODEL_DIR = 'model'

def load_model():
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    vectorizer_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict(question1, question2, model, vectorizer):
    combined = question1 + " " + question2
    combined_vec = vectorizer.transform([combined])
    prediction = model.predict(combined_vec)
    return prediction[0]
