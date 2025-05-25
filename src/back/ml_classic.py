

import joblib
import os

# Ruta del directorio de modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_ml")

# Carga del vectorizador y los modelos
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
naive_bayes_model = joblib.load(os.path.join(MODEL_DIR, "modelo_naivebayes.pkl"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "modelo_svm.pkl"))

def predict_with_logistic_regression(text: str) -> dict:
    X = vectorizer.transform([text])
    prediction = logistic_model.predict(X)[0]
    return {
        "etiqueta_aita": prediction,
        "razonamiento": "Predicción generada por regresión logística basada en texto vectorizado.",
        "text": text
    }

def predict_with_naive_bayes(text: str) -> dict:
    X = vectorizer.transform([text])
    prediction = naive_bayes_model.predict(X)[0]
    return {
        "etiqueta_aita": prediction,
        "razonamiento": "Predicción generada por Naive Bayes basada en texto vectorizado.",
        "text": text
    }

def predict_with_svm(text: str) -> dict:
    # TODO
    X = vectorizer.transform([text])
    prediction = svm_model.predict(X)[0]
    return {
        "etiqueta_aita": prediction,
        "razonamiento": "Predicción generada por SVM basada en texto vectorizado.",
        "text": text
    }