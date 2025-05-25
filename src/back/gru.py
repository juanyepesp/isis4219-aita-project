# predict_gru.py

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = 'deep_learning/best_gru_model.h5'
VECTORIZER_PATH = 'deep_learning/vectorizer.pkl'
ENCODER_PATH = 'deep_learning/label_encoder.pkl'


SEQUENCE_LENGTH = 200

model = load_model(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def create_sequence_from_text(text, vectorizer, seq_len):
    vec = vectorizer.transform([text])
    dense = vec.toarray().astype(np.float32)
    
    if dense.shape[1] < seq_len:
        padded = np.zeros((1, seq_len), dtype=np.float32)
        padded[:, :dense.shape[1]] = dense
        return padded.reshape(1, seq_len, 1)
    else:
        feature_importance = np.var(dense, axis=0)
        top_features = np.argsort(feature_importance)[-seq_len:]
        reduced = dense[:, top_features]
        return reduced.reshape(1, seq_len, 1)

def get_prediction(text_input):
    sequence = create_sequence_from_text(text_input, vectorizer, SEQUENCE_LENGTH)
    prediction = model.predict(sequence, verbose=0)
    label_index = np.argmax(prediction)
    return label_encoder.inverse_transform([label_index])[0]
