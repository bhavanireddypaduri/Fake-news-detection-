"""
predict.py - Load trained artefacts and predict a single news article.
"""

import os
import joblib

from preprocessing import clean_text
from feature_extraction import transform, load_vectorizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# If confidence is below this threshold, return UNCERTAIN instead of a wrong label
CONFIDENCE_THRESHOLD = 0.50


def load_model(path: str = None):
    path = path or os.path.join(MODEL_DIR, 'fake_news_model.pkl')
    return joblib.load(path)


def predict(text: str, model=None, vectorizer=None) -> dict:
    """
    Predict whether *text* is Fake or Real news.

    Returns
    -------
    dict with keys:
        label      : "FAKE" | "REAL"
        confidence : float (0–1), probability of the predicted class
    """
    if model is None:
        model = load_model()
    if vectorizer is None:
        vectorizer = load_vectorizer(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

    cleaned  = clean_text(text)
    features = transform(vectorizer, [cleaned])

    prediction    = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence    = float(max(probabilities))

    if confidence < CONFIDENCE_THRESHOLD:
        label = "UNCERTAIN"
    else:
        label = "REAL" if prediction == 1 else "FAKE"

    return {"label": label, "confidence": round(confidence * 100, 2)}


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = input("Paste news text and press Enter:\n> ")
    result = predict(sample)
    print(f"\nPrediction : {result['label']}")
    print(f"Confidence : {result['confidence']}%")