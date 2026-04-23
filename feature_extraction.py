"""
feature_extraction.py - TF-IDF vectorisation helpers
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Return a configured (unfitted) TF-IDF vectoriser."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,    # apply log normalization
    )


def fit_transform(vectorizer: TfidfVectorizer, texts):
    """Fit the vectoriser on *texts* and return the document-term matrix."""
    return vectorizer.fit_transform(texts)


def transform(vectorizer: TfidfVectorizer, texts):
    """Transform *texts* using an already-fitted vectoriser."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = "model/tfidf_vectorizer.pkl"):
    joblib.dump(vectorizer, path)
    print(f"[INFO] Vectoriser saved → {path}")


def load_vectorizer(path: str = "model/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    return joblib.load(path)