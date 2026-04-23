"""
train_model.py - Load dataset, train models, evaluate and save the best one.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_dataframe
from feature_extraction import (
    build_tfidf_vectorizer,
    fit_transform,
    transform,
    save_vectorizer,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR   = BASE_DIR   # save model files in the same folder
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load and label Fake / True news CSVs, then combine them."""
    fake_path = os.path.join(DATASET_DIR, 'Fake.csv')
    true_path = os.path.join(DATASET_DIR, 'True.csv')

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Expect a 'text' column in both files; add label column
    fake_df['label'] = 0   # 0 = Fake
    true_df['label'] = 1   # 1 = Real

    df = pd.concat([fake_df, true_df], ignore_index=True)
    # Combine title + text for richer signal; fill missing titles gracefully
    df['combined'] = (df.get('title', '').fillna('') + ' ' + df['text'].fillna('')).str.strip()
    df = df[['combined', 'label']].dropna()
    df = df.rename(columns={'combined': 'text'})
    return df


def train():
    print("[INFO] Loading data …")
    df = load_data()

    print("[INFO] Preprocessing text …")
    df = preprocess_dataframe(df, text_col='text')

    X = df['cleaned_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Extracting TF-IDF features …")
    vectorizer = build_tfidf_vectorizer(max_features=10000)
    X_train_vec = fit_transform(vectorizer, X_train)
    X_test_vec  = transform(vectorizer, X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.3),
        "Naive Bayes":         MultinomialNB(alpha=0.5),
    }

    best_model, best_acc = None, 0.0

    for name, model in models.items():
        print(f"\n[INFO] Training {name} …")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc   = accuracy_score(y_test, preds)
        print(f"  Accuracy : {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

        if acc > best_acc:
            best_acc, best_model = acc, model

    # Save best model and vectoriser
    model_path = os.path.join(MODEL_DIR, 'fake_news_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n[INFO] Best model saved → {model_path}  (accuracy={best_acc:.4f})")

    save_vectorizer(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

if __name__ == "__main__":
    train()