"""
preprocessing.py - Text cleaning and preprocessing utilities
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Full pipeline: lowercase → remove noise → tokenize → remove stopwords → stem.
    Returns a single cleaned string ready for TF-IDF vectorisation.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 1b. Remove wire-service datelines / bylines
    # e.g. "WASHINGTON (Reuters) - ", "NEW YORK (AP) - ", "LONDON (AFP) - "
    text = re.sub(r'^[a-z/ ]+\([a-z/ .]+\)\s*[-–]\s*', '', text)
    # Also strip bare city datelines: "WASHINGTON - "
    text = re.sub(r'^[A-Z][A-Z ]+\s*[-–]\s*', '', text)

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. Tokenise
    tokens = word_tokenize(text)

    # 6. Remove stopwords and very short tokens, then stem
    # Also remove known journalist boilerplate words that are stylistic, not semantic
    _style_words = {'reuter', 'report', 'report', 'said', 'say', 'told', 'tell',
                    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                    'saturday', 'sunday', 'edt', 'gmt', 'est', 'jan', 'feb',
                    'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}
    cleaned = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 2 and token not in _style_words
    ]

    return " ".join(cleaned)


def preprocess_dataframe(df, text_col: str = 'text') -> object:
    """
    Apply clean_text to every row of *text_col* in a DataFrame.
    Adds a new column 'cleaned_text' and drops rows where it is empty.
    """
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)
    return df