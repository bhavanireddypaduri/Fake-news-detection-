"""
index.py - Vercel entry point for Fake News Detection
"""

import os
import sys
import json
import urllib.request
import urllib.error

# Add parent directory to sys.path to allow imports from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, jsonify
from predict import predict, load_model
from feature_extraction import load_vectorizer

# Point Flask to the root directory for templates and static files
app = Flask(__name__, template_folder='../', static_folder='../')

# BASE_DIR is the root of the project
BASE_DIR    = parent_dir
_model      = None
_vectorizer = None

# Load .env file manually without external libraries (for local debugging)
env_file = os.path.join(BASE_DIR, '.env')
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def verify_with_groq(text):
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a fake news detection assistant. Reply with only one word: 'REAL', 'FAKE', or 'UNCERTAIN' based on the factual accuracy and tone of the text."},
            {"role": "user", "content": text[:1500]}
        ],
        "temperature": 0.1
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            res = json.loads(response.read().decode('utf-8'))
            reply = res['choices'][0]['message']['content'].strip().upper()
            
            # Stricter parsing
            if reply in ["REAL", "REAL.", '"REAL"', "'REAL'"]: return "REAL"
            if reply in ["FAKE", "FAKE.", '"FAKE"', "'FAKE'"]: return "FAKE"
            
            # If the LLM is wordy
            if "FAKE" in reply and "REAL" not in reply: return "FAKE"
            if "REAL" in reply and "FAKE" not in reply: return "REAL"
            
            return "UNCERTAIN"
    except Exception as e:
        print(f"Groq API error: {e}")
        return None

def get_artifacts():
    global _model, _vectorizer
    if _model is None:
        _model      = load_model(os.path.join(BASE_DIR, 'fake_news_model.pkl'))
        _vectorizer = load_vectorizer(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
    return _model, _vectorizer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    model, vectorizer = get_artifacts()
    result = predict(text, model=model, vectorizer=vectorizer)
    
    # LLM Second Opinion for UNCERTAIN results OR low-confidence ML results
    result['llm_verified'] = False
    low_confidence = result.get('confidence', 100) < 70
    if (result["label"] == "UNCERTAIN" or low_confidence) and GROQ_API_KEY:
        llm_label = verify_with_groq(text)
        if llm_label in ["REAL", "FAKE"]:
            result["label"] = llm_label
            result["llm_verified"] = True

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, port=5001)