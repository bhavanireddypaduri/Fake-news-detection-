"""
main.py - Top-level entry point.

Usage
-----
# Train the model:
    python main.py train

# Predict a single article interactively:
    python main.py predict

# Start the web app:
    python main.py web
"""

import sys
import os

# Make src/ importable from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_train():
    from train_model import train
    train()


def run_predict():
    from predict import predict
    text = input("Paste your news article and press Enter:\n> ").strip()
    if not text:
        print("[ERROR] Empty input.")
        return
    result = predict(text)
    print(f"\n{'='*40}")
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"{'='*40}\n")


def run_web():
    from app import app
    print("[INFO] Starting Fake News Detector web app on http://127.0.0.1:5001")
    app.run(debug=False, port=5001)


COMMANDS = {
    'train':   run_train,
    'predict': run_predict,
    'web':     run_web,
}

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'web'
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Choose from: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()