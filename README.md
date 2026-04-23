# 🔍 Fake News Detection

A Machine Learning + NLP system that classifies news articles as **Fake** or **Real**.

---

## 📁 Project Structure

```
Fake-News-Detection/
├── dataset/
│   ├── fake_news.csv        # Fake news articles (label = 0)
│   └── true_news.csv        # Real news articles (label = 1)
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py     # Text cleaning pipeline
│   ├── feature_extraction.py# TF-IDF vectorisation
│   ├── train_model.py       # Model training & evaluation
│   └── predict.py           # Single-article prediction
├── model/
│   ├── fake_news_model.pkl  # Saved best model
│   └── tfidf_vectorizer.pkl # Saved TF-IDF vectoriser
├── web_app/
│   ├── app.py               # Flask application
│   ├── templates/index.html # UI
│   └── static/style.css     # Styles
├── requirements.txt
├── main.py                  # Unified entry point
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / download the project
cd Fake-News-Detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

Download the **Fake and Real News Dataset** from Kaggle:  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place the files as:
- `dataset/fake_news.csv`  (Fake.csv renamed)
- `dataset/true_news.csv`  (True.csv renamed)

Both files must contain a **`text`** column with the article body.

---

## 🚀 Usage

### 1 · Train the model
```bash
python main.py train
```
Trains Logistic Regression and Naive Bayes, saves the best model to `model/`.

### 2 · Predict from the command line
```bash
python main.py predict
```
Paste any news text and get an instant Fake / Real verdict.

### 3 · Run the web app
```bash
python main.py web
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🔄 System Workflow

```
User Input
    ↓
Text Preprocessing  (lowercase, remove URLs/HTML/punctuation, stem)
    ↓
TF-IDF Feature Extraction  (5 000 features, unigrams + bigrams)
    ↓
ML Model  (Logistic Regression or Naive Bayes)
    ↓
Prediction  →  FAKE | REAL  +  Confidence %
```

---

## 🛠 Tech Stack

| Layer           | Tools                                  |
|-----------------|----------------------------------------|
| Language        | Python 3.9+                            |
| Data            | pandas, NumPy                          |
| NLP             | NLTK (tokeniser, stopwords, stemmer)   |
| Features        | scikit-learn TfidfVectorizer           |
| Models          | Logistic Regression, Naive Bayes       |
| Serialisation   | joblib                                 |
| Web             | Flask, HTML, CSS                       |