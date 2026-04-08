import os
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Constants
DATA_PATH = "spam.csv"
MODEL_PATH = "spam_classifier_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def download_nltk_resources():
    """Download required NLTK data."""
    resources = [
        'punkt', 
        'punkt_tab', 
        'stopwords', 
        'wordnet', 
        'omw-1.4', 
        'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker',
        'words'
    ]
    for res in resources:
        nltk.download(res, quiet=True)

class SpamClassifierBackend:
    def __init__(self):
        download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.vectorizer = None
        self.load_or_train()

    def preprocess(self, text):
        """Full NLP pipeline for a single message."""
        # 1. Tokenization
        tokens = word_tokenize(text.lower())
        
        # 2. Stopword Removal & Lemmatization
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        # 3. POS Tagging (for visualization)
        pos_tags = nltk.pos_tag(tokens)
        
        return {
            "original": text,
            "tokens": tokens,
            "cleaned_tokens": cleaned_tokens,
            "pos_tags": pos_tags,
            "processed_text": " ".join(cleaned_tokens)
        }

    def load_or_train(self):
        """Loads existing model or trains a new one if missing."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            self.train_and_save()

    def train_and_save(self):
        """Trains the model using the dataset."""
        df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=["v1", "v2"])
        df = df.rename(columns={"v1": "label", "v2": "message"})
        df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
        
        X = df["message"]
        y = df["label_num"]

        self.vectorizer = CountVectorizer()
        X_vec = self.vectorizer.fit_transform(X)

        self.model = MultinomialNB()
        self.model.fit(X_vec, y)

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)

    def predict(self, text):
        """Predicts spam/ham and returns detailed info."""
        nlp_data = self.preprocess(text)
        vec = self.vectorizer.transform([nlp_data["processed_text"]])
        
        # Get probability
        prob = self.model.predict_proba(vec)[0]
        prediction = self.model.predict(vec)[0]
        
        return {
            "prediction": "SPAM" if prediction == 1 else "HAM",
            "confidence": max(prob) * 100,
            "nlp_steps": nlp_data
        }

    def get_stats(self):
        """Returns basic stats for the UI."""
        df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=["v1", "v2"])
        df = df.rename(columns={"v1": "label", "v2": "message"})
        return df['label'].value_counts().to_dict()
