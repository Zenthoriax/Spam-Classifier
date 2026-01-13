import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="Spam Classifier",
    layout="wide"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
DATA_PATH = "spam.csv"
MODEL_PATH = "spam_classifier_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=["v1", "v2"])
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df

df = load_data()

# --------------------------------------------------
# Train Model (RUNS ONLY ON FIRST START)
# --------------------------------------------------
def train_and_save_model(dataframe):
    X = dataframe["message"]
    y = dataframe["label_num"]

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, metrics

# --------------------------------------------------
# Load or Train Model
# --------------------------------------------------
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metrics = {}
    st.sidebar.success("Loaded pre-trained model")
else:
    st.sidebar.warning("Training model for first run...")
    model, vectorizer, metrics = train_and_save_model(df)

# --------------------------------------------------
# Plot: Spam vs Ham Distribution
# --------------------------------------------------
def plot_distribution(dataframe):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        x="label",
        data=dataframe,
        palette={"ham": "skyblue", "spam": "salmon"},
        ax=ax
    )
    ax.set_title("Ham vs Spam Distribution")
    ax.set_xlabel("Message Type")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return fig

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def classify_message(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "🚨 SPAM" if pred == 1 else "✅ HAM (Not Spam)"

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📨 Spam / Ham Classifier")
st.markdown("---")

st.header("Dataset Overview")
st.pyplot(plot_distribution(df))

st.markdown("---")

st.header("Test a Message")

user_input = st.text_area(
    "Enter a message:",
    "You have won a FREE prize! Claim now!!!",
    height=100
)

if st.button("Classify"):
    if user_input.strip():
        with st.spinner("Classifying..."):
            result = classify_message(user_input)
        if "SPAM" in result:
            st.error(result)
        else:
            st.success(result)
    else:
        st.warning("Please enter a message.")

# --------------------------------------------------
# Sidebar Metrics
# --------------------------------------------------
st.sidebar.header("Model Info")
st.sidebar.markdown("**Algorithm:** Multinomial Naive Bayes")
st.sidebar.markdown("**Vectorizer:** CountVectorizer")

if metrics:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Evaluation Metrics")
    st.sidebar.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    st.sidebar.metric("Precision", f"{metrics['Precision']:.4f}")
    st.sidebar.metric("Recall", f"{metrics['Recall']:.4f}")
    st.sidebar.metric("F1 Score", f"{metrics['F1']:.4f}")
