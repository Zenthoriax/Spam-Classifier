import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Set ML Engineer Master title for the app
st.set_page_config(page_title="Spam Classifier", layout="wide")

## --- 1. Load Data ---
@st.cache_data
def load_data():
    """Loads, preprocesses, and returns the dataset."""
    try:
        data = pd.read_csv('spam.csv', encoding='latin-1', usecols=['v1', 'v2'])
        data = data.rename(columns={'v1': 'label', 'v2': 'message'})
        data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})
        return data
    except FileNotFoundError:
        st.error("Error: 'spam.csv' not found. Please ensure the file is in the same directory.")
        return pd.DataFrame() 

df = load_data()

## --- 2. Plot Generation (Streamlit Native) ---
def create_distribution_plot(dataframe):
    """Generates the Ham vs. Spam distribution plot using Matplotlib/Seaborn."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Use seaborn to create the bar plot
    sns.countplot(
        x='label', 
        data=dataframe, 
        palette={'ham': 'skyblue', 'spam': 'salmon'}, 
        order=dataframe['label'].value_counts().index, 
        ax=ax
    )
    ax.set_title('Ham vs. Spam Message Distribution (Imbalanced Data)', fontsize=14)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Use st.cache_data to cache the plot figure object
    return fig

## --- 3. Model Training and Saving ---
def train_and_save_model(dataframe):
    """Trains the Multinomial Naive Bayes model, calculates metrics, and saves components."""
    if dataframe.empty:
        return None, None, {}

    X = dataframe['message']
    y = dataframe['label_num']

    # FEATURE EXTRACTION
    vectorizer = CountVectorizer()
    X_features = vectorizer.fit_transform(X)

    # DATA SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # MODEL TRAINING
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # METRICS CALCULATION
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Algorithm': 'Multinomial Naive Bayes',
        'Vectorizer': 'CountVectorizer'
    }

    # PERSISTENCE
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return model, vectorizer, metrics

# --- Load or Train Model ---
model_path = 'spam_classifier_model.pkl'
vectorizer_path = 'vectorizer.pkl'

# Initial check for saved files
is_model_saved = os.path.exists(model_path) and os.path.exists(vectorizer_path)

if is_model_saved:
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        # Recalculate metrics for display (fast on a small model)
        _, _, metrics = train_and_save_model(df) 
        st.sidebar.info("Pre-trained model loaded.")
    except Exception:
        # Fallback if files are corrupted or fail to load
        st.sidebar.error("Error loading saved model. Retraining.")
        model, vectorizer, metrics = train_and_save_model(df)
else:
    st.sidebar.info("Training and saving new model...")
    model, vectorizer, metrics = train_and_save_model(df)


## --- 4. Prediction Logic ---
def classify_message(model, vectorizer, text):
    """Predicts if a given message is spam or not."""
    if model is None or vectorizer is None:
        return "Model unavailable."
        
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)[0]
    
    return "SPAM 🚨" if prediction == 1 else "HAM (Not Spam) ✅"

# --- Streamlit App Layout ---
st.title("✉️ Weekend ML Project: Enhanced Spam/Ham Classifier")
st.markdown("---")

# Display the Distribution Plot using Streamlit's native plotting function
st.header("1. Data Overview: Message Distribution")
if not df.empty:
    st.pyplot(create_distribution_plot(df)) # <--- **FIXED LINE**
else:
    st.warning("Cannot display data overview as the dataset failed to load.")

st.markdown("---")

# Add Text Classification UI
st.header("2. Test Your Message")
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Paste your message here:", 
        "You have been selected to receive a FREE £1000 prize! Text CLAIM to 87021 to collect now.",
        height=100,
        label_visibility="collapsed"
    )

with col2:
    # Add an empty space so the button aligns better with the text area
    st.write(" ") 
    if st.button("Classify Message", use_container_width=True):
        if user_input and model and vectorizer:
            with st.spinner('Running classification...'):
                result = classify_message(model, vectorizer, user_input)
            
            st.subheader("Classification Result:")
            if "SPAM" in result:
                st.error(result)
            else:
                st.success(result)
        else:
            st.warning("Please enter a message and ensure the model loaded successfully.")

# --- Sidebar Updates (Algorithm and Metrics) ---
st.sidebar.header("Model Configuration")
if metrics:
    st.sidebar.markdown(f"**Algorithm:** `{metrics.get('Algorithm', 'N/A')}`")
    st.sidebar.markdown(f"**Vectorizer:** `{metrics.get('Vectorizer', 'N/A')}`")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Test Set Performance (20%)")
    
    # Display all key metrics
    st.sidebar.metric(label="Accuracy", value=f"{metrics['Accuracy']:.4f}")
    st.sidebar.metric(label="Precision", value=f"{metrics['Precision']:.4f}")
    st.sidebar.metric(label="Recall (Spam Detection)", value=f"{metrics['Recall']:.4f}")
    st.sidebar.metric(label="F1-Score", value=f"{metrics['F1-Score']:.4f}")