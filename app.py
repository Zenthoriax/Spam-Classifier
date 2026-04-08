import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from backend import SpamClassifierBackend

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Sentin-AI | Premium Spam Classifier",
    page_icon="📨",
    layout="wide",
)

# --------------------------------------------------
# Custom Premium CSS
# --------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    .stat-val {
        font-size: 2rem;
        font-weight: 600;
        color: #a855f7;
    }

    .nlp-step {
        border-left: 3px solid #6366f1;
        padding-left: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Backend Initialization
# --------------------------------------------------
@st.cache_resource
def get_backend():
    return SpamClassifierBackend()

backend = get_backend()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield.png", width=80)
    st.title("Sentin-AI")
    st.markdown("Automated Email & SMS spam detection with deep neural insights.")
    st.markdown("---")
    
    stats = backend.get_stats()
    st.header("Dataset Overview")
    colA, colB = st.columns(2)
    with colA:
        st.metric("HAM", stats.get('ham', 0))
    with colB:
        st.metric("SPAM", stats.get('spam', 0))

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.title("📨 AI Message Guardian")
st.markdown("Detect anomalies and malicious content in your messages using Multinomial Bayes & NLP Analysis.")

tab1, tab2, tab3 = st.tabs(["🔒 Classifier", "🧠 NLP Insights", "📊 Analytics"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Paste message content below:",
        placeholder="e.g., Congratulations! You've won a $1000 Gift Card. Click here to claim.",
        height=150
    )
    
    if st.button("Analyze Security"):
        if user_input.strip():
            with st.spinner("Decoding transmission..."):
                result = backend.predict(user_input)
                
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if result["prediction"] == "SPAM":
                        st.error(f"### 🚨 {result['prediction']}")
                    else:
                        st.success(f"### ✅ {result['prediction']}")
                    
                    st.metric("Confidence", f"{result['confidence']:.2f}%")
                
                with col2:
                    st.info("💡 **Model Note:** Prediction based on frequency analysis of tokens against the provided dataset.")
                    st.write(f"Processed as: `{result['nlp_steps']['processed_text']}`")
                
                # Store in session state for tab2
                st.session_state['last_result'] = result
        else:
            st.warning("Input required for analysis.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        nlp = res['nlp_steps']
        
        st.header("NLP Pipeline Execution")
        
        col_nlp1, col_nlp2 = st.columns(2)
        
        with col_nlp1:
            st.markdown("<div class='nlp-step'>", unsafe_allow_html=True)
            st.subheader("1. Tokenization")
            st.write("Breaking string into discrete lexical units.")
            st.json(nlp['tokens'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='nlp-step'>", unsafe_allow_html=True)
            st.subheader("3. Stopword Removal")
            st.write("Filtering non-essential words (the, is, at, etc.).")
            st.write(f"Original Count: {len(nlp['tokens'])} → Filtered: {len(nlp['cleaned_tokens'])}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_nlp2:
            st.markdown("<div class='nlp-step'>", unsafe_allow_html=True)
            st.subheader("2. Part-of-Speech Tagging")
            st.write("Identifying grammatical categories (NN=Noun, VB=Verb).")
            st.dataframe(pd.DataFrame(nlp['pos_tags'], columns=["Token", "Tag"]), height=250)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Run an analysis in the 'Classifier' tab to see NLP insights.")

with tab3:
    st.header("Dataset Deep Dive")
    
    @st.cache_data
    def get_df():
        df = pd.read_csv("spam.csv", encoding="latin-1", usecols=["v1", "v2"])
        df = df.rename(columns={"v1": "label", "v2": "message"})
        return df

    df = get_df()
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.subheader("Word Frequency (WordCloud)")
        label_filter = st.radio("Show Cloud for:", ["spam", "ham"], horizontal=True)
        text = " ".join(df[df['label'] == label_filter]['message'])
        wc = WordCloud(width=800, height=400, background_color=None, mode="RGBA").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    with col_v2:
        st.subheader("Distribution Balance")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='label', data=df, palette=['#6366f1', '#a855f7'], ax=ax2)
        st.pyplot(fig2)
