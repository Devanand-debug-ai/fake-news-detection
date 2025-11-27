import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="TruthLens | AI Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #212529; }

    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        color: #1a1a1a;
    }

    .main-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 0 0 20px 20px;
        margin-bottom: 3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 3rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }

    .stTextArea textarea {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.05rem;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #2a5298;
        background-color: #ffffff;
        box-shadow: 0 0 0 4px rgba(42,82,152,0.1);
    }

    .stButton button {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(30,60,114,0.3);
    }

    .result-card {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 1rem;
        border-top: 6px solid #ccc;
        transition: transform 0.3s ease;
    }

    .result-card:hover { transform: translateY(-5px); }

    .confidence-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }

    .confidence-bar {
        height: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# HELPER FUNCTION
# ------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ------------------------------------------------------
# LOAD MODEL + VECTORIZER
# ------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("fake_news_model.pkl", "rb"))
        vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model()

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/news.png", width=150)
    st.title("About TruthLens")

    st.info("""
        **TruthLens** uses machine learning models  
        to classify fake vs real news.

        **Model:** Passive Aggressive Classifier  
        **Accuracy:** ~93%  
    """)

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Paste the article text  
    2. Click **Analyze**  
    3. View prediction + confidence  
    """)

    st.markdown("---")
    st.caption("v2.1.0 | Built by Antigravity")

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>TruthLens</h1>
    <div class="subtitle">Professional Misinformation Detection System</div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# MAIN CONTENT
# ------------------------------------------------------
if model is None or vectorizer is None:
    st.error("❌ Model or vectorizer file not found. Ensure .pkl files are in the working directory.")
else:
    col1, col2 = st.columns([1.8, 1.2])

    # ------------------------- INPUT AREA ------------------------
    with col1:
        st.markdown("### 📝 Input Article")
        st.markdown('<div class="input-container">', unsafe_allow_html=True)

        news_text = st.text_area(
            "Paste content here",
            height=350,
            label_visibility="collapsed",
            placeholder="Paste the news article here..."
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Analyze Content"):
            if news_text.strip() == "":
                st.warning("Please enter text to analyze.")
            else:
                with st.spinner("Analyzing content..."):
                    time.sleep(0.5)

                    tfidf_test = vectorizer.transform([news_text])
                    prediction = model.predict(tfidf_test)[0]

                    # CONFIDENCE HANDLING
                    if hasattr(model, "decision_function"):
                        raw = model.decision_function(tfidf_test)[0]
                        prob = sigmoid(raw) if raw > 0 else 1 - sigmoid(raw)
                    elif hasattr(model, "predict_proba"):
                        prob = max(model.predict_proba(tfidf_test)[0])
                    else:
                        prob = 0.50

                    st.session_state["done"] = True
                    st.session_state["prediction"] = prediction
                    st.session_state["confidence"] = round(prob * 100, 1)

    # ------------------------- RESULTS AREA ------------------------
    with col2:
        st.markdown("### 📊 Analysis Result")

        if st.session_state.get("done"):
            pred = st.session_state["prediction"]
            conf = st.session_state["confidence"]

            if pred == 1:
                label = "Likely Authentic"
                color = "#28a745"
                icon = "✅"
            else:
                label = "Likely Fake"
                color = "#dc3545"
                icon = "🚨"

            st.markdown(f"""
            <div class="result-card">
                <h3 style="color:{color}; font-size:2.2rem; font-weight:800;">{label} {icon}</h3>
                <div class="confidence-section">
                    <div style="display:flex; justify-content:space-between; font-weight:600;">
                        <span>Confidence</span>
                        <span>{conf}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{conf}%; background:{color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; background:white; border-radius:15px; border:2px dashed #e9ecef;">
                <div style="font-size:4rem; opacity:0.3;">🔍</div>
                <h3 style="color:#adb5bd;">Ready to Analyze</h3>
                <p style="color:#ced4da;">Paste an article and click Analyze.</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#6c757d; font-size:0.9rem; padding-bottom:2rem;">
    © 2025 TruthLens Analytics. All rights reserved.
</div>
""", unsafe_allow_html=True)
