import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="MailGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.stApp {
    background-color: #0d1117;
}

.mailguard-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 2rem 0 1rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}

.mailguard-logo {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: -1px;
}

.mailguard-tagline {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 2px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.model-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.model-card.spam {
    border-color: #da3633;
    background: #1a0f0f;
}

.model-card.ham {
    border-color: #238636;
    background: #0d1a10;
}

.model-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.verdict-text {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.verdict-spam { color: #f85149; }
.verdict-ham  { color: #3fb950; }

.confidence-text {
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 0.8rem;
}

.threshold-badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
}

.conf-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    margin: 8px 0 12px 0;
    overflow: hidden;
}

.conf-bar-fill-spam {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #da3633, #f85149);
}

.conf-bar-fill-ham {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #238636, #3fb950);
}

.verdict-banner {
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.verdict-banner.spam {
    background: #1a0f0f;
    border: 1px solid #da3633;
}

.verdict-banner.ham {
    background: #0d1a10;
    border: 1px solid #238636;
}

.verdict-banner-title {
    font-size: 1.3rem;
    font-weight: 700;
}

.verdict-banner-sub {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 2px;
}

.token-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #79c0ff;
}

.token-pill.high {
    border-color: #f85149;
    color: #f85149;
    background: #1a0f0f;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
    margin-top: 1.5rem;
}

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8b949e;
    border-radius: 6px;
    font-size: 0.85rem;
}

.stTabs [aria-selected="true"] {
    background-color: #21262d !important;
    color: #e6edf3 !important;
}

.stButton > button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #2ea043;
    border: none;
}

.stTextArea textarea {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
}

.stTextArea textarea:focus {
    border-color: #58a6ff;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1);
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    vectorizer = joblib.load('models/vectorizer.pkl')
    models = {
        'Naive Bayes':         joblib.load('models/naive_bayes.pkl'),
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'LinearSVC':           joblib.load('models/linear_svc.pkl')
    }
    return vectorizer, models

vectorizer, models = load_models()

BEST_THRESHOLDS = {
    'Naive Bayes':         0.4664,
    'Logistic Regression': 0.5337,
    'LinearSVC':           0.4045
}

MODEL_INFO = {
    'Naive Bayes':         'Fast probabilistic classifier. Handles imbalanced data naturally.',
    'Logistic Regression': 'Linear model with balanced class weights. Best overall accuracy.',
    'LinearSVC':           'Support vector classifier. Most balanced precision/recall tradeoff.'
}

def predict_email(email_text, threshold_mode, manual_threshold=0.5):
    email_tfidf = vectorizer.transform([email_text])
    predictions = {}

    for name, model in models.items():
        prob_spam = model.predict_proba(email_tfidf)[0][1]
        threshold = BEST_THRESHOLDS[name] if threshold_mode == "Auto" else manual_threshold
        is_spam   = prob_spam >= threshold
        predictions[name] = {
            'prob_spam': prob_spam,
            'prob_ham':  1 - prob_spam,
            'is_spam':   is_spam,
            'threshold': threshold,
            'verdict':   'SPAM' if is_spam else 'HAM'
        }

    return predictions

def get_top_tokens(email_text, n=10):
    email_tfidf  = vectorizer.transform([email_text])
    feature_names = vectorizer.get_feature_names_out()
    scores        = email_tfidf.toarray()[0]
    top_indices   = scores.argsort()[-n:][::-1]
    return [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]

def render_model_card(name, result):
    verdict  = result['verdict']
    prob     = result['prob_spam'] if verdict == 'SPAM' else result['prob_ham']
    bar_pct  = int(prob * 100)
    cls      = 'spam' if verdict == 'SPAM' else 'ham'
    emoji    = '🚨' if verdict == 'SPAM' else '✅'
    v_cls    = 'verdict-spam' if verdict == 'SPAM' else 'verdict-ham'
    bar_cls  = 'conf-bar-fill-spam' if verdict == 'SPAM' else 'conf-bar-fill-ham'

    st.markdown(f"""
    <div class="model-card {cls}">
        <div class="model-name">{name}</div>
        <div class="verdict-text {v_cls}">{emoji} {verdict}</div>
        <div class="confidence-text">Confidence: {prob:.1%}</div>
        <div class="conf-bar-bg">
            <div class="{bar_cls}" style="width:{bar_pct}%"></div>
        </div>
        <span class="threshold-badge">threshold = {result['threshold']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="mailguard-header">
    <div>
        <div class="mailguard-logo">🛡️ MailGuard</div>
        <div class="mailguard-tagline">ML-Powered Spam Detection · 3 Models · Threshold Tuning</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    threshold_mode = st.radio(
        "Threshold Mode",
        ["Auto (recommended)", "Manual"],
        help="Auto uses the best threshold found during evaluation. Manual lets you control sensitivity."
    )

    manual_threshold = 0.5
    if threshold_mode == "Manual":
        st.markdown(" ")
        manual_threshold = st.slider(
            "Spam Sensitivity",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.01,
            help="Lower = catches more spam but may flag legitimate emails. Higher = more precise but may miss some spam."
        )
        col1, col2 = st.columns(2)
        col1.metric("Current", f"{manual_threshold:.2f}")
        col2.metric("Default", "0.50")

        if manual_threshold < 0.4:
            st.warning("⚠️ Low threshold — expect more false positives")
        elif manual_threshold > 0.7:
            st.warning("⚠️ High threshold — some spam may slip through")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    for name, info in MODEL_INFO.items():
        with st.expander(name):
            st.caption(info)
            if threshold_mode == "Auto (recommended)":
                st.code(f"Best threshold: {BEST_THRESHOLDS[name]}")

    st.markdown("---")
    st.markdown("### 📈 Evaluation Results")
    st.caption("From evaluate.py on test set")
    results_data = {
        'Model':     ['Naive Bayes', 'Logistic Reg', 'LinearSVC'],
        'Accuracy':  ['98.53%', '98.73%', '98.73%'],
        'Spam F1':   ['98.30%', '98.54%', '98.54%']
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(results_data), hide_index=True, use_container_width=True)

tab1, tab2 = st.tabs(["✏️  Paste Email", "📁  Upload File"])

email_text = ""

with tab1:
    email_input = st.text_area(
        "Paste your email content below",
        placeholder="Subject: Congratulations! You've won a free prize...\n\nDear winner, click here to claim your reward...",
        height=200,
        label_visibility="collapsed"
    )
    if email_input:
        email_text = email_input

with tab2:
    uploaded_file = st.file_uploader(
        "Upload a .txt or .eml file",
        type=['txt', 'eml'],
        label_visibility="collapsed"
    )
    if uploaded_file:
        email_text = uploaded_file.read().decode('utf-8', errors='ignore')
        st.success(f"✅ File loaded: {uploaded_file.name} ({len(email_text)} characters)")
        with st.expander("Preview file content"):
            st.text(email_text[:500] + ("..." if len(email_text) > 500 else ""))

mode = "Auto" if threshold_mode == "Auto (recommended)" else "Manual"
analyze = st.button("🔍 Analyze Email", use_container_width=True)

if analyze:
    if not email_text.strip():
        st.error("⚠️ Please paste or upload an email first.")
    else:
        with st.spinner("Analyzing..."):
            predictions = predict_email(email_text, mode, manual_threshold)
            top_tokens  = get_top_tokens(email_text)

        spam_votes = sum(1 for r in predictions.values() if r['is_spam'])
        is_spam_overall = spam_votes >= 2
        overall_verdict = "SPAM" if is_spam_overall else "HAM"

        banner_cls   = "spam" if is_spam_overall else "ham"
        banner_emoji = "🚨" if is_spam_overall else "✅"
        banner_color = "#f85149" if is_spam_overall else "#3fb950"
        vote_text    = f"{spam_votes}/3 models flagged as spam"

        st.markdown(f"""
        <div class="verdict-banner {banner_cls}">
            <div style="font-size:2.5rem">{banner_emoji}</div>
            <div>
                <div class="verdict-banner-title" style="color:{banner_color}">
                    Overall Verdict: {overall_verdict}
                </div>
                <div class="verdict-banner-sub">{vote_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Model Breakdown</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for col, (name, result) in zip(cols, predictions.items()):
            with col:
                render_model_card(name, result)

        st.markdown('<div class="section-label">Why was this flagged?</div>', unsafe_allow_html=True)

        if top_tokens:
            max_score = max(score for _, score in top_tokens)
            pills_html = ""
            for token, score in top_tokens:
                is_high = score > (max_score * 0.5)
                cls     = "token-pill high" if is_high else "token-pill"
                pills_html += f'<span class="{cls}">{token}</span>'

            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #21262d; border-radius:10px; padding:1rem; margin-top:0.5rem">
                <div style="font-size:0.8rem; color:#8b949e; margin-bottom:0.6rem">
                    Top TF-IDF weighted tokens — red = strongest spam signal
                </div>
                {pills_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No significant tokens found — email may be too short.")

        st.markdown('<div class="section-label">Probability Breakdown</div>', unsafe_allow_html=True)
        prob_data = {
            'Model':       list(predictions.keys()),
            'Spam Prob':   [f"{r['prob_spam']:.4f}" for r in predictions.values()],
            'Ham Prob':    [f"{r['prob_ham']:.4f}"  for r in predictions.values()],
            'Threshold':   [f"{r['threshold']:.4f}" for r in predictions.values()],
            'Verdict':     [r['verdict']             for r in predictions.values()]
        }
        st.dataframe(pd.DataFrame(prob_data), hide_index=True, use_container_width=True)

elif not email_text:
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#8b949e;">
        <div style="font-size:3rem; margin-bottom:1rem">📧</div>
        <div style="font-size:1rem; font-weight:500; color:#e6edf3">Paste or upload an email to get started</div>
        <div style="font-size:0.85rem; margin-top:0.5rem">
            MailGuard runs 3 ML models simultaneously and shows you exactly why an email was flagged
        </div>
    </div>
    """, unsafe_allow_html=True)