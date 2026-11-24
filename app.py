"""
================================================================================
Email Spam Detection - Streamlit Web Application
================================================================================
A beginner-friendly, interactive Web UI to test email and SMS spam detection in
real-time using your trained Machine Learning model.

Run with:
    streamlit run app.py
================================================================================
"""

import os
import joblib
import streamlit as st

# Configure the Streamlit page
st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "model.pkl"

# Preset examples for easy 1-click testing
SAMPLE_EMAILS = {
    "Spam Example 1 (Lottery / Prize)": (
        "CONGRATULATIONS! You have been selected as the WINNER of a $1,000,000 cash prize! "
        "Claim your reward now by calling 0800-888-999 immediately. Valid for 24 hours only!"
    ),
    "Spam Example 2 (Phishing / Account Urgency)": (
        "URGENT: Your account access has been restricted due to suspicious activity. "
        "Click the link below to verify your password and identity now or your account will be suspended."
    ),
    "Ham Example 1 (Work / Project Update)": (
        "Hi Alex, please find attached the revised quarterly sales report. "
        "Let me know if you have any questions or feedback before our 3 PM meeting today."
    ),
    "Ham Example 2 (Casual / Personal)": (
        "Hey! Are you still free for lunch tomorrow? Let's catch up at the cafe around noon. See you soon!"
    )
}

# Common spam trigger words for keyword highlighting
TRIGGER_WORDS = [
    'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 'urgent!', 
    'selected', 'reward', 'call', 'txt', 'text', 'guaranteed', 'credit', 
    'bonus', 'lottery', 'congratulations', 'congrats', 'apply', 'offer', 
    '100%', 'risk-free', 'buy', 'order', 'deal', 'discount', 'expire', 
    'password', 'verify', 'account', 'suspended', 'click', 'link'
]

@st.cache_resource
def load_model():
    """Caches the model in memory so it doesn't reload on every interaction."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

# ==============================================================================
# UI HEADER
# ==============================================================================
st.title("🛡️ AI Email Spam Detector")
st.markdown(
    "Detect whether an incoming email or SMS is **Legitimate (Ham)** or **Spam / Phishing** "
    "using **Natural Language Processing (NLP)** and **Machine Learning**."
)

# Sidebar
st.sidebar.header("⚙️ Project Overview")
st.sidebar.markdown("""
- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF Vectorizer
- **Dataset:** 5,500+ Email/SMS samples
- **Accuracy:** ~98.3%
- **Training Program:** Deutsche Bank & Rubicon
- **Mentor:** Amol Navgire
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Try Quick Examples")
selected_sample = st.sidebar.selectbox(
    "Choose a preset message to test:",
    ["-- Select a sample --"] + list(SAMPLE_EMAILS.keys())
)

initial_text = ""
if selected_sample and selected_sample != "-- Select a sample --":
    initial_text = SAMPLE_EMAILS[selected_sample]

# ==============================================================================
# MAIN INPUT AREA
# ==============================================================================
model = load_model()

if model is None:
    st.error("⚠️ Model file `model.pkl` not found!")
    st.info("Please train the model first by opening your terminal and running:")
    st.code("python train.py", language="bash")
    st.stop()

# Text input box
email_input = st.text_area(
    "✉️ Paste or type the email message below:",
    value=initial_text,
    height=170,
    placeholder="e.g. You have won a $1000 gift card! Click here to claim your prize..."
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🔄 Clear", use_container_width=False)

if clear_button:
    st.rerun()

# ==============================================================================
# PREDICTION & RESULTS
# ==============================================================================
if analyze_button:
    clean_text = email_input.strip()
    
    if not clean_text:
        st.warning("⚠️ Please enter some email text before analyzing.")
    else:
        # Predict using pipeline
        prediction = model.predict([clean_text])[0]
        probabilities = model.predict_proba([clean_text])[0]
        
        ham_prob = probabilities[0] * 100
        spam_prob = probabilities[1] * 100
        is_spam = (prediction == 1)
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        if is_spam:
            st.error(f"### 🚨 Result: SPAM DETECTED")
            st.markdown(f"**Confidence:** **{spam_prob:.1f}%** confident this email is Spam.")
        else:
            st.success(f"### ✅ Result: LEGITIMATE (HAM)")
            st.markdown(f"**Confidence:** **{ham_prob:.1f}%** confident this email is Legitimate.")
            
        # Probability bars
        col_ham, col_spam = st.columns(2)
        with col_ham:
            st.metric("Legitimate (Ham) Probability", f"{ham_prob:.1f}%")
            st.progress(float(ham_prob / 100.0))
        with col_spam:
            st.metric("Spam Probability", f"{spam_prob:.1f}%")
            st.progress(float(spam_prob / 100.0))
            
        # Keyword triggers
        words = set(clean_text.lower().split())
        matched_triggers = [w for w in words if any(t in w for t in TRIGGER_WORDS)]
        
        st.markdown("#### 🔍 Detected Trigger Keywords")
        if matched_triggers:
            st.markdown(
                "The following words often correlate with spam/promotional content: " +
                " ".join([f"`{w}`" for w in matched_triggers])
            )
        else:
            st.markdown("No common high-risk spam keywords detected in this text.")

# ==============================================================================
# EDUCATIONAL SECTION
# ==============================================================================
with st.expander("🎓 How Does This AI Spam Filter Work? (Beginner Friendly)"):
    st.markdown("""
    This project demonstrates the classic machine learning workflow for text classification:
    
    1. **Text Preprocessing & Tokenization:**
       The raw message is cleaned (converted to lowercase, punctuation and standard English stopwords like *'the'*, *'is'* removed).
    
    2. **TF-IDF Vectorization:**
       Computers cannot read text directly, so we convert words into numbers using **TF-IDF (Term Frequency - Inverse Document Frequency)**:
       - **TF:** How often a word appears in this message.
       - **IDF:** Penalizes words that appear everywhere, and rewards words that are unique/informative (like *'lottery'* or *'jackpot'*).
       
    3. **Multinomial Naive Bayes:**
       A probabilistic classifier based on **Bayes' Theorem**. It calculates the likelihood of a message being Spam given the set of words it contains.
       
    4. **Decision & Confidence:**
       If the spam probability is > 50%, the message is classified as **Spam**, otherwise **Ham**.
    """)

st.markdown("---")
st.caption("Built with Python, Scikit-Learn, and Streamlit • Deutsche Bank Training Program by Rubicon (Mentor: Amol Navgire)")
