import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.title {
font-size:65px;
font-weight:800;
text-align:center;
color:#1F618D;
}
.subtitle {
text-align:center;
font-size:22px;
color:gray;
margin-bottom:20px;
}
.result-box {
padding:20px;
border-radius:12px;
font-size:20px;
text-align:center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">📧 AI Email Spam Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect spam emails using Machine Learning</p>', unsafe_allow_html=True)
st.divider()

# Input Section
st.markdown("### ✉ Enter Email Message")
message = st.text_area(
    "Paste the email content below",
    height=150
)

# Prediction
if st.button("🔍 Analyze Email"):
    if message.strip() == "":
        st.warning("⚠ Please enter an email message first.")
    else:

        # Clean text
        cleaned = re.sub(r'[^a-zA-Z]', ' ', message.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Vectorize
        vector = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vector)[0]

        # Probability
        prob = model.predict_proba(vector)[0][prediction]
        st.divider()
        if prediction == 1:
            st.error(f"🚨 This message is **SPAM** (Confidence: {prob*100:.2f}%)")
        else:
            st.success(f"✅ This message is **NOT Spam** (Confidence: {prob*100:.2f}%)")
st.divider()

# Model Information
st.markdown("""
### 🤖 Model Details

**Model Used:** Multinomial Naive Bayes  

**Text Processing:**
- Lowercasing
- Regex cleaning
- TF-IDF vectorization

**Accuracy:** ~98%

**Use Case:** Email / SMS spam detection
""")
