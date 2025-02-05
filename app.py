import streamlit as st
import pickle
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Set background image (use local path or URL)
background_image_url = "url('https://st4.depositphotos.com/9999814/37811/i/450/depositphotos_378110030-stock-photo-customer-review-satisfaction-feedback-survey.jpg')"

# Apply background image using custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: {background_image_url};
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        min-height: 100vh;
    }}
    h1 {{
        color: #FF6F61;  /* Light Coral for Main Heading */
        text-align: center;
    }}
    h4 {{
        color: #4682B4;  /* Steel Blue for Sub-heading */
        text-align: center;
    }}
    .prediction-box {{
        border: 2px solid #ddd;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }}
    .positive {{
        color: #228B22;  /* Forest Green for Positive Sentiment */
    }}
    .negative {{
        color: #DC143C;  /* Crimson for Negative Sentiment */
    }}
    
    </style>
    """,
    unsafe_allow_html=True
)

# Load pre-trained model and vectorizer (ensure they are saved beforehand)
with open(r"D:\FULLSTACK DATASCIECE AND AI\Classroomwork\ARTIFICIALINTELLIGENCE\NATURAL LANGUAGE PROCESSING\NLPPTOJECTS\PROJECTNLPBYU\CUSTOMER FEEDBACK\lgbm_model1.pkl", "rb") as model_file:
    lgbm_model = pickle.load(model_file)

with open(r"D:\FULLSTACK DATASCIECE AND AI\Classroomwork\ARTIFICIALINTELLIGENCE\NATURAL LANGUAGE PROCESSING\NLPPTOJECTS\PROJECTNLPBYU\CUSTOMER FEEDBACK\tfidf_vectorizer1.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Streamlit App UI
st.markdown("<h1>Customer Feedback Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h4>Predict whether customer feedback is Positive (1) or Negative (0).</h4>", unsafe_allow_html=True)

# Input from user
# Heading for the user input area
st.markdown("<h5 style='text-align: center; color:  green;'>Enter a customer review:</h5>", unsafe_allow_html=True)

# Input from user (text area)
user_input = st.text_area("", "",label_visibility="visible")

# Preprocess and predict
if st.button("Predict Sentiment"):
    if user_input.strip():
        # Preprocess the input
        user_input_processed = tfidf.transform([user_input])

        # Predict sentiment
        prediction = lgbm_model.predict(user_input_processed)

        # Determine sentiment and color
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        sentiment_color = "positive" if prediction[0] == 1 else "negative"

        # Display result in a box-like element
        st.markdown(f"<div class='prediction-box {sentiment_color}'>The predicted sentiment is: **{sentiment}**</div>", unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown("<p style='text-align: center; color: red;'>Model: LightGBM |Developed by <strong>Sainath</strong></p>", unsafe_allow_html=True)
