import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

st.title("Fake News Detection")

news_text = st.text_area("Enter news text to check:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        text_vec = vectorizer.transform([news_text])
        prediction = model.predict(text_vec)[0]
        st.success(f"Prediction: {prediction}")
