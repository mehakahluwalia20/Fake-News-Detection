Fake News Detection

This project is a Fake News Detection system built using Python and machine learning.

It classifies news articles as either REAL or FAKE based on their text content.

Features:

Uses TF-IDF vectorization to convert text into numerical features.

Implements a Logistic Regression (or Passive Aggressive) classifier for prediction.

Provides a Streamlit web app interface for easy news classification.

Trained on publicly available datasets of fake and real news articles.

Project Structure

Fake-News-Detection/

app.py                 # Streamlit app to input news and get prediction

train_model.py         # Script to train and save the ML model and vectorizer

model.joblib           # Trained model file (do not upload large CSVs)

vectorizer.joblib      # TF-IDF vectorizer used for feature extraction

 requirements.txt       # Python dependencies

Deployment
This app is deployed on Streamlit Cloud and can be accessed via the public URL:
https://fake-news-detection-em9zcbsx4gb97ivbjvh6be.streamlit.app/

Dataset
The model is trained on two datasets:
Fake.csv — contains fake news articles
True.csv — contains real news articles

Note: These CSV files are not included in the repository due to their large size.

Dependencies
pandas
numpy
scikit-learn
joblib
streamlit
