Fake News Detection System : https://fake-news-detection-em9zcbsx4gb97ivbjvh6be.streamlit.app/

This project is a Fake News Detection system built using Python and machine learning.

It classifies news articles as either REAL or FAKE based on their text content.

Features:

Uses TF-IDF vectorization to convert text into numerical features.

Implements a Logistic Regression (or Passive Aggressive) classifier for prediction.

Provides a Streamlit web app interface for easy news classification.

Trained on publicly available datasets of fake and real news articles.

Deployment
This app is deployed on Streamlit Cloud and can be accessed via the public URL:
https://fake-news-detection-em9zcbsx4gb97ivbjvh6be.streamlit.app/

Dataset
The model is trained on two datasets:
Fake.csv — contains fake news articles
True.csv — contains real news articles

Note: These CSV files are not included in the repository due to their large size.

Dependencies:
pandas
numpy
scikit-learn
joblib
streamlit

Example of REAL news detection
![WhatsApp Image 2025-05-20 at 23 50 39_fa4a2f8b](https://github.com/user-attachments/assets/2cddca11-75ab-456f-8066-2988392b988a)

Example of FAKE news detection
![WhatsApp Image 2025-05-20 at 23 52 48_9a9c5993](https://github.com/user-attachments/assets/1d9e31b5-c297-4bd2-a369-f688ce15fc37)


