import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set the page configuration for the Streamlit app
st.set_page_config(page_title= "Movie Review Sentiment Analysis", page_icon=":movie_camera:", layout="wide")

# Load the trained model
model = load_model("sentiment_model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Function for prediction
def predict_sentiment(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)[0][0]
    
    sentiment = "😊 Positive" if prediction > 0.5 else "😞 Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return sentiment, confidence

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and check its sentiment!")

# Text input
review_text = st.text_area("Enter your movie review here:", "")

if st.button("Analyze Sentiment"):
    if review_text.strip():
        sentiment, confidence = predict_sentiment(review_text)
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: **{confidence:.2%}**")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")


