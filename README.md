# Movie Review Sentiment Analysis

## Problem Statement

The IMDB dataset contains **50,000 movie reviews** for **Natural Language Processing (NLP)** or **Text Analytics**. This dataset is designed for **binary sentiment classification**, significantly larger than previous benchmark datasets. It includes **25,000 highly polar movie reviews** for training and **25,000 for testing**. The objective is to **predict the number of positive and negative reviews** using **classification or deep learning algorithms**.

## Project Overview

This project implements a **Deep Learning-based Sentiment Analysis Model** using **LSTM (Long Short-Term Memory)** neural networks to classify movie reviews as either **positive or negative**.

## Dataset

- **Source:** IMDB Dataset
- **Size:** 50,000 movie reviews
- **Training Data:** 25,000 reviews
- **Testing Data:** 25,000 reviews
- **Labels:**
  - **1**: Positive Review
  - **0**: Negative Review

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **NLTK** (Natural Language Toolkit)
- **Scikit-learn**
- **Pandas & NumPy**
- **Streamlit** (For Web App Deployment)

## Model Architecture

1. **Text Preprocessing:**
   - Tokenization using **NLTK**
   - Removing stopwords and special characters
   - Padding sequences for uniform input size

2. **Model Development:**
   - **Embedding Layer**: Converts words into dense vectors
   - **BiLSTM (Bidirectional Long Short-Term Memory)**: Captures context from both past and future words
   - **Fully Connected Layer**: Classifies reviews as positive or negative

3. **Training & Evaluation:**
   - Loss Function: **Binary Cross-Entropy**
   - Optimizer: **Adam**
   - Accuracy Metrics

## Streamlit Web Application

A simple user-friendly **Streamlit** web application is built to allow users to input a movie review and get a sentiment prediction instantly.

### Steps to Run the Project

1. **Install Dependencies:**
   ```sh
   pip install tensorflow pandas numpy nltk scikit-learn streamlit
   ```
2. **Run the Streamlit App:**
   ```sh
   streamlit run app.py
   ```

## Conclusion

This project demonstrates how to apply **Deep Learning** techniques to perform sentiment analysis on movie reviews. The **BiLSTM model** improves the accuracy by capturing both **past and future** context of words in a review.
