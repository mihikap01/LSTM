import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
print('one')
# Load models
lstm_model = load_model("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/reviews_model.keras")
bi_lstm_model = load_model("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/bi_lstm_model.keras")
ann_model = load_model("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/bi_lstm_attention_model_with_negation_handling.keras")
log_reg_model = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/log_reg_model.pkl", "rb"))
naive_bayes_model = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/naive_bayes_model.pkl", "rb"))
print('one')
# Load tokenizer and vectorizer
tokenizer = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/tokenizer.pickle", "rb"))
vectorizer = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/vectorizer.pkl", "rb"))
print('one')
# Model dictionary
models = {
    "LSTM": lstm_model,
    "Bi-LSTM": bi_lstm_model,
    "ANN": ann_model,
    "Logistic Regression": log_reg_model,
    "Naive Bayes": naive_bayes_model
}

# Streamlit app
st.title("Sentiment Analysis Web App")
st.write("Enter a review and select a model to predict the sentiment")

# User input
user_input = st.text_area("Enter the review:")

# Model selection
model_option = st.selectbox("Choose the model for prediction:", models.keys())

# Predict function
def predict_sentiment(model, user_input, model_type):
    if model_type in ["LSTM", "Bi-LSTM", "ANN"]:
        # Preprocess for LSTM-based models
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=200)  # Adjust maxlen as per your model's training
        prediction = model.predict(padded_seq)
        return "Positive" if prediction[0] > 0.5 else "Negative"
    elif model_type in ["Logistic Regression", "Naive Bayes"]:
        # Preprocess for Logistic Regression and Naive Bayes models
        processed_input = vectorizer.transform([user_input])
        prediction = model.predict(processed_input)
        return "Positive" if prediction[0] == 1 else "Negative"

# On button click
if st.button("Predict Sentiment"):
    prediction = predict_sentiment(models[model_option], user_input, model_option)
    st.write(f"Predicted Sentiment: {prediction}")
