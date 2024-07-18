# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle
# import re
#
# # Load the tokenizer
# with open('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# # Load the model
# model = tf.keras.models.load_model('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/bi_lstm_attention_model_with_negation_handling.keras')
#
# # Streamlit app
# st.title("Movie Review Sentiment Analysis")
#
# # User input
# user_input = st.text_area("Enter your movie review:")
#
# # Predict sentiment
# if st.button("Predict Sentiment"):
#     # Handle negations in user input
#     processed_input = re.sub(r'\bnot\s+', 'not_', user_input)
#     sequences = tokenizer.texts_to_sequences([processed_input])
#     padded_input = pad_sequences(sequences, maxlen=200)
#
#     # Make prediction
#     prediction = model.predict(padded_input)
#     sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
#
#     # Display result
#     st.write(f"Predicted Sentiment: {sentiment}")


import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ann_model = load_model("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/bi_lstm_attention_model_with_negation_handling.keras")

tokenizer = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/tokenizer.pickle", "rb"))
vectorizer = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/vectorizer.pkl", "rb"))

user_input = "The movie was bad"

def predict_sentiment(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=200)
    prediction = ann_model.predict(padded_seq)
    return "Positive" if prediction[0] > 0.5 else "Negative"

predict_sentiment(user_input)