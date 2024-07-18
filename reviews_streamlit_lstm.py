import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model and tokenizer
model = tf.keras.models.load_model('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/reviews_model.keras')
with open('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

st.title('IMDB Review Sentiment Analysis LSTM')

# User text input
user_input = st.text_area("Enter a review:")

if st.button('Predict'):
    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([user_input])
    data = pad_sequences(sequences, maxlen=200)

    # Predict sentiment
    prediction = model.predict(data)

    if prediction[0][0] > 0.5:
        st.write("Positive Sentiment")
    else:
        st.write("Negative Sentiment")
