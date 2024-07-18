import streamlit as st
import pickle

# Load the Naive Bayes model and vectorizer
nb_model = pickle.load(open('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/naive_bayes_model.pkl', 'rb'))
vectorizer = pickle.load(open('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/vectorizer_nb.pkl', 'rb'))

st.title('IMDB Review Sentiment Analysis')

# User text input
user_input = st.text_area("Enter a review:")

if st.button('Predict'):
    # Vectorize the user input
    user_input_vec = vectorizer.transform([user_input])

    # Predict sentiment
    prediction = nb_model.predict(user_input_vec)

    if prediction[0] == 1:
        st.write("Positive Sentiment")
    else:
        st.write("Negative Sentiment")
