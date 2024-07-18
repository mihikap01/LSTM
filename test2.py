import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ann_model = load_model("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/bi_lstm_attention_model_with_negation_handling.keras")

tokenizer = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/tokenizer.pickle", "rb"))
vectorizer = pickle.load(open("/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/vectorizer.pkl", "rb"))

user_input = "The prequel was good last time but this time it was a snooze fest"

def predict_sentiment(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=200)
    prediction = ann_model.predict(padded_seq)
    # return "Positive" if prediction[0] > 0.5 else "Negative"
    if prediction[0] > 0.5:
        print("Positive")
    else:
        print("negative")

predict_sentiment(user_input)