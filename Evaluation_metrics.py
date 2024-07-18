'''
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained models
lstm_model = load_model("reviews_model.keras")
bi_lstm_model = load_model("bi_lstm_model.keras")
ann_model = load_model("bi_lstm_attention_model.keras")
log_reg_model = pickle.load(open("log_reg_model.pkl", "rb"))
naive_bayes_model = pickle.load(open("naive_bayes_model.pkl", "rb"))

# Load tokenizer
tokenizer = pickle.load(open("tokenizer.pickle", "rb"))

# Load test data
test_df = pd.read_csv('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/IMDB_Dataset.csv')
X_test = test_df['review']  # Replace with your actual text column name
y_test = test_df['sentiment']  # Replace with your actual label column name

# Convert y_test to numerical if it's in string format
if y_test.dtype == 'object':
    y_test = y_test.map({'positive': 1, 'negative': 0}).values

# Preprocess test data
max_length = 200  # Adjust this based on your model's training configuration
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length)


# Define a function to evaluate the models
def evaluate_model(model, X_test, y_test, is_neural_network=True):
    if is_neural_network:
        predictions = model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        # Reshape for sklearn models if necessary
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


# Evaluate each model
# LSTM
accuracy, precision, recall, f1 = evaluate_model(lstm_model, X_test_padded, y_test)
print(f"LSTM - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Bi-LSTM
accuracy, precision, recall, f1 = evaluate_model(bi_lstm_model, X_test_padded, y_test)
print(f"Bi-LSTM - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# ANN
accuracy, precision, recall, f1 = evaluate_model(ann_model, X_test_padded, y_test)
print(f"ANN - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Logistic Regression
# Note: Ensure that X_test is transformed the same way as during training
accuracy, precision, recall, f1 = evaluate_model(log_reg_model, X_test_padded, y_test, is_neural_network=False)
print(f"Logistic Regression - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Naive Bayes
# Note: Ensure that X_test is transformed the same way as during training
accuracy, precision, recall, f1 = evaluate_model(naive_bayes_model, X_test_padded, y_test, is_neural_network=False)
print(f"Naive Bayes - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


'''

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained models
lstm_model = load_model("reviews_model.keras")
bi_lstm_model = load_model("bi_lstm_model.keras")
ann_model = load_model("bi_lstm_attention_model.keras")
log_reg_model = pickle.load(open("log_reg_model.pkl", "rb"))
naive_bayes_model = pickle.load(open("naive_bayes_model.pkl", "rb"))

# Load tokenizer and vectorizer
tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))  # Load your vectorizer here

# Load test data
test_df = pd.read_csv(
    '/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/IMDB_Dataset.csv')
X_test = test_df['review']
y_test = test_df['sentiment'].map({'positive': 1, 'negative': 0}).values

# Preprocess test data for LSTM models
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=200)

# Preprocess test data for Logistic Regression and Naive Bayes
X_test_vectorized = vectorizer.transform(X_test)


# Define a function to evaluate the models
def evaluate_model(model, X_test, y_test, is_neural_network=True):
    if is_neural_network:
        predictions = model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


# Evaluate each model
# LSTM, Bi-LSTM, ANN
# LSTM
accuracy, precision, recall, f1 = evaluate_model(lstm_model, X_test_padded, y_test)
print(f"LSTM - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Bi-LSTM
accuracy, precision, recall, f1 = evaluate_model(bi_lstm_model, X_test_padded, y_test)
print(f"Bi-LSTM - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# ANN
accuracy, precision, recall, f1 = evaluate_model(ann_model, X_test_padded, y_test)
print(f"ANN - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

for model, name in zip([lstm_model, bi_lstm_model, ann_model], ["LSTM", "Bi-LSTM", "ANN"]):
    accuracy, precision, recall, f1 = evaluate_model(model, X_test_padded, y_test)
    print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Logistic Regression, Naive Bayes
for model, name in zip([log_reg_model, naive_bayes_model], ["Logistic Regression", "Naive Bayes"]):
    accuracy, precision, recall, f1 = evaluate_model(model, X_test_vectorized, y_test, is_neural_network=False)
    print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


# Evaluation metrics(output):
# LSTM - Accuracy: 0.94646, Precision: 0.9423374152895019, Recall: 0.95112, F1 Score: 0.9467083391396094
# Bi-LSTM - Accuracy: 0.9434, Precision: 0.9218999771637361, Recall: 0.96888, F1 Score: 0.9448063345945314
# ANN - Accuracy: 0.96452, Precision: 0.956127258444619, Recall: 0.97372, F1 Score: 0.964843440348791
# Logistic Regression - Accuracy: 0.90876, Precision: 0.902386202551583, Recall: 0.91668, F1 Score: 0.9094769426144934
# Naive Bayes - Accuracy: 0.84864, Precision: 0.8397785747699984, Recall: 0.86168, F1 Score: 0.8505883282002684