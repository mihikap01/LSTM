# LSTM model

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/IMDB_Dataset.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Tokenize text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
data = pad_sequences(sequences, maxlen=200)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, df['sentiment'].values, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(5000, 128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))

# Save the model and tokenizer
model.save('reviews_model.keras')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(len(df[df["sentiment"] == 1]))
print(len(df[df["sentiment"] == 0]))