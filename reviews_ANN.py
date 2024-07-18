'''
# Training the model with bi- LSTM and an attention layer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate, Permute, Dot, Multiply, Flatten
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure you have the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('IMDB_Dataset.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Text Length Analysis
df['review_length'] = df['review'].apply(len)
avg_len_positive = df[df['sentiment'] == 1]['review_length'].mean()
avg_len_negative = df[df['sentiment'] == 0]['review_length'].mean()
print(f"Average length of positive reviews: {avg_len_positive}")
print(f"Average length of negative reviews: {avg_len_negative}")

# Function to preprocess and tokenize text
def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove punctuation and lowercase
    return tokens

# Split the dataset into positive and negative reviews
positive_reviews = df[df['sentiment'] == 1]['review']
negative_reviews = df[df['sentiment'] == 0]['review']

# Tokenize and count word frequencies in each class
positive_words = Counter(tokenize(' '.join(positive_reviews)))
negative_words = Counter(tokenize(' '.join(negative_reviews)))

# Print the most common words in positive reviews
print("Most common words in positive reviews:")
for word, freq in positive_words.most_common(20):
    print(f"{word}: {freq}")

# Print the most common words in negative reviews
print("\nMost common words in negative reviews:")
for word, freq in negative_words.most_common(20):
    print(f"{word}: {freq}")

# Data Preprocessing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
data = pad_sequences(sequences, maxlen=200)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, df['sentiment'], test_size=0.2, random_state=42)

# Attention Layer
def attention_layer(inputs, units):
    attention_score = Dense(units, activation='tanh')(inputs)
    attention_weights = Dense(1, activation='softmax')(attention_score)
    context_vector = Dot(axes=[1, 1])([attention_weights, inputs])
    return context_vector

# Model building
input_layer = Input(shape=(200,))
embedding_layer = Embedding(5000, 128, input_length=200)(input_layer)
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
attention = attention_layer(lstm_layer, 128)
flatten = Flatten()(attention)
output_layer = Dense(1, activation='sigmoid')(flatten)
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))

# Save the model
model.save('bi_lstm_attention_model.keras')
'''

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Flatten, Dropout
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/IMDB_Dataset.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Custom tokenizer function to handle negations like 'not bad'
def custom_tokenize(texts):
    updated_texts = []
    for text in texts:
        text = re.sub(r'\bnot\s+', 'not_', text)
        updated_texts.append(text)
    return updated_texts
df['review'] = custom_tokenize(df['review'])

# Data Preprocessing
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
data = pad_sequences(sequences, maxlen=200)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, df['sentiment'], test_size=0.2, random_state=42)

# Attention Layer
def attention_layer(inputs, units):
    attention_score = Dense(units, activation='tanh')(inputs)
    attention_weights = Dense(1, activation='softmax')(attention_score)
    context_vector = tf.matmul(attention_weights, inputs, transpose_a=True)
    context_vector = tf.squeeze(context_vector, -2)
    return context_vector

# Model building
input_layer = Input(shape=(200,))
embedding_layer = Embedding(10000, 128, input_length=200)(input_layer)
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
attention = attention_layer(lstm_layer, 128)
dropout = Dropout(0.5)(attention)
output_layer = Dense(1, activation='sigmoid')(Flatten()(dropout))
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))

# Save the model
model.save('bi_lstm_attention_model_with_negation_handling1.keras')

# Plotting training and validation error
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
