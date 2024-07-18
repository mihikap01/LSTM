import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
df = pd.read_csv('/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/IMDB_Dataset.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = nb_model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
pickle.dump(nb_model, open('naive_bayes_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer_nb.pkl', 'wb'))
