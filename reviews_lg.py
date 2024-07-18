
# Logistic Regression model predicts whether reviews are good or bad
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st


# Function to train and save the model
def train_and_save_model():
    df = pd.read_csv(
        '/Users/mihikapall/PycharmProjects/pythonProject/Python_Exercises/linear_regression/LSTM/IMDB_Dataset.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=200)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    log_reg = LogisticRegression()
    log_reg.fit(X_train_tfidf, y_train)
    pickle.dump(log_reg, open('log_reg_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))


# Check if the model and vectorizer are already saved
if not os.path.exists('log_reg_model.pkl') or not os.path.exists('vectorizer.pkl'):
    train_and_save_model()



# Streamlit app
def run_app():
    st.title('IMDB Review Sentiment Analysis')
    user_input = st.text_area("Enter a review:")

    if st.button('Predict'):
        log_reg = pickle.load(open('log_reg_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = log_reg.predict(user_input_tfidf)

        if prediction[0] == 1:
            st.write("Positive Sentiment")
        else:
            st.write("Negative Sentiment")


# Run the app
if __name__ == '__main__':
    run_app()
