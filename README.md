# IMDB Sentiment Analysis with LSTM

Compares five machine learning approaches for classifying IMDB movie reviews as positive or negative, from simple logistic regression to Bi-LSTM with custom attention.

## What This Does

Takes 50,000 IMDB movie reviews and trains five different models to predict sentiment (positive/negative). Each model represents a different level of complexity, making it easy to see how deep learning architectures improve on traditional ML for text classification.

## How It Works

### Models (simplest to most complex)

| Model | Approach | Accuracy |
|-------|----------|----------|
| Naive Bayes | TF-IDF + Multinomial NB | 84.9% |
| Logistic Regression | TF-IDF (200 features) + LR | 90.9% |
| LSTM | Embedding (128d) + LSTM (128 units) | 94.6% |
| Bi-LSTM | Embedding (128d) + Bidirectional LSTM (64 units) | 94.3% |
| Bi-LSTM + Attention | Embedding (128d) + Bi-LSTM + custom attention layer + negation handling | 96.5% |

### Pipeline

```
IMDB_Dataset.csv (50,000 reviews)
    |
    v
Preprocessing: Tokenize (5,000-10,000 words), pad to 200 tokens
    |
    +--> TF-IDF path --> Logistic Regression, Naive Bayes
    |
    +--> Embedding path --> LSTM, Bi-LSTM, Bi-LSTM + Attention
    |
    v
Evaluation: Accuracy, Precision, Recall, F1 Score
```

### Key Techniques

- **Keras Tokenizer** with vocabulary cap (5,000-10,000 words) and sequence padding (maxlen=200)
- **TF-IDF Vectorization** (200 features) for traditional ML baselines
- **Custom Attention Layer** that learns which words matter most for sentiment
- **Negation Handling** — rewrites "not bad" as "not_bad" so the model treats it as a single positive token instead of two separate words
- **Bidirectional LSTM** reads reviews forward and backward to capture context in both directions

## Sample Output

### Model Comparison
```
LSTM             - Accuracy: 0.9465, Precision: 0.9423, Recall: 0.9511, F1: 0.9467
Bi-LSTM          - Accuracy: 0.9434, Precision: 0.9219, Recall: 0.9689, F1: 0.9448
Bi-LSTM+Attn     - Accuracy: 0.9645, Precision: 0.9561, Recall: 0.9737, F1: 0.9648
Logistic Reg.    - Accuracy: 0.9088, Precision: 0.9024, Recall: 0.9167, F1: 0.9095
Naive Bayes      - Accuracy: 0.8486, Precision: 0.8398, Recall: 0.8617, F1: 0.8506
```

### Streamlit App
```
Enter a review: "This movie was absolutely terrible and a waste of time"
Prediction: Negative Sentiment

Enter a review: "A masterpiece of storytelling with incredible performances"
Prediction: Positive Sentiment
```

## Files

```
LSTM/
├── reviews.py                  # LSTM training (128-unit, 5 epochs)
├── reviews_bi-LSTM.py          # Bi-LSTM training (64-unit bidirectional)
├── reviews_ANN.py              # Bi-LSTM + Attention with negation handling
├── reviews_lg.py               # Logistic Regression + Streamlit app
├── reviews_naiyes_bayes.py     # Naive Bayes training
├── Evaluation_metrics.py       # Side-by-side comparison of all 5 models
├── Common_Streamlit.py         # Unified Streamlit app for all models
├── test2.py                    # Quick inference test script
├── IMDB_Dataset.csv            # 50,000 labeled reviews
├── *.keras, *.h5               # Saved model weights
├── *.pkl, *.pickle             # Saved tokenizers and vectorizers
└── README.md
```

## Quick Start

```bash
pip install tensorflow pandas scikit-learn streamlit nltk matplotlib

# Train the LSTM model
python reviews.py

# Train Bi-LSTM + Attention
python reviews_ANN.py

# Compare all models
python Evaluation_metrics.py

# Launch the Streamlit app
streamlit run reviews_lg.py
```

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- scikit-learn, pandas, nltk, matplotlib, streamlit

## Dataset

- **IMDB Dataset**: 50,000 movie reviews (25,000 positive, 25,000 negative)
- Source: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Note

This is the original development version of the project. A restructured version with proper configuration, relative paths, and environment management is available at [sentiment-analysis](https://github.com/mihikap01/sentiment-analysis).
