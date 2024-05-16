#%%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors  # For Word2Vec

from functions import load_datasets

train_data, val_data, test_data = load_datasets()

#%%
def get_vectorizer(vectorization_type):
    if vectorization_type == 'tfidf':
        return TfidfVectorizer(max_features=1000)
    elif vectorization_type == 'count':
        return CountVectorizer(max_features=1000)
    elif vectorization_type == 'word2vec':
        # Load pre-trained Word2Vec model (this path needs to be adjusted to your actual model path)
        word_vectors = KeyedVectors.load_word2vec_format('path_to_word2vec.bin', binary=True)
        return word_vectors
    else:
        raise ValueError("Unsupported vectorization type specified.")

def vectorize_data(data, vectorizer):
    if isinstance(vectorizer, TfidfVectorizer) or isinstance(vectorizer, CountVectorizer):
        return vectorizer.transform(data['tweet'])
    elif isinstance(vectorizer, KeyedVectors):  # Handling Word2Vec
        # Transform each tweet to an average Word2Vec vector
        def tweet_to_vector(tweet):
            words = tweet.split()
            word_vectors = [vectorizer[word] for word in words if word in vectorizer]
            if not word_vectors:
                return np.zeros(vectorizer.vector_size)
            return np.mean(word_vectors, axis=0)
        return np.array([tweet_to_vector(tweet) for tweet in data['tweet']])
    else:
        raise ValueError("Unsupported vectorizer instance.")

# Example usage
vectorization_type = 'count'  # Change to 'tfidf', 'count', or 'word2vec'
vectorizer = get_vectorizer(vectorization_type)
vectorizer.fit(train_data['tweet'])

# Fit and transform data
X_train = vectorize_data(train_data, vectorizer)
X_val = vectorize_data(val_data, vectorizer)
X_test = vectorize_data(test_data, vectorizer)

#%%
# Create and train the Naive Bayes model
model = MultinomialNB() if vectorization_type != 'word2vec' else LogisticRegression()
model.fit(X_train, train_data['label'])

# Predict on validation data
val_predictions = model.predict(X_val)
val_f1 = f1_score(val_data['label'], val_predictions, average="weighted")
print(f'Validation F1: {val_f1:.2f}')

#%%
# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, train_data['label'])

# Predict on validation data
test_predictions = model.predict(X_test)
test_f1 = f1_score(test_data['label'], test_predictions, average="weighted")
print(f'Validation F1: {test_f1:.2f}')

# %%
