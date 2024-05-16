#%%
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.metrics import accuracy_score, f1_score
from functions import load_data, load_datasets, VectorizerWrapper
from sklearn.model_selection import GridSearchCV

# # Paths to the data
# train_tweets_path = './preprocessed_data/preprocessed_train.txt'
# train_labels_path = './datasets/sentiment/train_labels.txt'
# val_tweets_path = './preprocessed_data/preprocessed_validation.txt'
# val_labels_path = './datasets/sentiment/val_labels.txt'
# test_tweets_path = './preprocessed_data/preprocessed_test.txt'
# test_labels_path = './datasets/sentiment/test_labels.txt'

# # Load datasets
# train_data = load_data(train_tweets_path, train_labels_path)
# val_data = load_data(val_tweets_path, val_labels_path)
# test_data = load_data(test_tweets_path, test_labels_path)

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


# Fit and transform the training data
X_train = vectorizer.fit_transform(train_data['tweet'])

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

val_accuracy = accuracy_score(val_data['label'], val_predictions)
print(f'Validation Accuracy: {val_accuracy:.2f}')
#%%
# Create and train the Naive Bayes model #TODO Why here another model fit?
model = MultinomialNB()
model.fit(X_train, train_data['label'])

# Predict on test data
test_predictions = model.predict(X_test)
test_f1 = f1_score(test_data['label'], test_predictions, average="weighted")
print(f'test F1: {test_f1:.2f}')

# %%
test_accuracy = accuracy_score(test_data['label'], test_predictions)
print(f'test Accuracy: {test_accuracy:.2f}')

#%% 
# grid search for Multinomial NB 
vectorizer_dict = {"tfidf": {'max_features': 20000, 'max_df':0.8}, "count": {'max_features': 20000, 'max_df':0.8}}
for vect_name, vect_args in vectorizer_dict.items(): 
    vectorizer = VectorizerWrapper(vectorizer_name=vect_name)

    X_train = vectorizer.fit_transform(train_data['tweet'], **vect_args)

    # Transform the validation and test data
    X_val = vectorizer.transform(val_data['tweet'])
    X_test = vectorizer.transform(test_data['tweet'])
    
    
    parameters = {'alpha': (0.0, 0.25, 0.5, 0.75, 1.0)}
    model = MultinomialNB()
    grid_clf = GridSearchCV(model, parameters, verbose= True)
    grid_clf.fit(X_train, train_data['label'])
    print(sorted(grid_clf.cv_results_.keys()))
    
#%% 
# grid search for Categorical NB 
vectorizer_dict = {"tfidf": {'max_features': 20000, 'max_df':0.8}, "count": {'max_features': 20000, 'max_df':0.8}}
for vect_name, vect_args in vectorizer_dict.items(): 
    vectorizer = VectorizerWrapper(vectorizer_name=vect_name)

    X_train = vectorizer.fit_transform(train_data['tweet'], **vect_args)

    # Transform the validation and test data
    X_val = vectorizer.transform(val_data['tweet'])
    X_test = vectorizer.transform(test_data['tweet'])
    
    
    parameters = {'alpha': (0.0, 0.25, 0.5, 0.75, 1.0)}
    model = CategoricalNB()
    grid_clf = GridSearchCV(model, parameters, verbose= True)
    grid_clf.fit(X_train, train_data['label'])
    print(sorted(grid_clf.cv_results_.keys()))    
