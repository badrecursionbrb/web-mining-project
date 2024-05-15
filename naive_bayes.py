#%%
# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from functions import load_data, load_datasets

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
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the training data
X_train = vectorizer.fit_transform(train_data['tweet'])

# Transform the validation and test data
X_val = vectorizer.transform(val_data['tweet'])
X_test = vectorizer.transform(test_data['tweet'])

#%%
# Create and train the Naive Bayes model
model = MultinomialNB()
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
