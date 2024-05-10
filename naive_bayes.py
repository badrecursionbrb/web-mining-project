#%%
# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#%%
# Load preprocessed tweets
def load_data(tweets_path, labels_path):
    with open(tweets_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = file.readlines()
        labels = [int(label.strip()) for label in labels]
    return pd.DataFrame({'tweet': tweets, 'label': labels})

# Paths to the data
train_tweets_path = './preprocessed_data/preprocessed_train.txt'
train_labels_path = './datasets/sentiment/train_labels.txt'
val_tweets_path = './preprocessed_data/preprocessed_validation.txt'
val_labels_path = './datasets/sentiment/val_labels.txt'
test_tweets_path = './preprocessed_data/preprocessed_test.txt'
test_labels_path = './datasets/sentiment/test_labels.txt'

# Load datasets
train_data = load_data(train_tweets_path, train_labels_path)
val_data = load_data(val_tweets_path, val_labels_path)
test_data = load_data(test_tweets_path, test_labels_path)

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
val_accuracy = accuracy_score(val_data['label'], val_predictions)
print(f'Validation Accuracy: {val_accuracy:.2f}')
