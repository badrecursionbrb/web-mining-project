#%%
# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from functions import load_data, load_datasets

# Paths to the data
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
svm_model = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 1, 10, 100],  # Kernel coefficient
    'kernel': ['rbf', 'poly', 'sigmoid']  # Type of kernel
}

# Set up GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=2, scoring='accuracy')
grid_search.fit(X_train, train_data['label'])

#%%
# evaluate best model
best_svm = grid_search.best_estimator_
val_predictions = best_svm.predict(X_val)
val_accuracy = accuracy_score(val_data['label'], val_predictions)
print(f'Best SVM Validation Accuracy: {val_accuracy:.2f}')

#%%
#final testing
test_predictions = best_svm.predict(X_test)
test_accuracy = accuracy_score(test_data['label'], test_predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')
# %%
