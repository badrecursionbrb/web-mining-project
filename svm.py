#%%
# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
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
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear']  # Type of kernels
}

# Set up GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=2, scoring='f1_weighted', return_train_score=True)
grid_search.fit(X_train, train_data['label'])
print(sorted(grid_search.cv_results_.keys()))

#%%
# evaluate best model
best_svm = grid_search.best_estimator_
val_predictions = best_svm.predict(X_val)
val_f1_score = f1_score(val_data['label'], val_predictions, average='weighted')
print(f'Best SVM Validation F1 Score: {val_f1_score:.2f}')

#%%
# Final testing using F1 score
test_predictions = best_svm.predict(X_test)
test_f1_score = f1_score(test_data['label'], test_predictions, average='weighted')
print(f'Test F1 Score: {test_f1_score:.2f}')

# %%
# Output the parameters of the best model
best_params = grid_search.best_params_
print(f'Best Model Parameters: {best_params}')
# %%
