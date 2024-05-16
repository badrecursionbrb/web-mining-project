#%%
# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.metrics import accuracy_score
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
    
