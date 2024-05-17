#%%
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from functions import load_data, load_datasets, VectorizerWrapper, analyze_model, meta_grid_search, write_to_file
from sklearn.model_selection import GridSearchCV

#%%
vectorizer_name = "tfidf"
vectorizer = VectorizerWrapper(vectorizer_name=vectorizer_name)

train_data, val_data, test_data = load_datasets(vectorizer_name=vectorizer_name)

X_train = vectorizer.fit_transform(train_data['tweet'], max_features=10000)

# Transform the validation and test data
X_val = vectorizer.transform(val_data['tweet'])
X_test = vectorizer.transform(test_data['tweet'])
train_labels = train_data['label']
val_labels = val_data['label']
test_labels = test_data['label']

#%%
# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, train_data['label'])

#%%
# Analyze the model 
analyze_model(model=model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)

#%% 
parameters = {'alpha': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}

model = MultinomialNB()
grid_clf = GridSearchCV(model, parameters, verbose=True, scoring='accuracy')
grid_clf.fit(X_train, train_data['label'])
print(sorted(grid_clf.cv_results_.keys()))

best_estimator = grid_clf.best_estimator_

best_params = grid_clf.best_params_
estimator_name = best_estimator.__class__.__name__
write_to_file(estimator_name=estimator_name, vect_name=vectorizer_name, best_params=best_params, analyze_results={"metric": grid_clf.best_score_}, params_grid=parameters)

#%% 
# grid search for Multinomial NB 
vectorizer_dict = {"tfidf": {'max_features': 7000, 'max_df':0.8}, "count": {'max_features': 7000, 'max_df':0.8}}
parameters = {'alpha': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
model = MultinomialNB()

grid_search_result = meta_grid_search(model=model, vectorizer_dict=vectorizer_dict, parameters=parameters, 
                            train_data=train_data, val_data=val_data, test_data=test_data)

#%% 
# grid search for Categorical NB 
vectorizer_dict = {"tfidf": {'max_features': 7000, 'max_df':0.8}, "count": {'max_features': 7000, 'max_df':0.8}}
parameters = {'alpha': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
model = CategoricalNB()

grid_search_result = meta_grid_search(model=model, vectorizer_dict=vectorizer_dict, parameters=parameters, 
                            train_data=train_data, val_data=val_data, test_data=test_data)


#%% 
# grid search for Gaussian NB 
vectorizer_dict = {"word2vec": {}, "fasttext": {}, "spacy": {}}
parameters = {'alpha': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
model = GaussianNB()

grid_search_result = meta_grid_search(model=model, vectorizer_dict=vectorizer_dict, parameters=parameters, 
                            train_data=train_data, val_data=val_data, test_data=test_data)



