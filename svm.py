#%%
# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from functions import load_datasets, VectorizerWrapper, analyze_model, meta_grid_search, write_to_file

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
model = SVC()
model.fit(X_train, train_data['label'])

#%%
# Analyze the model 
analyze_model(model=model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)

#%%
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear']  # Type of kernels
}
model = SVC()
grid_clf = GridSearchCV(model, parameters, cv=5, verbose=2, scoring='f1_weighted')
grid_clf.fit(X_train, train_data['label'])
print(sorted(grid_clf.cv_results_.keys()))

best_estimator = grid_clf.best_estimator_

best_params = grid_clf.best_params_
estimator_name = best_estimator.__class__.__name__
write_to_file(estimator_name=estimator_name, vect_name=vectorizer_name, best_params=best_params, analyze_results={"metric": grid_clf.best_score_}, params_grid=parameters)

#%%
# evaluate best model
vectorizer_dict = {"tfidf": {'max_features': 7000, 'max_df':0.8}, "count": {'max_features': 7000, 'max_df':0.8}, "word2vec": {}, "fasttext": {}, "spacy": {}}
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear']  # Type of kernels
}
grid_search_result = meta_grid_search(model=model, vectorizer_dict=vectorizer_dict, parameters=parameters, 
                            train_data=train_data, val_data=val_data, test_data=test_data)
# %%
