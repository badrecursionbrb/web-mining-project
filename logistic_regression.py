#%%

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from functions import load_data, load_datasets, VectorizerWrapper, analyze_model, meta_grid_search, write_to_file

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

#%%
vectorizer_name = "tfidf"
vectorizer = VectorizerWrapper(vectorizer_name=vectorizer_name)

train_data, val_data, test_data = load_datasets(vectorizer_name=vectorizer_name)

X_train = vectorizer.fit_transform(train_data['tweet'], max_features=20000, max_df=0.8)

# Transform the validation and test data
X_val = vectorizer.transform(val_data['tweet'])
X_test = vectorizer.transform(test_data['tweet'])
train_labels = train_data['label']
val_labels = val_data['label']
test_labels = test_data['label']

#%%
# Create and train the Logistic Regression model

model = LogisticRegression(multi_class="ovr", penalty="l2", solver="sag", max_iter=1000)
model.fit(X_train, train_data['label'])

#%%
# Analyze the model 
results = analyze_model(model=model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)

#%% 
parameters = {'multi_class':('ovr', 'multinomial'), 'penalty': ('l1', 'l2', 'elasticnet'), 'solver': ('newton-cg', 'sag', 'saga', 'lbfgs'), 'max_iter': (1000,)}
parameters = {'multi_class':('ovr', 'multinomial'), 'max_iter': (1000, )}

grid_clf = GridSearchCV(model, parameters, verbose=3, return_train_score=True, scoring="f1_weighted", n_jobs=-1)
grid_clf.fit(X_train, train_data['label'])

#%%
print(sorted(grid_clf.cv_results_.keys()))

best_estimator = grid_clf.best_estimator_

best_params = grid_clf.best_params_
estimator_name = best_estimator.__class__.__name__
write_to_file(estimator_name=estimator_name, vect_name=vectorizer_name, best_params=best_params, analyze_results={"metric": grid_clf.best_score_}, params_grid=parameters, grid_clf=grid_clf)


# %%
# Run meta grid search with different vectorizers
#vectorizer_dict = {"tfidf": {'max_features': 5000, 'max_df':0.8}, "count": {'max_features': 5000, 'max_df':0.8}}
vectorizer_dict = {"word2vec": {}, "fasttext": {}, "spacy": {}, "tfidf": {'max_features': 7000, 'max_df':0.8}, "count": {'max_features': 7000, 'max_df':0.8}, }

#parameters = {'multi_class':('ovr', 'multinomial'), 'penalty': ('l2',), 'solver': ('lbfgs',), 'max_iter': (1000,)}
parameters = {'multi_class':('ovr', 'multinomial'), 'penalty': ('l1', 'l2', 'elasticnet'), 'solver': ('newton-cg', 'sag', 'saga', 'lbfgs'), 'max_iter': (1000,)}

model = LogisticRegression()
grid_search_result = meta_grid_search(model=model, vectorizer_dict=vectorizer_dict, parameters=parameters, 
                            train_data=train_data, val_data=val_data, test_data=test_data)

# %%
