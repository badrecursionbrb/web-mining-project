#%%

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from functions import load_data, load_datasets, VectorizerWrapper, analyze_model

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV



#%%
vectorizer_name = "word2vec"
vectorizer = VectorizerWrapper(vectorizer_name=vectorizer_name)

train_data, val_data, test_data = load_datasets(vectorizer_name=vectorizer_name)

X_train = vectorizer.fit_transform(train_data['tweet'])

# Transform the validation and test data
X_val = vectorizer.transform(val_data['tweet'])
X_test = vectorizer.transform(test_data['tweet'])
train_labels = train_data['label']
val_labels = val_data['label']
test_labels = test_data['label']

#%%
# Create and train the Logistic Regression model
model = LogisticRegression(multi_class="ovr")
model.fit(X_train, train_data['label'])

#%%
# Analyze the model 
analyze_model(model=model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)



#%% 
parameters = {'multi_class':('ovr', 'multinomial'), 'penalty': ('l1', 'l2', 'elasticnet'), 'solver': ('newton-cg', 'sag', 'saga', 'lbfgs'), 'max_iter': (1000,)}

grid_clf = GridSearchCV(model, parameters, verbose=True)
grid_clf.fit(X_train, train_data['label'])
print(sorted(grid_clf.cv_results_.keys()))


# %%

vectorizer_dict = {"tfidf": {'max_features': 20000, 'max_df':0.8}, "count": {'max_features': 20000, 'max_df':0.8}}
for vect_name, vect_args in vectorizer_dict.items(): 
    vectorizer = VectorizerWrapper(vectorizer_name=vect_name)
    
    X_train = vectorizer.fit_transform(train_data['tweet'], **vect_args)

    # Transform the validation and test data
    X_val = vectorizer.transform(val_data['tweet'])
    X_test = vectorizer.transform(test_data['tweet'])
    
    parameters = {'multi_class':('ovr', 'multinomial'), 'penalty': ('l1', 'l2', 'elasticnet'), 'solver': ('newton-cg', 'sag', 'saga', 'lbfgs'), 'max_iter': (1000,)}

    grid_clf = GridSearchCV(model, parameters, verbose= True)
    grid_clf.fit(X_train, train_data['label'])
    print(sorted(grid_clf.cv_results_.keys()))
    
    best_estimator = grid_clf.best_estimator_



