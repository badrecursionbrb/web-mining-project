#%%
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from functions import load_datasets, VectorizerWrapper, analyze_model, meta_grid_search, write_to_file, print_scores
from sklearn.model_selection import GridSearchCV

import copy


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
grid_clf = GridSearchCV(model, parameters, verbose=3, scoring='f1_weighted', return_train_score=True, n_jobs=-1)
grid_clf.fit(X_train, train_data['label'])

#%%
print(sorted(grid_clf.cv_results_.keys()))

best_estimator = grid_clf.best_estimator_

best_params = grid_clf.best_params_
estimator_name = best_estimator.__class__.__name__

write_to_file(estimator_name=estimator_name, vect_name=vectorizer_name, best_params=best_params, analyze_results={"metric": grid_clf.best_score_}, params_grid=parameters, grid_clf=grid_clf)

#%% 
# grid search for Multinomial NB 
vectorizer_dict = {"tfidf": {'max_features': 7000, 'max_df':0.8}, "count": {'max_features': 7000, 'max_df':0.8}}
parameters = {'alpha': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
model = MultinomialNB()

grid_search_result = meta_grid_search(model=model, vectorizer_dict=vectorizer_dict, parameters=parameters, 
                            train_data=train_data, val_data=val_data, test_data=test_data)


#%% 
# grid search for Gaussian NB 
vectorizer_dict = {"word2vec": {}, "fasttext": {}, "spacy": {}}
parameters = {'var_smoothing': (1e-10, 1e-9, 1e-8)}
model = GaussianNB()

grid_search_result = meta_grid_search(model=model, vectorizer_dict=vectorizer_dict, parameters=parameters, 
                            train_data=train_data, val_data=val_data, test_data=test_data)


# %%


vectorizer_name = "count"
vectorizer = VectorizerWrapper(vectorizer_name=vectorizer_name)
model = MultinomialNB(alpha=0.9)

train_data, val_data, test_data = load_datasets(vectorizer_name=vectorizer_name)
train_labels = train_data['label']
val_labels = val_data['label']
test_labels = test_data['label']

max_features_ls, max_df_ls = [1000, 5000, 7000, 12000, 20000, 30000], [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
print(vectorizer_name + ", max_features:" + str(max_features_ls) + ", max_df:" + str(max_df_ls))

highest_score = 0
best_features = {}
results_ls = []
for max_features in max_features_ls:
    for max_df in max_df_ls: 
        vect_tmp = copy.deepcopy(vectorizer)
        model_tmp = copy.deepcopy(model)

        print("fitting a model for: max_features: {} max_df: {}".format(max_features, max_df))

        X_train = vect_tmp.fit_transform(train_data['tweet'], max_features=max_features, max_df=max_df)
        X_val = vect_tmp.transform(val_data['tweet'])
        X_test = vect_tmp.transform(test_data['tweet'])
        model_tmp.fit(X_train, train_data['label'])

        train_pred = model_tmp.predict(X_train)
        print_scores(s="train", y_true=train_labels, y_pred=train_pred)

        results = analyze_model(model=model_tmp, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels, plot_conf_mats=False)
        cur_score = results.get("val_f1")
        if cur_score > highest_score:
            highest_score = cur_score
            best_features = {'max_features': max_features, 'max_df': max_df}
        results_ls.append([max_features, max_df, results])

#%%
print(highest_score)
print(best_features)

for result in results_ls:
    print(result)


#%% [markdown]
### Fitting the final Multinomial model 7000 model with max_df=0.2

#%%
vectorizer_name = "count"
vectorizer = VectorizerWrapper(vectorizer_name=vectorizer_name)

train_data, val_data, test_data = load_datasets(vectorizer_name=vectorizer_name)

X_train = vectorizer.fit_transform(train_data['tweet'], max_features=7000, max_df=0.2)

# Transform the validation and test data
X_val = vectorizer.transform(val_data['tweet'])
X_test = vectorizer.transform(test_data['tweet'])
train_labels = train_data['label']
val_labels = val_data['label']
test_labels = test_data['label']

#%%
# Create and train the final Naive Bayes model
model = MultinomialNB(alpha=0.9)
model.fit(X_train, train_data['label'])

#%%
# Analyze the model 
results = analyze_model(model=model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels, plot_conf_mats=True)

print(results)



#%% [markdown]
### Fitting the final Gaussian model for word2vec and var_smoothing 1e-10

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
# Create and train the final Naive Bayes model
model = GaussianNB(var_smoothing= 1e-10)
model.fit(X_train, train_data['label'])

#%%
# Analyze the model 
results = analyze_model(model=model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels, plot_conf_mats=True)

print(results)

# %%
