
#%%
import numpy as np

from textblob import TextBlob
from functions import load_datasets, analyze_model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class TextBlobWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, positive_boundary=0.25, negative_boundary=-0.1) -> None:
        self.positive_boundary = positive_boundary
        self.negative_boundary = negative_boundary

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        return self

    def predict(self, data):
        prediction_ls = []
        polarity_ls = self.calculate_polarity(data=data)
        for pol_ls in polarity_ls:
            pol = pol_ls[0]
            if pol >= self.positive_boundary:
                prediction_ls.append(2) # positive 
            elif pol <= self.negative_boundary:
                prediction_ls.append(0) # negative
            else: # neutral
                prediction_ls.append(1) 
        return prediction_ls
    
    def calculate_polarity(self, data) -> list:
        polarity_ls = []
        for doc in data: 
            res = TextBlob(doc)
            pol = res.sentiment.polarity
            polarity_ls.append([pol])
        return polarity_ls


train_data, val_data, test_data = load_datasets(use_joined=True)

text_blob_model = TextBlobWrapper()

# X_train = vectorizer.fit_transform(train_data['tweet'])
X_train = train_data['tweet']

# Transform the validation and test data
X_val = val_data['tweet']
X_test = test_data['tweet']

train_labels = train_data['label']
val_labels = val_data['label']
test_labels = test_data['label']


#%%
# Analyze the model with custom set decision boundaries (found by manual inspection)
results = analyze_model(model=text_blob_model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)

#%% Analyze model again with equal sized intervals for decision boundaries
text_blob_model = TextBlobWrapper(positive_boundary=0.33, negative_boundary=-0.33)
results = analyze_model(model=text_blob_model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)




#%%
from sklearn.naive_bayes import GaussianNB

X_train_polarity = text_blob_model.calculate_polarity(X_train)
X_val_polarity = text_blob_model.calculate_polarity(X_val)
X_test_polarity = text_blob_model.calculate_polarity(X_test)

gnb = GaussianNB()
gnb.fit(X_train_polarity, train_labels)
y_train_pred = gnb.predict(X_train_polarity)
y_val_pred = gnb.predict(X_val_polarity)
y_test_pred = gnb.predict(X_test_polarity)

results = analyze_model(model=gnb, X_val=X_val_polarity, val_labels=val_labels, X_test=X_test_polarity, test_labels=test_labels)



# %%
print(gnb.get_params())

# %%

decision_polarity_boundary = np.arange(-1, 1.05, 0.01).reshape(-1, 1)

y_pred_boundary = gnb.predict(decision_polarity_boundary)

for i in range(len(decision_polarity_boundary)):
    print(decision_polarity_boundary[i], y_pred_boundary[i])

# %%
