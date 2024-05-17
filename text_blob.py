
from textblob import TextBlob
from functions import load_datasets, analyze_model

class TextBlobWrapper():

    def __init__(self) -> None:
        pass

    def predict(data):
        prediction_ls = []
        for doc in data:
            res = TextBlob(doc)
            
            pol = res.sentiment.polarity
            if pol >= 0.25:
                prediction_ls.append(2) # positive 
            elif pol <= -0.1:
                prediction_ls.append(0) # negative
            else: # neutral
                prediction_ls.append(1) 
        return prediction_ls



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
# Create and train the Naive Bayes model


#%%
# Analyze the model 
results = analyze_model(model=text_blob_model, X_val=X_val, val_labels=val_labels, X_test=X_test, test_labels=test_labels)





