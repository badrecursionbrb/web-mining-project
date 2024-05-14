


from textblob import TextBlob
from functions import load_datasets

def transform(s: str): 
    res = TextBlob(s)
    
    pol = res.sentiment.polarity
    if pol >= 0.25:
        return 2 # positive 
    elif pol <= -0.1:
        return 0 # negative
    else: 
        return 1 
    


train_data, val_data, test_data = load_datasets()

# X_train = vectorizer.fit_transform(train_data['tweet'])

# Transform the validation and test data
X_val = transform(val_data['tweet'])
X_test = transform(test_data['tweet'])