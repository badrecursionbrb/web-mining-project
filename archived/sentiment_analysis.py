
#%%
from textblob import TextBlob
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import argparse
import os
import urllib.request
import csv

#%%
testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")

testimonial.sentiment

# %%
# download label mapping
mapping_link =  "datasets/sentiment/mapping.txt"
with open(mapping_link, "r") as file:
    content = file.read().split("\n")
    csvreader = csv.reader(content, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]
print(labels)

#%%
def predict(s: str): 
    res = TextBlob(s)
    
    pol = res.sentiment.polarity
    if pol >= 0.25:
        return 2 # positive 
    elif pol <= -0.1:
        return 0 # negative
    else: 
        return 1 # neutral
    # if pol >= 0.33:
    #     return 2 # positive
    # elif pol <= -0.33: 
    #     return 0 # negative
    # else: 
    #     return 1 # neutral


#%% 
import spacy 

# TODO select the components that are needed actually 
# nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger", "parser", "ner", "lemmatizer"])
nlp = spacy.load("en_core_web_sm")

def preprocess_w_spacy(s: str): 
    s = nlp(s)
    return s

def preprocess_list_w_spacy(str_list: list): 
    return nlp.pipe(str_list)

#%%
dataset_path = "datasets"
task_path = "sentiment"

test_name =  "test_text.txt"
train_name = "train_text.txt"
validation_name = "val_text.txt"

train_labels_name = "train_labels.txt"
validation_labels_name = "val_labels.txt"
test_labels_gold_name  = "test_labels.txt"

train_text_path = os.path.join(dataset_path, task_path, train_name)
validation_text_path = os.path.join(dataset_path, task_path, validation_name)
test_text_path = os.path.join(dataset_path, task_path, test_name)

train_labels_path = os.path.join(dataset_path, task_path, train_labels_name)
validation_labels_path = os.path.join(dataset_path, task_path, validation_labels_name)
gold_test_labels_path = os.path.join(dataset_path, task_path, test_labels_gold_name)


train_labels_str = open(train_labels_path).read().split("\n")[:-1]
validation_labels_str = open(validation_labels_path).read().split("\n")[:-1]
test_labels_str = open(gold_test_labels_path).read().split("\n")[:-1]

train_data_raw = open(train_text_path).read().split("\n")[:-1]
validation_data_raw = open(validation_text_path).read().split("\n")[:-1]
test_data_raw = open(test_text_path).read().split("\n")[:-1]

train_labels = [int(x) for x in train_labels_str]
validation_labels = [int(x) for x in validation_labels_str]
test_labels_gold = [int(x) for x in test_labels_str]


#%%

def clean_text(s: str): 
    return s.replace("@user", "")

def preprocess_tweets_old(s: str) -> str:
    s = clean_text(s)
    return s

def preprocess_tweets(ls: list):
    ls = [clean_text(s) for s in ls]
    ls = preprocess_list_w_spacy(ls) # returning an iterator (could do same for clean text)
    return ls


# %% 
train_data_preprocessed = preprocess_tweets(ls=train_data_raw) 
validation_data_preprocessed = preprocess_tweets(ls=validation_data_raw)
test_data_preprocessed = preprocess_tweets(ls=test_data_raw)

# %%
for i in range(10):
    x = next(train_data_preprocessed)
# Train the models here 

#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


sklearn_pipe = Pipeline(steps=[
    ('select', SelectKBest(k=2)),
    ('clf', LogisticRegression())])

# %% 

test_data = [preprocess_tweets_old(s) for s in test_data_raw]


# %%
test_data_sentiments = [predict(s) for s in test_data]

# %%

results = classification_report(test_labels_gold, test_data_sentiments, output_dict=True)

print(results)

tweeteval_result = results['macro avg']['recall']
print(tweeteval_result)

# %%

cm = confusion_matrix(test_labels_gold, test_data_sentiments)
print(cm)
cm_disp = ConfusionMatrixDisplay(cm).plot()

