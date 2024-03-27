
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
dataset_path = "datasets"
task_path = "sentiment"

test_name =  "test_text.txt"
test_labels_gold_name  = "test_labels.txt"

gold_test_labels_path = os.path.join(dataset_path, task_path, test_labels_gold_name)
test_text_path = os.path.join(dataset_path, task_path, test_name)

test_labels_str = open(gold_test_labels_path).read().split("\n")[:-1]
test_data_raw = open(test_text_path).read().split("\n")[:-1]

test_labels_gold = [int(x) for x in test_labels_str]

#%%

def preprocess_tweets(s: str) -> str:
    s = s.replace("@user", "")
    return s

test_data = [preprocess_tweets(s) for s in test_data_raw]


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

# %%
