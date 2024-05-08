#%%
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import os
import spacy
import string


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
test_data_raw = open(test_text_path, encoding='utf-8').read().split("\n")[:-1]

train_labels = [int(x) for x in train_labels_str]
validation_labels = [int(x) for x in validation_labels_str]
test_labels_gold = [int(x) for x in test_labels_str]

#%%

# functions for the pipeline

def clean_text(data): 
    data_cleaned = [string.replace("@user ", "")for string in data]
    data_cleaned = [string.replace("u2019", "")for string in data_cleaned]
    return data_cleaned


def lowercase(data):
    return [string.lower() for string in data]


def remove_punctuation(data):
    translation_table = str.maketrans("", "", string.punctuation)
    return [string.translate(translation_table) for string in data]




#%%
#define wrapper functions
clean_transformer = FunctionTransformer(clean_text)
lowercase_transformer = FunctionTransformer(lowercase)
punctuation_transformer = FunctionTransformer(remove_punctuation)


#%%
# create the pipeline
pipeline = Pipeline([
('Cleaner', clean_transformer),
('Lowercase', lowercase_transformer),
('Punctuation', punctuation_transformer)    
])


#%%


data_files = {'train_data_raw': 'preprocessed_data/preprocessed_train.txt',
              'validation_data_raw': 'preprocessed_data/preprocessed_validation.txt',
              'test_data_raw': 'preprocessed_data/preprocessed_test.txt'}



for file, export_path in data_files.items():#, test_data_raw]:
    tweets_preprocessed = pipeline.fit_transform(eval(file))

    with open(export_path, 'w', encoding='utf-8') as file:
        for item in tweets_preprocessed:
            file.write(str(item) + '\n')




# %%
print(tweets_preprocessed[23])

# %%
export_path = 'preprocessed_data/preprocessed_train.txt'

with open(export_path, 'w') as file:
    for item in tweets_preprocessed:
        file.write(str(item) + '\n')
