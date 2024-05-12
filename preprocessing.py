#%%
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import os
import spacy
import string
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Load the pre-trained Word2Vec model
model_path = "path_to_pretrained_model"  # Replace 'path_to_pretrained_model' with the actual path to the model file
word2vec_model = Word2Vec.load(model_path)


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

nlp = spacy.load("en_core_web_sm")

# functions for the pipeline
def emoji_to_name(emoji):
    return emoji.demojize(emoji).replace(":", "").replace("_", " ") # TODO check if should replace the lowercase by spaces


def clean_text(data): 
    data_cleaned = [string.replace("@user ", "") for string in data]
    data_cleaned = [string.replace("u2019", "’") for string in data_cleaned]
    data_cleaned = [string.replace("u002c", ",") for string in data_cleaned]
    
    return data_cleaned


def lowercase(data):
    return [string.lower() for string in data]


def remove_punctuation(data):
    translation_table = str.maketrans("", "", string.punctuation)
    return [string.translate(translation_table) for string in data]


def vectorize(data, vectorizer_name="tfidf", **kwargs):
    if vectorizer_name == "count":
        return CountVectorizer(**kwargs).fit_transform(data)
    elif vectorizer_name == "spacy": 
        matrix = [nlp(doc).vector_norm for doc in data]
        return matrix 
    elif vectorizer_name == "word2vec":
        matrix = [word2vec_model[word] for doc in data for word in doc]
        return matrix 
    elif vectorizer_name == "tfidf": 
        return TfidfVectorizer(**kwargs).fit_transform(data)
    else:
        return TfidfVectorizer(**kwargs).fit_transform(data)


def tokenize(data): 
    data = [word_tokenize(word_list) for word_list in data]
    return data


def lemmatizer(data):
    lemmatizer = nlp.copy().add_pipe("lemmatizer")
    data = [word.lemma_ for word_list in data for word in lemmatizer(word_list)]
    return data


def stopword_remover(data): 
    return [word for word_list in data for word in word_list if word not in stopwords.words('english')]


#%%
#define wrapper functions
clean_transformer = FunctionTransformer(clean_text)
lowercase_transformer = FunctionTransformer(lowercase)
punctuation_transformer = FunctionTransformer(remove_punctuation)
emoji_transformer = FunctionTransformer(emoji_to_name)
lemmatizer_transformer = FunctionTransformer(lemmatizer)
stopword_transformer = FunctionTransformer(stopword_remover)
tokenizer_transformer = FunctionTransformer(tokenize)
vectorizer_transformer = FunctionTransformer(vectorize)


#%%

PIPELINE_DICT = {
    'cleaner': clean_transformer,
    'lowercase': lowercase_transformer,
    'punctuation': punctuation_transformer,
    'emoji': emoji_transformer,
    'tokenizer': tokenizer_transformer,
    'lemmatizer': lemmatizer_transformer,
    'stopwords': stopword_transformer,
    'vectorizer': vectorizer_transformer
}

# create the pipeline
pipeline = Pipeline([
    ('cleaner', clean_transformer),
    ('lowercase', lowercase_transformer),
    ('punctuation', punctuation_transformer),
    ('emoji', emoji_transformer),
    ('tokenizer', tokenizer_transformer),
    ('lemmatizer', lemmatizer_transformer),
    ('stopwords', stopword_transformer)   
])

def create_pipeline(pipeline_dict: dict): 
    pipeline_elements_ls = []
    for k, args in pipeline_dict.items():
        pipeline_elements_ls.append(PIPELINE_DICT.get(k))
    
    pipeline = Pipeline(pipeline_elements_ls)
    return pipeline


#%%
folder_str =  'preprocessed_data'
data_files = {'train_data_raw': folder_str + '/preprocessed_train.txt',
            'validation_data_raw': folder_str + '/preprocessed_validation.txt',
            'test_data_raw': folder_str + '/preprocessed_test.txt'}


for file, export_path in data_files.items():#, test_data_raw]:
    tweets_preprocessed = pipeline.fit_transform(eval(file))

    with open(export_path, 'w', encoding='utf-8') as file:
        for item in tweets_preprocessed:
            file.write(str(item) + '\n')


# %%
print(tweets_preprocessed[23])

