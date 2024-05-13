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
import re


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
spacy_corpus= "en_core_web_sm"


# functions for the pipeline
def replace_emoji(emoji_letter):
    return emoji.demojize(emoji_letter).replace(":", "").replace("_", "") # TODO check if should replace the lowercase by spaces

def emoji_to_name(data):
    data_cleaned = []
    for doc in data: 
        # doc_cleaned = []
        # for token in doc: 
        #     if emoji.is_emoji(token): 
        #         doc_cleaned.append(replace_emoji(token))
        #     else:
        #         doc_cleaned.append(doc_cleaned)
        cleaned_doc = [replace_emoji(token) if emoji.is_emoji(token) else token for token in doc]
        # data_cleaned.append([recognize_and_convert_emoji(token) for token in cleaned_doc])
        data_cleaned.append(cleaned_doc)
        # data_cleaned.append(doc_cleaned)
    return data_cleaned 


def clean_text(data): 
    data_cleaned = [string.replace("@user ", "").replace("u2019", "’").replace("u002c", ",") for string in data]

    return data_cleaned


def lowercase(data):
    data_cleaned = []
    for doc in data: 
        data_cleaned.append([token.lower() for token in doc])
    return data_cleaned


def remove_punctuation(data):
    translation_table = str.maketrans("", "", string.punctuation)
    data_cleaned = []
    for doc in data: 
        data_cleaned.append([str(token).translate(translation_table).replace("'", "").replace("’", "") for token in doc])
    return data_cleaned


def vectorize(data, vectorizer_name="tfidf", **kwargs):
    if vectorizer_name == "count":
        return CountVectorizer(**kwargs).fit_transform(data)
    elif vectorizer_name == "spacy": 
        nlp = spacy.load(spacy_corpus)
        matrix = [nlp(doc).vector_norm for doc in data]
        return matrix 
    elif vectorizer_name == "word2vec":
        # Load the pre-trained Word2Vec model
        model_path = "datasets/glove-twitter-27B-25d-w2v.txt"  # Replace 'path_to_pretrained_model' with the actual path to the model file
        word2vec_model = Word2Vec.load(model_path)
        matrix = [word2vec_model[word] for doc in data for word in doc]
        return matrix 
    elif vectorizer_name == "tfidf": 
        return TfidfVectorizer(**kwargs).fit_transform(data)
    else:
        return TfidfVectorizer(**kwargs).fit_transform(data)


def tokenize(data): 
    data = [word_tokenize(doc) for doc in data]
    return data


def lemmatizer_and_tokenizer(data):
    # using spacy for lemmatization - comes with tagger, parser for parsing the sentence structures,
    # sentence recognition and named entity recognition (ner)
    nlp = spacy.load(spacy_corpus, disable=['tok2vec']) 
    # nlp.add_pipe("tagger")
    # lemmatizer = nlp.add_pipe("lemmatizer")
    data_cleaned = []
    for doc in data: 
        data_cleaned.append([word.lemma_ if word.lemma_ else word for word in nlp(doc)])
    return data_cleaned


def stopword_remover(data): 
    stopwords_set = set(stopwords.words('english'))
    data_cleaned = []
    for doc in data: 
        data_cleaned.append([word for word in doc if word not in stopwords_set])
    return data_cleaned

def clean_empty_strings(data): 
    data_cleaned = []
    for doc in data: 
        data_cleaned.append([word for word in doc if str(word) and not str(word).isspace() and not str(word) == '' and not str(word).isnumeric()])
    return data_cleaned


#%%
#define wrapper functions
clean_transformer = FunctionTransformer(clean_text)
lowercase_transformer = FunctionTransformer(lowercase)
punctuation_transformer = FunctionTransformer(remove_punctuation)
emoji_transformer = FunctionTransformer(emoji_to_name)
lemmatizer_tokenizer_transformer = FunctionTransformer(lemmatizer_and_tokenizer)
stopword_transformer = FunctionTransformer(stopword_remover)
tokenizer_transformer = FunctionTransformer(tokenize)
vectorizer_transformer = FunctionTransformer(vectorize)
cleaner_empty_transformer = FunctionTransformer(clean_empty_strings)


#%%

PIPELINE_DICT = {
    'cleaner': clean_transformer,
    'tokenizer': tokenizer_transformer,
    'lemmatizer_tokenizer': lemmatizer_tokenizer_transformer,
    'punctuation': punctuation_transformer,
    'emoji': emoji_transformer,
    'lowercase': lowercase_transformer,
    'stopwords': stopword_transformer,
    'vectorizer': vectorizer_transformer,
    'emptycleaner': cleaner_empty_transformer
}

# create the pipeline
pipeline = Pipeline([
    ('cleaner', clean_transformer),
    ('lemmatizer_tokenizer', lemmatizer_tokenizer_transformer),
    ('emoji', emoji_transformer),
    ('punctuation', punctuation_transformer),
    ('lowercase', lowercase_transformer),
    ('stopwords', stopword_transformer),
    ('emptycleaner', cleaner_empty_transformer)   
], verbose=True)


def create_pipeline(pipeline_dict: dict={'cleaner': {}, 'lowercase': {}}): 
    pipeline_elements_ls = []
    for k, args in pipeline_dict.items():
        pipeline_elements_ls.append(PIPELINE_DICT.get(k))
    
    pipeline = Pipeline(pipeline_elements_ls)
    return pipeline


#%%
folder_str =  'preprocessed_data'
data_files = {
            'train_data_raw': folder_str + '/preprocessed_train',
            'validation_data_raw': folder_str + '/preprocessed_validation',
            'test_data_raw': folder_str + '/preprocessed_test'
            }


for file, export_path in data_files.items():#, test_data_raw]:
    print("processing file: ".format(str(export_path)))
    tweets_preprocessed = pipeline.fit_transform(eval(file))

    with open(export_path + '.txt', 'w', encoding='utf-8') as file:
        for item in tweets_preprocessed:
            file.write(str(item) + '\n')
    
    tweets_preprocessed_joined = []
    for doc in tweets_preprocessed:
        tweets_preprocessed_joined.append(" ".join(doc))
    with open(export_path + '_joined' + '.txt', 'w', encoding='utf-8') as file:
        for item in tweets_preprocessed_joined:
            file.write(str(item) + '\n')


# %%
print(tweets_preprocessed[23])

