import pandas as pd
from gensim.models import Word2Vec, Doc2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import datapath
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
import numpy as np
import fasttext
import fasttext.util

SPACY_CORPUS = "en_core_web_sm"

TWEETS_FOLDER = './preprocessed_data'
LABELS_FOLDER = './datasets'

TRAIN_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_train'
TRAIN_LABELS_PATH = LABELS_FOLDER + '/sentiment/train_labels.txt'
VAL_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_validation'
VAL_LABELS_PATH = LABELS_FOLDER + '/sentiment/val_labels.txt'
TEST_TWEETS_PATH = TWEETS_FOLDER + '/preprocessed_test'
TEST_LABELS_PATH = LABELS_FOLDER + '/sentiment/test_labels.txt'

# Load preprocessed tweets
def load_data(tweets_path, labels_path):
    with open(tweets_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = file.readlines()
        labels = [int(label.strip()) for label in labels]
    return pd.DataFrame({'tweet': tweets, 'label': labels})

def load_datasets(use_joined=True): 
    train_path = TRAIN_TWEETS_PATH
    valid_path = VAL_TWEETS_PATH
    test_path = TEST_TWEETS_PATH
    if use_joined: 
        train_path = train_path + "_joined"
        valid_path = valid_path + "_joined"
        test_path = test_path + "_joined"
        
    train_path = train_path + ".txt"
    valid_path = valid_path + ".txt"
    test_path = test_path  + ".txt"
    train_frame = load_data(tweets_path=train_path, labels_path=TRAIN_LABELS_PATH)
    validation_frame = load_data(tweets_path=valid_path, labels_path=VAL_LABELS_PATH)
    test_frame = load_data(tweets_path=test_path, labels_path=TEST_LABELS_PATH)
    return train_frame, validation_frame, test_frame



class VectorizerWrapper():
    # TODO It is not beautiful but it works usually one would need the vectorizers to be fixed variables 
    # and the vectorizers implementing fit_transform and _transform
    sklearn_vectorizers = {"tfidf", "count"}
    
    def __init__(self, vectorizer_name="tfidf") -> None:
        self.vectorizer_name = vectorizer_name
        self.vectorizer = None
        self.dims = None
    
    def fit_transform(self, data, **kwargs): 
        if self.vectorizer_name == "count":
            self.vectorizer = CountVectorizer(**kwargs)
            return self.vectorizer.fit_transform(data)
        elif self.vectorizer_name == "spacy": 
            nlp = spacy.load(SPACY_CORPUS, disable=["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"])
            self.vectorizer = nlp
            return self.transform(data) 
        elif self.vectorizer_name == "word2vec":
            # Load the pre-trained Word2Vec model
            #model_path = "./datasets/glove.twitter.27B.25d.txt"  # Replace 'path_to_pretrained_model' with the actual path to the model file
            #word2vec_model =  KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)
            word2vec_model = api.load("glove-twitter-25")
            self.dims = 25
            # word2vec_model = Word2Vec.load(model_path)
            self.vectorizer = word2vec_model 
            return self.transform(data) 
        elif self.vectorizer_name == "doc2vec":
            def tagged_document(data_ls_ls):
                for i, word_list in enumerate(data_ls_ls):
                    yield TaggedDocument(word_list, [i])
            data = list(tagged_document(data))
            doc2vec_model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
            
            doc2vec_model.build_vocab(data)
            doc2vec_model.train(data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

            self.vectorizer = doc2vec_model
            return self.transform(data)
        elif self.vectorizer_name == "fasttext":
            model_path = "./datasets/cc.en.300.bin"
            fasttext_model = fasttext.load_model(model_path)
            self.dims= 100 # setting dims here manually 
            fasttext.util.reduce_model(fasttext_model, self.dims)
            print(fasttext_model.get_dimension())
            self.vectorizer = fasttext_model
            return self.transform(data)
        elif self.vectorizer_name == "tfidf": 
            self.vectorizer = TfidfVectorizer(**kwargs)
            return self.vectorizer.fit_transform(data)
        else:
            self.vectorizer = TfidfVectorizer(**kwargs)
            return self.vectorizer.fit_transform(data)
        
    def transform(self, data): 
        if self.vectorizer_name in self.sklearn_vectorizers: 
            return self.vectorizer.transform(data)
        elif self.vectorizer_name == "spacy": 
            print("using spacy")
            #return [self.vectorizer(doc).vector for doc in data]
            return [doc.vector for doc in self.vectorizer.pipe(data)]
        elif self.vectorizer_name == "word2vec": 
            print("using word2vec")
            matrix = []
            for doc in data: 
                word_vectors = []
                for word in doc:
                    try: 
                        word_vectors.append(self.vectorizer[word])
                    except KeyError:
                        pass
                #word_vectors = [self.vectorizer[word] for word in doc]
                if len(word_vectors) == 0: 
                    word_vectors.append([0] * self.dims)
                matrix.append(np.mean(word_vectors, axis=0))
            return matrix
        elif self.vectorizer_name == "doc2vec": 
            print("using doc2vec")
            matrix = [self.vectorizer.infer_vector(doc) for doc in data]
            return matrix
        elif self.vectorizer_name == "fasttext":
            print("using fasttext")
            matrix = []
            for doc in data: 
                word_vectors = [self.vectorizer.get_word_vector(word) for word in doc]
                matrix.append(np.mean(word_vectors, axis=0))
            return matrix 
        else: 
            return self.vectorizer.transform(data) 
        
        

