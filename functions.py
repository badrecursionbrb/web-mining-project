import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy

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
    
    def fit_transform(self, data, **kwargs): 
        if self.vectorizer_name == "count":
            self.vectorizer = CountVectorizer(**kwargs)
            return self.vectorizer.fit_transform(data)
        elif self.vectorizer_name == "spacy": 
            nlp = spacy.load(SPACY_CORPUS)
            self.vectorizer = nlp
            return self.transform(data) 
        elif self.vectorizer_name == "word2vec":
            # Load the pre-trained Word2Vec model
            model_path = "datasets/glove-twitter-27B-25d-w2v.txt"  # Replace 'path_to_pretrained_model' with the actual path to the model file
            word2vec_model = Word2Vec.load(model_path)
            self.vectorizer = word2vec_model
            matrix = []
            for doc in data: 
                
            matrix = [self.vectorizer[word] for doc in data for word in doc]
            return matrix 
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
            return [self.vectorizer(doc) for doc in data]
        elif self.vectorizer_name == "word2vec": 
            
        else: 
            return self.vectorizer.transform(data) 
        
        
def transform(data, vectorizer_name="tfidf", **kwargs):
    if vectorizer_name == "count":
        return CountVectorizer(**kwargs).fit_transform(data)
    elif vectorizer_name == "spacy": 
        nlp = spacy.load(SPACY_CORPUS)
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

